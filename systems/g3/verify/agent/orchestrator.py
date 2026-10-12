"""
Stage 2 Orchestrator — submit-intercept framework with 4 experiment variants.

Architecture:
    Phase 1 agent runs with full budget.  When it calls submit(), the
    submit gate intercepts the call and hands control to variant-specific
    logic:

    Variant A (Clean Verifier):
        Fresh agent receives ONLY the task instruction + current env state.
        It verifies/fixes with remaining budget.

    Variant B (Reflexion Verifier):
        LLM generates a reflexion summary from Phase 1 trace.
        Fresh agent receives instruction + reflexion + current env state.

    Variant C (Judge → Resume):
        Judge LLM evaluates the work.  If APPROVED → submit.
        If NEEDS_FIX → feedback injected into a clone of the original
        agent's memory → agent continues with remaining budget.

    Variant D (Are-You-Sure Gate):
        Submit is intercepted N times.  Agent must verify its own work
        before the Nth+1 submit goes through.  No new LLM instance.

Key insight: /tests/run-tests.sh does NOT exist in the Docker container
during agent execution (the harness copies tests AFTER the agent ends).
Therefore all verification must be agent-driven, not test-driven.
"""
from __future__ import annotations

import copy
import json as _json_module
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from llm.base import BaseLLM
from terminal.session import SafeTerminal
from agent.executor import ToolExecutor
from agent.react import ReActAgent, AgentResult
from agent.memory import ConversationMemory, estimate_tokens


def _verdict_approved(text) -> bool:
    """Conservative robust check for an APPROVED judge verdict. Strips a leading markdown
    fence / quotes / whitespace, then requires the verdict to START with APPROVED. A
    genuine NEEDS_FIX never starts with APPROVED, so this NEVER flips a rejection into an
    approval — it only recovers a fenced/whitespace-prefixed APPROVED that the old strict
    ``startswith`` missed. Errs toward NEEDS_FIX (more verification): the safe bias."""
    if not text:
        return False
    t = text.strip()
    while t.startswith("`"):
        t = t.lstrip("`").strip()
        for tag in ("json", "text"):
            if t[:len(tag)].lower() == tag:
                t = t[len(tag):].strip()
    t = t.lstrip("\"'").strip()
    return t.upper().startswith("APPROVED")
from agent.tools import get_tool_schemas
from agent.prompts import (
    get_verifier_system_prompt,
    build_verifier_initial_message,
    REFLEXION_SYSTEM,
    build_reflexion_user_message,
    JUDGE_SYSTEM,
    build_judge_message,
)

logger = logging.getLogger(__name__)

# Minimum budget thresholds for the verifier phase
_MIN_VERIFIER_STEPS = 3
_MIN_VERIFIER_WALLCLOCK = 30.0  # seconds

# How many times variant D intercepts before allowing real submit
_VARIANT_D_MAX_INTERCEPTS = 2


@dataclass
class Stage2Result:
    """Outcome of a Stage 2 orchestrator run."""

    # Final agent result (the one that matters for TerminalBench)
    agent_result: AgentResult

    # Stage 2 metadata
    variant: str = ""
    phase1_steps: int = 0
    phase1_elapsed: float = 0.0
    was_intercepted: bool = False
    verifier_steps: int = 0
    verifier_elapsed: float = 0.0
    reflexion: Optional[str] = None
    judge_feedback: Optional[str] = None
    judge_approved: Optional[bool] = None
    intercept_count: int = 0

    # Phase 1 token accounting (not captured in agent_result which is the verifier's)
    phase1_input_tokens: int = 0
    phase1_output_tokens: int = 0
    reflexion_input_tokens: int = 0
    reflexion_output_tokens: int = 0


class Stage2Orchestrator:
    """
    Submit-intercept orchestrator for Stage 2 experiments.

    Wraps the Phase 1 ReAct agent with variant-specific verification logic.
    """

    def __init__(
        self,
        llm: BaseLLM,
        executor: ToolExecutor,
        terminal: SafeTerminal,
        config: Config,
        variant: str,
        supervisor_llm: BaseLLM = None,   # heterogeneous supervisor (verifier/judge role); None => same as llm
    ):
        self._llm = llm
        self._sup_llm = supervisor_llm if supervisor_llm is not None else llm
        self._executor = executor
        self._terminal = terminal
        self._config = config
        self._variant = variant.lower().strip()

        if self._variant not in ("a", "b", "c", "d"):
            raise ValueError(
                f"Invalid stage2 variant: '{self._variant}'. "
                f"Must be one of: a, b, c, d"
            )

        logger.info(f"Stage 2 orchestrator initialized | variant={self._variant}")

    @property
    def variant(self) -> str:
        return self._variant

    def run(
        self,
        instruction: str,
        initial_observation: str,
        env_info: str,
    ) -> Stage2Result:
        """
        Run the full Stage 2 pipeline for the given task.

        Returns:
            Stage2Result with the final AgentResult and metadata.
        """
        if self._variant == "d":
            return self._run_variant_d(instruction, initial_observation, env_info)
        else:
            return self._run_variant_abc(instruction, initial_observation, env_info)

    # ==================================================================
    # Variant D: Are-You-Sure Gate
    # ==================================================================

    def _run_variant_d(
        self,
        instruction: str,
        initial_obs: str,
        env_info: str,
    ) -> Stage2Result:
        """
        Variant D: intercept submit N times, agent must verify its own work.

        The executor is configured in 'continue' mode — the agent sees a
        "[HOLD] verify your work" message and keeps running.  After N
        intercepts, the next submit goes through.
        """
        self._executor.configure_submit_gate(
            max_intercepts=_VARIANT_D_MAX_INTERCEPTS,
            mode="continue",
        )

        agent = ReActAgent(
            llm=self._llm,
            executor=self._executor,
            config=self._config,
        )
        result = agent.run(
            instruction=instruction,
            initial_observation=initial_obs,
            env_info=env_info,
        )

        return Stage2Result(
            agent_result=result,
            variant="d",
            phase1_steps=result.steps,
            phase1_elapsed=result.elapsed_seconds,
            was_intercepted=self._executor._submit_intercept_count > 0,
            intercept_count=self._executor._submit_intercept_count,
        )

    # ==================================================================
    # Variants A/B/C: Submit-Intercept → Verifier/Resume
    # ==================================================================

    def _run_variant_abc(
        self,
        instruction: str,
        initial_obs: str,
        env_info: str,
    ) -> Stage2Result:
        """
        Variants A/B/C: Phase 1 agent runs, submit intercepted, then
        variant-specific verification/fix logic takes over.
        """
        # Phase 1: run with 1 submit intercept (stop mode)
        self._executor.configure_submit_gate(max_intercepts=1, mode="stop")

        phase1_agent = ReActAgent(
            llm=self._llm,
            executor=self._executor,
            config=self._config,
        )
        phase1_result = phase1_agent.run(
            instruction=instruction,
            initial_observation=initial_obs,
            env_info=env_info,
        )

        # If agent didn't get intercepted (hit budget, crashed, etc.)
        # just return the Phase 1 result as-is
        if not self._executor.submit_intercepted:
            logger.info(
                f"Phase 1 ended without submit intercept: "
                f"{phase1_result.termination_reason}"
            )
            return Stage2Result(
                agent_result=phase1_result,
                variant=self._variant,
                phase1_steps=phase1_result.steps,
                phase1_elapsed=phase1_result.elapsed_seconds,
                was_intercepted=False,
            )

        # Calculate remaining budget
        remaining_steps = self._config.agent.max_steps - phase1_result.steps
        remaining_time = (
            self._config.agent.max_wallclock_seconds - phase1_result.elapsed_seconds
        )

        logger.info(
            f"Submit intercepted at step {phase1_result.steps} | "
            f"remaining: {remaining_steps} steps, {remaining_time:.0f}s"
        )

        # Not enough budget for verification — force-submit the intercepted answer
        if remaining_steps < _MIN_VERIFIER_STEPS or remaining_time < _MIN_VERIFIER_WALLCLOCK:
            logger.warning(
                "Insufficient budget for verification — submitting intercepted answer"
            )
            self._executor._submitted = True
            self._executor._final_answer = self._executor.intercepted_answer
            phase1_result.submitted = True
            phase1_result.answer = self._executor.intercepted_answer
            phase1_result.termination_reason = "Submit intercepted (budget insufficient for verifier)"
            return Stage2Result(
                agent_result=phase1_result,
                variant=self._variant,
                phase1_steps=phase1_result.steps,
                phase1_elapsed=phase1_result.elapsed_seconds,
                was_intercepted=True,
            )

        # Dispatch to variant-specific logic
        if self._variant == "a":
            return self._verify_clean(
                instruction, env_info, phase1_result,
                remaining_steps, remaining_time,
            )
        elif self._variant == "b":
            return self._verify_reflexion(
                instruction, env_info, phase1_agent, phase1_result,
                remaining_steps, remaining_time,
            )
        elif self._variant == "c":
            return self._verify_and_resume(
                instruction, env_info, phase1_agent, phase1_result,
                remaining_steps, remaining_time,
            )
        else:
            raise ValueError(f"Unexpected variant: {self._variant}")

    # ------------------------------------------------------------------
    # Variant A: Clean Verifier
    # ------------------------------------------------------------------

    def _verify_clean(
        self,
        instruction: str,
        env_info: str,
        phase1_result: AgentResult,
        remaining_steps: int,
        remaining_time: float,
    ) -> Stage2Result:
        """Fresh agent with clean context (instruction + env only)."""
        current_obs = self._get_current_env_state()

        # Save Phase 1's intercepted answer before reset clears it
        phase1_answer = self._executor.intercepted_answer

        # Reset executor for verifier (clears submit state + tool caches)
        self._executor.reset_for_verifier()

        # Build verifier agent with verifier-specific prompts
        verifier = ReActAgent(
            llm=self._sup_llm,
            executor=self._executor,
            config=self._config,
        )

        # Override the memory with verifier system prompt
        from agent.memory import ConversationMemory
        from agent.tools import get_tool_schemas

        _TOOL_SCHEMA_ESTIMATED_TOKENS = 2000

        task_message = build_verifier_initial_message(
            instruction=instruction,
            current_obs=current_obs,
            env_info=env_info,
        )
        verifier._memory = ConversationMemory(
            system_prompt=get_verifier_system_prompt(),
            task_message=task_message,
            max_turns=self._config.agent.max_context_turns,
            max_output_chars=self._config.agent.max_output_chars,
            max_context_tokens=self._config.agent.max_context_tokens,
            summarizer_fn=verifier._make_summarizer(),
            overhead_tokens=_TOOL_SCHEMA_ESTIMATED_TOKENS,
        )

        verify_result = verifier.run(
            instruction=instruction,
            initial_observation=current_obs,
            env_info=env_info,
            max_steps_override=remaining_steps,
            max_wallclock_override=remaining_time,
            skip_initial_message=True,
        )

        # Use verifier's result as final if it submitted; else use phase1's
        final_result = verify_result if verify_result.submitted else phase1_result
        if not final_result.submitted:
            # Neither submitted — force-submit phase1's saved answer
            self._executor._submitted = True
            self._executor._final_answer = phase1_answer
            final_result = phase1_result
            final_result.submitted = True
            final_result.answer = phase1_answer

        return Stage2Result(
            agent_result=final_result,
            variant="a",
            phase1_steps=phase1_result.steps,
            phase1_elapsed=phase1_result.elapsed_seconds,
            was_intercepted=True,
            verifier_steps=verify_result.steps,
            verifier_elapsed=verify_result.elapsed_seconds,
            phase1_input_tokens=phase1_result.total_input_tokens,
            phase1_output_tokens=phase1_result.total_output_tokens,
        )

    # ------------------------------------------------------------------
    # Variant B: Reflexion Verifier
    # ------------------------------------------------------------------

    def _verify_reflexion(
        self,
        instruction: str,
        env_info: str,
        phase1_agent: ReActAgent,
        phase1_result: AgentResult,
        remaining_steps: int,
        remaining_time: float,
    ) -> Stage2Result:
        """Fresh agent with reflexion context verifies and fixes."""
        # Generate reflexion from Phase 1 trace
        reflexion, refl_tokens = self._generate_reflexion(instruction, phase1_result)
        logger.info(f"Generated reflexion ({len(reflexion)} chars): {reflexion[:150]}...")

        current_obs = self._get_current_env_state()

        # Save Phase 1's intercepted answer before reset clears it
        phase1_answer = self._executor.intercepted_answer

        # Reset executor for verifier (clears submit state + tool caches)
        self._executor.reset_for_verifier()

        # Build verifier agent with reflexion context
        verifier = ReActAgent(
            llm=self._sup_llm,
            executor=self._executor,
            config=self._config,
        )

        from agent.memory import ConversationMemory

        _TOOL_SCHEMA_ESTIMATED_TOKENS = 2000

        task_message = build_verifier_initial_message(
            instruction=instruction,
            current_obs=current_obs,
            env_info=env_info,
            reflexion=reflexion,
        )
        verifier._memory = ConversationMemory(
            system_prompt=get_verifier_system_prompt(),
            task_message=task_message,
            max_turns=self._config.agent.max_context_turns,
            max_output_chars=self._config.agent.max_output_chars,
            max_context_tokens=self._config.agent.max_context_tokens,
            summarizer_fn=verifier._make_summarizer(),
            overhead_tokens=_TOOL_SCHEMA_ESTIMATED_TOKENS,
        )

        verify_result = verifier.run(
            instruction=instruction,
            initial_observation=current_obs,
            env_info=env_info,
            max_steps_override=remaining_steps,
            max_wallclock_override=remaining_time,
            skip_initial_message=True,
        )

        final_result = verify_result if verify_result.submitted else phase1_result
        if not final_result.submitted:
            self._executor._submitted = True
            self._executor._final_answer = phase1_answer
            final_result = phase1_result
            final_result.submitted = True
            final_result.answer = phase1_answer

        return Stage2Result(
            agent_result=final_result,
            variant="b",
            phase1_steps=phase1_result.steps,
            phase1_elapsed=phase1_result.elapsed_seconds,
            was_intercepted=True,
            verifier_steps=verify_result.steps,
            verifier_elapsed=verify_result.elapsed_seconds,
            reflexion=reflexion,
            phase1_input_tokens=phase1_result.total_input_tokens,
            phase1_output_tokens=phase1_result.total_output_tokens,
            reflexion_input_tokens=refl_tokens[0],
            reflexion_output_tokens=refl_tokens[1],
        )

    # ------------------------------------------------------------------
    # Variant C: Judge → Resume Original Agent
    # ------------------------------------------------------------------

    def _verify_and_resume(
        self,
        instruction: str,
        env_info: str,
        phase1_agent: ReActAgent,
        phase1_result: AgentResult,
        remaining_steps: int,
        remaining_time: float,
    ) -> Stage2Result:
        """Judge evaluates; if NEEDS_FIX, original agent resumes with feedback."""
        # Get environment snapshot for judge
        env_snapshot = self._get_current_env_state()

        # Build trace summary for judge
        trace_summary = self._build_trace_summary(phase1_result)

        # Judge call
        judge_response = self._call_judge(
            instruction=instruction,
            trace_summary=trace_summary,
            env_snapshot=env_snapshot,
            intercepted_answer=self._executor.intercepted_answer or "",
        )

        if _verdict_approved(judge_response):
            logger.info("Judge APPROVED the solution — submitting")
            self._executor._submitted = True
            self._executor._final_answer = self._executor.intercepted_answer
            phase1_result.submitted = True
            phase1_result.answer = self._executor.intercepted_answer
            phase1_result.termination_reason = "Agent submitted (judge approved)"
            return Stage2Result(
                agent_result=phase1_result,
                variant="c",
                phase1_steps=phase1_result.steps,
                phase1_elapsed=phase1_result.elapsed_seconds,
                was_intercepted=True,
                judge_approved=True,
                judge_feedback=judge_response,
                phase1_input_tokens=phase1_result.total_input_tokens,
                phase1_output_tokens=phase1_result.total_output_tokens,
            )

        # Extract feedback from NEEDS_FIX response
        feedback = judge_response
        if feedback.upper().startswith("NEEDS_FIX:"):
            feedback = feedback[len("NEEDS_FIX:"):].strip()

        logger.info(f"Judge NEEDS_FIX: {feedback[:150]}...")

        # Save Phase 1's intercepted answer before reset clears it
        phase1_answer = self._executor.intercepted_answer

        # Reset executor for resumed agent
        self._executor.reset_submit_gate()

        # Create a new agent that inherits the original's memory
        resume_agent = ReActAgent(
            llm=self._llm,
            executor=self._executor,
            config=self._config,
        )
        # Deep-copy the original memory so we don't corrupt Phase 1's trace
        resume_agent._memory = copy.deepcopy(phase1_agent._memory)

        # Inject judge feedback as a system message
        resume_agent._memory.add_user_message(
            f"[SYSTEM] Your submission was reviewed by a judge and REJECTED.\n\n"
            f"Issues found:\n{feedback}\n\n"
            f"You have {remaining_steps} steps and {remaining_time:.0f}s remaining. "
            f"Fix the issues above and submit() again."
        )

        resume_result = resume_agent.run(
            instruction=instruction,
            max_steps_override=remaining_steps,
            max_wallclock_override=remaining_time,
            skip_initial_message=True,
        )

        final_result = resume_result if resume_result.submitted else phase1_result
        if not final_result.submitted:
            # Neither submitted — force-submit phase1's saved answer
            self._executor._submitted = True
            self._executor._final_answer = phase1_answer
            final_result = phase1_result
            final_result.submitted = True
            final_result.answer = phase1_answer

        return Stage2Result(
            agent_result=final_result,
            variant="c",
            phase1_steps=phase1_result.steps,
            phase1_elapsed=phase1_result.elapsed_seconds,
            was_intercepted=True,
            verifier_steps=resume_result.steps,
            verifier_elapsed=resume_result.elapsed_seconds,
            judge_approved=False,
            judge_feedback=feedback,
            phase1_input_tokens=phase1_result.total_input_tokens,
            phase1_output_tokens=phase1_result.total_output_tokens,
        )

    # ==================================================================
    # Helpers
    # ==================================================================

    def _get_current_env_state(self) -> str:
        """Capture a snapshot of the current environment."""
        parts: list[str] = []

        pwd_ls = self._terminal.exec("pwd && ls -la", timeout=10.0)
        parts.append(pwd_ls.output.strip())

        # Check /app contents (common workspace)
        app_ls = self._terminal.exec(
            "ls -la /app/ 2>/dev/null | head -30", timeout=5.0
        )
        if app_ls.output.strip():
            parts.append(f"\n/app/ contents:\n{app_ls.output.strip()}")

        return "\n".join(parts)

    def _build_trace_summary(self, result: AgentResult) -> str:
        """Build a concise summary of the agent's trace for reflexion/judge."""
        if not result.conversation_trace:
            return f"Agent ran for {result.steps} steps. No trace available."

        lines: list[str] = []
        for entry in result.conversation_trace:
            tool = entry.get("tool", "?")
            step = entry.get("step", "?")
            args = entry.get("args", {})
            result_preview = entry.get("result_preview", "")

            if tool == "bash":
                cmd = args.get("command", "")[:100]
                lines.append(f"Step {step}: bash({cmd})")
                if result_preview:
                    lines.append(f"  → {result_preview[:120]}")
            elif tool in ("write_file", "patch_file"):
                lines.append(f"Step {step}: {tool}({args.get('path', '?')})")
            elif tool == "think":
                lines.append(f"Step {step}: think({(args.get('content', '') or '')[:60]})")
            elif tool == "submit":
                lines.append(f"Step {step}: submit({(args.get('answer', '') or '')[:60]})")

        return "\n".join(lines)

    def _generate_reflexion(
        self,
        instruction: str,
        phase1_result: AgentResult,
    ) -> tuple[str, tuple[int, int]]:
        """Generate a reflexion summary from the Phase 1 trace (variant B).

        Returns:
            (reflexion_text, (input_tokens, output_tokens))
        """
        trace_summary = self._build_trace_summary(phase1_result)
        intercepted = self._executor.intercepted_answer or "(no answer)"

        user_msg = build_reflexion_user_message(
            instruction=instruction,
            trace_summary=trace_summary,
            intercepted_answer=intercepted,
        )

        response = self._sup_llm.chat(
            messages=[
                {"role": "system", "content": REFLEXION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            tools=None,
            temperature=0.0,
            max_tokens=250,
        )

        text = response.text.strip() if response.text else "No reflexion generated."
        tokens = (response.input_tokens, response.output_tokens)
        return text, tokens

    def _call_judge(
        self,
        instruction: str,
        trace_summary: str,
        env_snapshot: str,
        intercepted_answer: str,
    ) -> str:
        """Call the judge LLM to evaluate the solution (variant C)."""
        user_msg = build_judge_message(
            instruction=instruction,
            trace_summary=trace_summary,
            env_snapshot=env_snapshot,
            intercepted_answer=intercepted_answer,
        )

        response = self._sup_llm.chat(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            tools=None,
            temperature=0.0,
            max_tokens=200,
        )

        text = response.text.strip() if response.text else "APPROVED"
        logger.info(f"Judge response: {text[:200]}")
        return text
