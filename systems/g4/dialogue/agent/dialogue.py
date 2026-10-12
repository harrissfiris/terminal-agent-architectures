"""
Dialogue Orchestrator — G4 round-robin multi-agent system.

Two equal agents (A and B) take turns working on the same task.
Each agent gets a fixed number of steps per turn, then hands off
to the other with a compressed summary of what it did.

Variants (controlled by config.agent.dialogue_variant):
  cooperation — Agent B continues Agent A's work (relay/tag-team).
                Tests: does fresh context alone break stagnation?
  debate      — Agent B critiques Agent A's approach, then tries differently.
                Tests: does forced critique improve strategy selection?

Shared state: both agents operate on the SAME Docker terminal (same tmux
session), so file modifications by Agent A are visible to Agent B. This is
richer than text-only message passing — the agents share a real filesystem.

Literature basis:
  cooperation: CAMEL (Li et al., NeurIPS 2023), MoA (Wang et al., 2024)
  debate:      MAD (Liang et al., EMNLP 2024), Du et al. (NeurIPS 2023)
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from llm.base import BaseLLM
from terminal.session import SafeTerminal
from agent.executor import ToolExecutor
from agent.react import ReActAgent, AgentResult
from agent.prompts import (
    build_cooperation_handoff, build_debate_handoff,
    build_confirmation_handoff, build_evaluation_handoff,
    get_system_prompt,
)

logger = logging.getLogger(__name__)

# Agent labels for logging and handoff context
_AGENT_LABELS = ["Agent A", "Agent B"]


@dataclass
class TurnResult:
    """Outcome of a single agent turn within the dialogue."""

    agent_label: str
    turn_number: int
    agent_result: AgentResult
    action_log: str = ""
    last_reasoning: str = ""


@dataclass
class DialogueResult:
    """Outcome of the full dialogue orchestration."""

    # Final result (from whichever agent submitted, or the last one)
    agent_result: AgentResult

    # Per-turn metadata
    turns: list[TurnResult] = field(default_factory=list)

    # Orchestration metadata
    variant: str = ""
    total_turns: int = 0
    total_steps: int = 0
    total_elapsed: float = 0.0


def _build_action_log(conversation_trace: list[dict]) -> str:
    """
    Build a compact action log from a ReActAgent's conversation trace.

    Each step becomes one line, e.g.:
      Step 1: bash(ls -la /app/) → [file listing...]
      Step 2: read_file(/app/solve.py) → [file contents...]
      Step 3: think(...) → [OK] Reasoning noted.
    """
    lines: list[str] = []
    for entry in conversation_trace:
        step = entry.get("step", "?")
        tool = entry.get("tool", "?")
        args = entry.get("args", {})
        result_preview = (entry.get("result_preview", "") or "")[:300]

        # Build compact argument summary
        if tool == "bash":
            arg_str = args.get("command", "")[:200]
        elif tool in ("write_file", "patch_file", "read_file"):
            arg_str = args.get("path", "")
        elif tool == "think":
            arg_str = (args.get("content", "") or "")[:120]
        elif tool == "submit":
            arg_str = (args.get("answer", "") or "")[:80]
        else:
            arg_str = str(args)[:100]

        # Compact result
        result_str = result_preview.replace("\n", " ")[:200]

        lines.append(f"- Step {step}: {tool}({arg_str}) → {result_str}")

    return "\n".join(lines) if lines else "(no actions taken)"


def _extract_last_reasoning(conversation_trace: list[dict]) -> str:
    """
    Extract the last think() content from the conversation trace.

    This captures the agent's most recent analysis/reasoning, which is
    the most useful context for the next agent.
    """
    for entry in reversed(conversation_trace):
        if entry.get("tool") == "think":
            content = entry.get("args", {}).get("content", "")
            if content:
                return content[:600]
    return ""


class DialogueOrchestrator:
    """
    Manages the round-robin dialogue between two equal agents.

    Architecture (v2 — persistent agent identity):
        Each agent keeps its own ConversationMemory across turns.
        When Agent A resumes on turn 3, it has full memory of its turn 1
        work PLUS a handoff summary of what Agent B did on turn 2.

        Turn 1: Agent A works (N steps) — fresh memory
           → compress A's actions → handoff summary for B
        Turn 2: Agent B works (N steps) — fresh memory, A's handoff in task msg
           → compress B's actions → handoff summary for A
        Turn 3: Agent A resumes (same memory from turn 1 + B's handoff injected)
           → compress A's actions → handoff summary for B
        Turn 4: Agent B resumes (same memory from turn 2 + A's handoff injected)
           → ...
        Until: submit() called OR total budget exhausted
    """

    def __init__(
        self,
        llm: BaseLLM,
        terminal: SafeTerminal,
        config: Config,
    ):
        self._llm = llm
        self._terminal = terminal
        self._config = config

    def run(
        self,
        instruction: str,
        initial_observation: str,
        env_info: str,
    ) -> DialogueResult:
        """
        Execute the full dialogue.

        Args:
            instruction:         Task description.
            initial_observation: Initial pwd/ls output.
            env_info:            Environment discovery string.

        Returns:
            DialogueResult with the final agent result and per-turn metadata.
        """
        start_time = time.time()
        agent_cfg = self._config.agent
        variant = agent_cfg.dialogue_variant
        steps_per_turn = agent_cfg.dialogue_steps_per_turn
        max_steps = agent_cfg.max_steps
        max_wallclock = agent_cfg.max_wallclock_seconds

        max_turns = max_steps // steps_per_turn

        logger.info(
            f"Dialogue starting | variant={variant} | "
            f"steps_per_turn={steps_per_turn} | max_turns={max_turns} | "
            f"total_budget={max_steps} steps / {max_wallclock}s"
        )

        # Select handoff builder based on variant
        if variant == "debate":
            handoff_builder = build_debate_handoff
        else:
            handoff_builder = build_cooperation_handoff

        # Build system prompt with dialogue variant addendum
        system_prompt = get_system_prompt("v1", dialogue_variant=variant)

        # Persistent memory per agent (A=0, B=1)
        # Each agent keeps its ConversationMemory across all its turns.
        agent_memories: list[Optional["ConversationMemory"]] = [None, None]

        turns: list[TurnResult] = []
        total_steps_used = 0
        total_input_tokens = 0
        total_output_tokens = 0
        last_agent_result: Optional[AgentResult] = None
        handoff_context: Optional[str] = None
        proposed_answer: Optional[str] = None   # pending submit awaiting consensus
        proposer_idx: Optional[int] = None       # which agent proposed (0=A, 1=B)

        for turn_idx in range(max_turns):
            elapsed = time.time() - start_time

            # Wall-clock guard
            if elapsed >= max_wallclock:
                logger.warning(
                    f"Dialogue: wall-clock exhausted at turn {turn_idx + 1} "
                    f"({elapsed:.0f}s >= {max_wallclock}s)"
                )
                break

            # Step guard
            remaining_steps = max_steps - total_steps_used
            if remaining_steps <= 0:
                logger.warning("Dialogue: step budget exhausted")
                break

            # Which agent's turn (alternating A/B)
            agent_idx = turn_idx % 2
            agent_label = _AGENT_LABELS[agent_idx]

            # Steps for this turn: min of configured per-turn budget and remaining
            turn_steps = min(steps_per_turn, remaining_steps)

            # Wall-clock for this turn: proportional to remaining
            remaining_turns = max_turns - turn_idx
            remaining_wallclock = max(60, max_wallclock - elapsed)
            turn_wallclock = max(
                45,  # minimum 45s per turn
                int(remaining_wallclock / remaining_turns),
            )

            logger.info(
                f"Turn {turn_idx + 1}/{max_turns} | {agent_label} | "
                f"steps={turn_steps} | wallclock={turn_wallclock}s | "
                f"handoff={'yes' if handoff_context else 'no'} | "
                f"memory={'resume' if agent_memories[agent_idx] else 'fresh'}"
            )

            # ----------------------------------------------------------
            # Build turn-specific config
            # ----------------------------------------------------------
            turn_config = copy.deepcopy(self._config)
            turn_config.agent.max_steps = turn_steps
            turn_config.agent.max_wallclock_seconds = turn_wallclock

            # ----------------------------------------------------------
            # Create fresh executor + agent for this turn
            # (shares the same SafeTerminal — same Docker session)
            # ----------------------------------------------------------
            executor = ToolExecutor(
                terminal=self._terminal,
                default_timeout=self._config.agent.command_timeout,
                long_timeout=self._config.agent.long_command_timeout,
            )
            agent = ReActAgent(
                llm=self._llm,
                executor=executor,
                config=turn_config,
            )

            # ----------------------------------------------------------
            # Run the agent for this turn.
            # Suppress FINAL STEP forced submit for non-final turns
            # so the agent hands off instead of ending the dialogue.
            # ----------------------------------------------------------
            is_last_turn = (
                turn_idx == max_turns - 1
                or (remaining_steps - turn_steps) <= 0
            )

            if agent_memories[agent_idx] is not None:
                # Resuming — persistent memory + handoff injection
                result = agent.run(
                    instruction=instruction,
                    initial_observation=initial_observation,
                    env_info=env_info,
                    existing_memory=agent_memories[agent_idx],
                    handoff_inject=handoff_context,
                    dialogue_turn=not is_last_turn,
                )
            else:
                # First turn for this agent — fresh memory
                result = agent.run(
                    instruction=instruction,
                    initial_observation=initial_observation,
                    env_info=env_info,
                    handoff_context=handoff_context,
                    system_prompt_override=system_prompt,
                    dialogue_turn=not is_last_turn,
                )

            # Save memory for this agent's next turn
            agent_memories[agent_idx] = agent._memory

            # ----------------------------------------------------------
            # Accumulate stats
            # ----------------------------------------------------------
            total_steps_used += result.steps
            total_input_tokens += result.total_input_tokens
            total_output_tokens += result.total_output_tokens

            # Build turn result
            action_log = _build_action_log(result.conversation_trace)
            last_reasoning = _extract_last_reasoning(result.conversation_trace)

            turn_result = TurnResult(
                agent_label=agent_label,
                turn_number=turn_idx + 1,
                agent_result=result,
                action_log=action_log,
                last_reasoning=last_reasoning,
            )
            turns.append(turn_result)
            last_agent_result = result

            logger.info(
                f"Turn {turn_idx + 1} done | {agent_label} | "
                f"steps={result.steps} | "
                f"elapsed={result.elapsed_seconds:.1f}s | "
                f"submitted={result.submitted} | "
                f"tokens={result.total_input_tokens}in/"
                f"{result.total_output_tokens}out"
            )

            # ----------------------------------------------------------
            # Submit governance  (dual-submit / critic-authority)
            #
            # Cooperation: submit() is a *proposal*. The dialogue ends
            #   only when the OTHER agent also calls submit() (dual
            #   submit = consensus). The proposal persists across turns
            #   until the partner confirms or budget runs out.
            #   Same agent submitting again updates the proposal.
            #
            # Debate: Agent A (solver, idx=0) proposals require Agent B
            #   (critic, idx=1) to accept. Agent B can submit directly.
            #
            # Exception: last possible turn → accept immediately.
            # ----------------------------------------------------------
            real_submit = (
                result.submitted
                and result.termination_reason == "Agent submitted"
            )

            if real_submit:
                is_last_possible = (
                    turn_idx == max_turns - 1
                    or (remaining_steps - turn_steps) <= 0
                )

                # Last turn → accept unconditionally
                if is_last_possible:
                    logger.info(
                        f"Dialogue: {agent_label} submitted at turn "
                        f"{turn_idx + 1} (accepted: last turn)"
                    )
                    break

                # Debate: Agent B (critic idx=1) submits directly
                if variant == "debate" and agent_idx == 1:
                    logger.info(
                        f"Dialogue: {agent_label} submitted at turn "
                        f"{turn_idx + 1} (accepted: critic authority)"
                    )
                    break

                # Dual-submit consensus: different agent confirms
                if (
                    proposed_answer is not None
                    and proposer_idx is not None
                    and proposer_idx != agent_idx
                ):
                    logger.info(
                        f"Dialogue: {agent_label} submitted at turn "
                        f"{turn_idx + 1} (accepted: dual-submit consensus)"
                    )
                    break

                # -- Store / update proposal --
                proposed_answer = result.answer
                proposer_idx = agent_idx
                logger.info(
                    f"Dialogue: {agent_label} proposes at turn "
                    f"{turn_idx + 1} — awaiting partner consensus"
                )

                # Confirmation handoff only for the immediate next turn
                if variant == "debate":
                    handoff_context = build_evaluation_handoff(
                        turn_number=turn_idx + 1,
                        agent_label=agent_label,
                        action_log=action_log,
                        last_reasoning=last_reasoning,
                        proposed_answer=result.answer,
                    )
                else:
                    handoff_context = build_confirmation_handoff(
                        turn_number=turn_idx + 1,
                        agent_label=agent_label,
                        action_log=action_log,
                        last_reasoning=last_reasoning,
                        proposed_answer=result.answer,
                    )
                continue

            # No submit this turn — proposal stays alive (persistent).
            # Normal handoff for next turn.
            handoff_context = handoff_builder(
                turn_number=turn_idx + 1,
                agent_label=agent_label,
                action_log=action_log,
                last_reasoning=last_reasoning,
            )

        # ----------------------------------------------------------
        # Force-submit if no agent intentionally submitted
        # ----------------------------------------------------------
        real_submit = (
            last_agent_result is not None
            and last_agent_result.submitted
            and last_agent_result.termination_reason == "Agent submitted"
        )
        if last_agent_result and not real_submit:
            if proposed_answer is not None:
                # Unconfirmed proposal — use it as best available answer
                logger.warning(
                    "Dialogue: using unconfirmed proposal as final answer"
                )
                last_agent_result.submitted = True
                last_agent_result.answer = proposed_answer
                last_agent_result.termination_reason = (
                    "Unconfirmed proposal (partner did not confirm)"
                )
            else:
                logger.warning(
                    "Dialogue: no agent submitted — force-submitting"
                )
                # Use the terminal to check tests one more time
                self._force_submit(last_agent_result)

        # ----------------------------------------------------------
        # Build combined result.
        # Copy to avoid mutating the last turn's AgentResult in the
        # turns list (its per-turn values must stay intact for logging).
        # ----------------------------------------------------------
        if last_agent_result is not None:
            final_result = copy.copy(last_agent_result)
        else:
            final_result = AgentResult(
                termination_reason="no_turns_executed"
            )

        # Overwrite token totals with accumulated values
        final_result.total_input_tokens = total_input_tokens
        final_result.total_output_tokens = total_output_tokens
        final_result.steps = total_steps_used

        return DialogueResult(
            agent_result=final_result,
            turns=turns,
            variant=variant,
            total_turns=len(turns),
            total_steps=total_steps_used,
            total_elapsed=time.time() - start_time,
        )

    def _force_submit(self, last_result: AgentResult) -> None:
        """Force a submit when the dialogue ended without submission."""
        try:
            has_tests = self._terminal.file_exists("/tests/run-tests.sh")
            if has_tests:
                test_result = self._terminal.exec(
                    "cd /app && bash /tests/run-tests.sh 2>&1",
                    timeout=60.0,
                )
                output = test_result.output.upper()
                if "FAIL" not in output and "PASS" in output:
                    last_result.submitted = True
                    last_result.answer = (
                        "Task completed (auto-submitted: all tests pass)."
                    )
                    logger.info("Dialogue force-submit: tests pass")
                    return

            last_result.submitted = True
            last_result.answer = (
                "Task completed (auto-submitted on dialogue budget exhaustion)."
            )
            logger.info("Dialogue force-submit: budget exhaustion")

        except Exception as e:
            last_result.submitted = True
            last_result.answer = (
                "Task completed (auto-submitted: dialogue ended)."
            )
            logger.warning(f"Dialogue force-submit with error: {e}")
