"""
Orchestra agent — strategist orchestrator with persistent specialized sub-agents.

Architecture (v2 — persistent agent identity + optional shared KB):

  A Strategist Orchestrator (LLM-only, no tools) decides at runtime which
  specialist sub-agent to deploy and crafts detailed, multi-sentence
  instructions for that agent.  Each sub-agent keeps persistent memory
  across re-invocations, building expertise over the life of the task.

  Sub-agents:
    explorer  — read-only: bash, read_file, think [, note]
    coder     — implementation: bash, write_file, patch_file, read_file, think [, note]
    verifier  — test+fix+submit: bash, write_file, patch_file, read_file, think, submit [, note]

  All sub-agents share the same terminal (same tmux session).

  Cycle budget:
    1 orchestrator routing step (FREE — not counted) + up to 6 sub-agent steps
    Total: 60 specialist steps (same as single-agent), routing is excluded
    This ensures fair comparison: equal effective work budget
    Safety: max 20 cycles + wall-clock timeout
    Early termination: sub-agents can return early (< 6 steps), reclaiming budget

  Variants (controlled by THESIS_USE_KNOWLEDGE_BASE):
    0 (default) — Vanilla: orchestrator is sole information bridge
    1           — KB: + shared notepad with note() tool for all agents

  Persistent per-agent memory:
    Each sub-agent (explorer, coder, verifier) keeps its own ConversationMemory
    across all invocations.  When explorer is called on cycle 5, it remembers
    everything it did on cycles 1 and 3.  This mirrors the dialogue system's
    persistent identity pattern.

  Information flow:
    Vanilla: Orchestrator ←(summaries)→ Sub-agents  (orchestrator is sole bridge)
    KB:      Orchestrator ←(summaries)→ Sub-agents ←(note/read)→ SharedKnowledgeBase

  Orchestrator design:
    STATELESS — fresh messages each routing call.  All state is explicitly
    injected (progress history, budget, KB).  Matches Dan Austin's architecture.

        ┌─────────────────┐
        │   Strategist     │
        │  Orchestrator    │
        │  (LLM, no tools) │
        └────────┬─────────┘
                 │ {"next": role, "instruction": ..., "reasoning": ...}
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   ┌────────┐ ┌───────┐ ┌──────────┐
   │Explorer│ │ Coder │ │ Verifier │
   │ (read) │ │(write)│ │(test+fix │
   │  ↕mem  │ │ ↕mem  │ │ +submit) │
   └────────┘ └───────┘ │  ↕mem    │
        │         │      └──────────┘
        └─────────┴──────────┘
             shared terminal
          [shared KB if enabled]
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from llm.base import BaseLLM
from agent.memory import ConversationMemory, estimate_tokens


def _first_json_object(text):
    """Return the first complete JSON object in `text` as a dict, or None.
    String-aware balanced-brace scan (handles NESTED objects, unlike a flat regex)
    after stripping a leading Mistral [TOOL_CALLS] token. Additive: recovers routing
    JSON that the old flat ``{[^{}]*}`` regex truncated or missed (e.g. nested
    context_refs), without rejecting anything the flat regex previously accepted."""
    if not text:
        return None
    t = text.replace("[TOOL_CALLS]", " ")
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(t):
        if esc:
            esc = False
            continue
        if ch == chr(92):  # backslash → escape next char
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(t[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        start = -1
    return None
from agent.executor import ToolExecutor
from agent.react import AgentResult
from agent.tools import get_tool_schemas, NOTE_TOOL_SCHEMA
from agent.prompts import get_system_prompt
from agent.knowledge_base import SharedKnowledgeBase

logger = logging.getLogger(__name__)

# Tool subsets per role
ROLE_ALLOWED_TOOLS: dict[str, set[str]] = {
    "explorer": {"bash", "read_file", "think", "submit"},
    "coder":    {"bash", "write_file", "patch_file", "read_file", "think", "submit"},
    "verifier": {"bash", "write_file", "patch_file", "read_file", "think", "submit"},
}

_TOOL_SCHEMA_ESTIMATED_TOKENS = 2000


@dataclass
class BurstResult:
    """Outcome of a sub-agent burst."""

    role: str = ""
    steps: int = 0
    submitted: bool = False
    early_stop: bool = False  # True if agent signalled done (submit/text_only)
    answer: Optional[str] = None
    trace: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class OrchestraAgent:
    """
    Strategist orchestrator with persistent specialized sub-agents.

    Main loop (per cycle):
      1. Orchestrator decides: {"next": role, "instruction": ..., "reasoning": ...}
      2. Sub-agent runs a burst of up to N steps (can return early)
      3. Results are summarized → state updated → repeat

    Budget: orchestrator routing calls + sub-agent steps all count toward max_steps.
    """

    def __init__(
        self,
        llm: BaseLLM,
        executor: ToolExecutor,
        config: Config,
        burst_steps: int = 6,
        knowledge_base: Optional[SharedKnowledgeBase] = None,
        supervisor_llm: BaseLLM = None,   # heterogeneous supervisor (orchestrator/strategist routing); None => same as llm
    ):
        self._llm = llm
        self._sup_llm = supervisor_llm if supervisor_llm is not None else llm
        self._executor = executor
        self._config = config
        self._burst_steps = burst_steps
        self._knowledge_base = knowledge_base
        self._use_kb = knowledge_base is not None

        # Token accounting
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Conversation trace (for logging/analysis)
        self._conversation_trace: list[dict] = []

        # Runtime state (set in run())
        self._run_start_time: float = 0.0
        self._global_step: int = 0
        self._cycle: int = 0

        # Persistent per-agent memories (populated on first invocation)
        self._agent_memories: dict[str, ConversationMemory] = {}

        # Pre-filter tool schemas per role
        all_tools = get_tool_schemas("default")
        self._role_tools: dict[str, list] = {}
        for role, allowed in ROLE_ALLOWED_TOOLS.items():
            role_tools = [
                t for t in all_tools if t["function"]["name"] in allowed
            ]
            # Add note tool if KB is enabled
            if self._use_kb:
                role_tools = role_tools + [NOTE_TOOL_SCHEMA]
            self._role_tools[role] = role_tools

    def run(
        self,
        instruction: str,
        initial_observation: str = "",
        env_info: str = "",
    ) -> AgentResult:
        """Execute the orchestrated multi-agent loop."""
        self._run_start_time = time.time()
        self._global_step = 0
        self._cycle = 0
        agent_cfg = self._config.agent
        max_steps = agent_cfg.max_steps

        # Progress history for orchestrator context (all past cycles)
        phase_history: list[dict] = []
        termination_reason = ""

        logger.info(
            f"Orchestra starting | max_steps={max_steps} "
            f"| burst_steps={self._burst_steps} "
            f"| max_wallclock={agent_cfg.max_wallclock_seconds}s "
            f"| knowledge_base={'enabled' if self._use_kb else 'disabled'}"
        )

        _MAX_CYCLES = 20  # Safety cap against infinite routing

        while self._global_step < max_steps:
            elapsed = time.time() - self._run_start_time

            # Cycle cap safety
            if self._cycle >= _MAX_CYCLES:
                termination_reason = f"Max cycles reached ({_MAX_CYCLES})"
                break

            # Wall-clock check
            if elapsed > agent_cfg.max_wallclock_seconds:
                termination_reason = (
                    f"Wall-clock timeout ({elapsed:.0f}s "
                    f"> {agent_cfg.max_wallclock_seconds}s)"
                )
                logger.warning(termination_reason)
                break

            # ----- Orchestrator routing decision (free — not counted in budget) -----
            self._cycle += 1
            remaining = max_steps - self._global_step

            if remaining < 1:
                termination_reason = f"Max steps reached ({max_steps})"
                break

            routing = self._route(
                task=instruction,
                initial_obs=initial_observation,
                env_info=env_info,
                history=phase_history,
                steps_used=self._global_step,
                steps_remaining=remaining,
            )

            role = routing["next"]
            sub_instruction = routing["instruction"]
            reasoning = routing.get("reasoning", "")
            context_refs = routing.get("context_refs", [])

            logger.info(
                f"Cycle {self._cycle} | Orchestrator → {role} "
                f"(step {self._global_step}/{max_steps}): "
                f"{sub_instruction[:120]}"
            )
            if reasoning:
                logger.info(f"  Reasoning: {reasoning[:120]}")

            # Record routing decision in trace
            self._conversation_trace.append({
                "step": self._global_step,
                "elapsed_s": round(time.time() - self._run_start_time, 1),
                "tool": "_orchestrator_route",
                "args": {
                    "next": role,
                    "instruction": sub_instruction[:300],
                    "reasoning": reasoning[:200],
                },
                "result_preview": f"Routing to {role}",
                "input_tokens": 0,
                "output_tokens": 0,
                "burst_role": "orchestrator",
                "cycle": self._cycle,
            })

            # ----- Sub-agent burst -----
            burst_budget = min(self._burst_steps, remaining)
            burst_result = self._run_burst(
                role=role,
                instruction=sub_instruction,
                task=instruction,
                initial_obs=initial_observation,
                env_info=env_info,
                burst_steps=burst_budget,
                context_refs=context_refs,
            )

            self._global_step += burst_result.steps

            # Merge burst trace into main trace
            self._conversation_trace.extend(burst_result.trace)

            # Update progress history for orchestrator
            summary = self._summarize_burst(burst_result)
            phase_history.append({
                "cycle": self._cycle,
                "role": role,
                "instruction": sub_instruction,
                "summary": summary,
                "steps_used": burst_result.steps,
                "steps_budget": burst_budget,
                "early_stop": burst_result.early_stop,
            })

            logger.info(
                f"  {role} burst done: {burst_result.steps}/{burst_budget} steps, "
                f"{burst_result.input_tokens}in/{burst_result.output_tokens}out"
                f"{f' | KB entries: {self._knowledge_base.size}' if self._use_kb else ''}"
            )
            logger.info(f"  Summary: {summary[:200]}")

            if burst_result.submitted:
                termination_reason = "Agent submitted"
                break

        else:
            termination_reason = f"Max steps reached ({max_steps})"

        # ----- Force-submit if needed -----
        if not self._executor.submitted:
            self._force_submit(termination_reason)

        # ----- Build result -----
        elapsed = time.time() - self._run_start_time
        stats = self._executor.stats
        stats["orchestrator_calls"] = self._cycle
        stats["cycles"] = self._cycle
        if self._use_kb:
            stats["kb_entries"] = self._knowledge_base.size

        return AgentResult(
            success=self._executor.submitted,
            submitted=self._executor.submitted,
            answer=self._executor.final_answer,
            steps=self._global_step,
            elapsed_seconds=elapsed,
            termination_reason=termination_reason,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            tool_stats=stats,
            messages_dropped=sum(
                m._messages_dropped
                for m in self._agent_memories.values()
            ),
            conversation_trace=self._conversation_trace,
        )

    # ------------------------------------------------------------------
    # Orchestrator routing (stateless — fresh messages each call)
    # ------------------------------------------------------------------
    def _route(
        self, task, initial_obs, env_info, history,
        steps_used, steps_remaining,
    ) -> dict:
        """Ask the orchestrator LLM which sub-agent to deploy next."""
        system_prompt = get_system_prompt(
            "orchestrator", use_knowledge_base=self._use_kb,
        )
        state_text = self._format_orchestrator_state(
            history, steps_used, steps_remaining,
        )

        user_content_parts = [
            f"## Task\n{task}\n",
            f"\n## Environment\n{initial_obs}\n",
        ]
        if env_info and env_info.strip():
            user_content_parts.append(f"{env_info}\n")

        user_content_parts.append(f"\n{state_text}")

        # Include KB if enabled (compact view for routing decisions)
        if self._use_kb:
            user_content_parts.append(
                f"\n{self._knowledge_base.format_for_orchestrator()}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ]

        try:
            response = self._sup_llm.chat(
                messages=messages,
                tools=None,
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as e:
            logger.warning(f"Orchestrator LLM call failed: {e}")
            return {
                "next": "coder",
                "instruction": "Continue working on the task.",
                "reasoning": f"LLM error fallback: {e}",
                "context_refs": [],
            }

        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens

        return self._parse_routing(response.text or "")

    def _parse_routing(self, text: str) -> dict:
        """Parse JSON routing decision from orchestrator response."""
        try:
            data = _first_json_object(text)
            if data is not None:
                role = data.get("next", "coder")
                if role not in ROLE_ALLOWED_TOOLS:
                    role = "coder"
                instruction = data.get("instruction", "Continue working.")
                reasoning = data.get("reasoning", "")
                context_refs = data.get("context_refs", [])
                if not isinstance(context_refs, list):
                    context_refs = []
                return {
                    "next": role,
                    "instruction": instruction,
                    "reasoning": reasoning,
                    "context_refs": context_refs,
                }
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: detect role from text
        text_lower = text.lower()
        if "explorer" in text_lower or "explore" in text_lower:
            return {
                "next": "explorer",
                "instruction": "Explore the environment: read README, check for tests, inspect input data files. Report file structure, data schemas, and test expectations.",
                "reasoning": "Fallback parse — detected explorer intent.",
                "context_refs": [],
            }
        if "verif" in text_lower or "test" in text_lower or "submit" in text_lower:
            return {
                "next": "verifier",
                "instruction": "Run tests with `bash /tests/run-tests.sh` and submit if all pass. If no tests, verify output files and submit.",
                "reasoning": "Fallback parse — detected verifier intent.",
                "context_refs": [],
            }

        return {
            "next": "coder",
            "instruction": "Continue working on the task. Read relevant files, implement the solution, and test it.",
            "reasoning": "Fallback parse — could not extract JSON.",
            "context_refs": [],
        }

    def _format_orchestrator_state(
        self, history, steps_used, steps_remaining,
    ) -> str:
        """Format the orchestration state for the orchestrator LLM."""
        parts = [
            f"## Budget\n"
            f"Steps used: {steps_used} | Steps remaining: {steps_remaining}"
        ]

        if steps_remaining <= 13:
            parts.append(
                "⚠ LOW BUDGET — consider routing to verifier for best-effort submission."
            )

        if not history:
            parts.append(
                "\n## Progress\n"
                "No actions taken yet. This is the first cycle."
            )
        else:
            parts.append("\n## Progress (chronological)")
            # Show all history (capped at last 8 for token budget)
            for phase in history[-8:]:
                steps_info = f"{phase['steps_used']}/{phase['steps_budget']} steps"
                status = ""
                if not phase.get("early_stop", True):
                    status = " ⚠ INTERRUPTED — agent ran out of steps and may need more time"
                parts.append(
                    f"\n**Cycle {phase['cycle']} — {phase['role']}** ({steps_info}{status}):\n"
                    f"Assignment: {phase['instruction'][:200]}\n"
                    f"Result: {phase['summary']}"
                )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Sub-agent burst execution (with persistent memory)
    # ------------------------------------------------------------------
    def _run_burst(
        self,
        role: str,
        instruction: str,
        task: str,
        initial_obs: str,
        env_info: str,
        burst_steps: int,
        context_refs: Optional[list[str]] = None,
    ) -> BurstResult:
        """Run a sub-agent for a burst of up to N steps.

        On first invocation for a role, creates a fresh ConversationMemory
        with the task context.  On subsequent invocations, injects the new
        orchestrator instruction into the existing memory.

        Sub-agents can return early by producing a response with no tool calls.
        """
        tools = self._role_tools[role]
        is_first = role not in self._agent_memories

        if is_first:
            # First invocation — create memory with full task context
            system_prompt = get_system_prompt(
                f"orchestra_{role}", use_knowledge_base=self._use_kb,
            )
            task_message = self._build_initial_task(
                task=task,
                instruction=instruction,
                initial_obs=initial_obs,
                env_info=env_info,
                burst_budget=burst_steps,
                context_refs=context_refs,
            )

            memory = ConversationMemory(
                system_prompt=system_prompt,
                task_message=task_message,
                max_turns=60,
                max_output_chars=self._config.agent.max_output_chars,
                max_context_tokens=self._config.agent.max_context_tokens,
                overhead_tokens=_TOOL_SCHEMA_ESTIMATED_TOKENS,
            )
            self._agent_memories[role] = memory
        else:
            # Subsequent invocation — inject new assignment into existing memory
            memory = self._agent_memories[role]
            injection = self._build_orchestrator_injection(
                instruction=instruction,
                burst_budget=burst_steps,
                context_refs=context_refs,
            )
            memory.add_user_message(injection)

        # ----- Burst loop -----
        trace: list[dict] = []
        total_steps = 0
        submitted = False
        _burst_done = False
        _early_stop = False
        in_tokens = 0
        out_tokens = 0
        _consecutive_thinks = 0
        _MAX_CONSECUTIVE_THINKS = 2

        for step_i in range(1, burst_steps + 1):
            # Wall-clock check
            elapsed = time.time() - self._run_start_time
            if elapsed > self._config.agent.max_wallclock_seconds:
                break

            total_steps += 1
            global_step = self._global_step + total_steps

            # Push budget info to executor (for think tool)
            self._executor._current_step = global_step
            self._executor._max_steps = self._config.agent.max_steps
            self._executor._elapsed_seconds = elapsed
            self._executor._max_wallclock = self._config.agent.max_wallclock_seconds

            messages = memory.get_messages()

            try:
                response = self._llm.chat(messages=messages, tools=tools)
            except Exception as e:
                logger.warning(f"Burst {role} step {step_i} LLM error: {e}")
                break

            in_tokens += response.input_tokens
            out_tokens += response.output_tokens
            self._total_input_tokens += response.input_tokens
            self._total_output_tokens += response.output_tokens

            if response.has_tool_calls:
                tool_calls_for_memory = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(
                                tc.arguments, ensure_ascii=False,
                            ),
                        },
                    }
                    for tc in response.tool_calls
                ]
                memory.add_assistant_message(
                    content=response.content,
                    tool_calls=tool_calls_for_memory,
                )

                for tc in response.tool_calls:
                    # Handle note tool (KB) — checked first, outside
                    # the role allowlist since it's conditionally added
                    if tc.name == "note":
                        if self._knowledge_base is not None:
                            result = self._knowledge_base.add(
                                category=tc.arguments.get("category", "discovery"),
                                title=tc.arguments.get("title", "Untitled"),
                                content=tc.arguments.get("content", ""),
                                author=role,
                                cycle=self._cycle,
                            )
                        else:
                            result = "[ERROR] note tool is not available."
                    # Explorer/coder submit = "I'm done" signal (not real submit)
                    elif tc.name == "submit" and role != "verifier":
                        answer = tc.arguments.get("answer", "")
                        result = (
                            "[DONE] Assignment complete. Your report "
                            "has been forwarded to the orchestrator."
                        )
                        # Record as text report in trace so _summarize_burst picks it up
                        memory.add_tool_result(tc.id, tc.name, result)
                        trace.append({
                            "step": global_step,
                            "elapsed_s": round(
                                time.time() - self._run_start_time, 1,
                            ),
                            "tool": "_agent_done",
                            "args": {"answer": answer[:300]},
                            "result_preview": answer[:300],
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "burst_role": role,
                            "cycle": self._cycle,
                        })
                        _burst_done = True
                        _early_stop = True
                        break
                    # Block any tool not in this role's allowed set
                    elif tc.name not in ROLE_ALLOWED_TOOLS[role]:
                        allowed = sorted(ROLE_ALLOWED_TOOLS[role])
                        result = (
                            f"[BLOCKED] Tool '{tc.name}' is not available "
                            f"for your role ({role}). "
                            f"Available tools: {', '.join(allowed)}."
                        )
                    else:
                        result = self._executor.execute(
                            tc.name, tc.arguments,
                        )

                    memory.add_tool_result(tc.id, tc.name, result)

                    # Trace entry
                    result_preview = (
                        result[:200] + "..."
                        if len(result) > 200
                        else result
                    )
                    trace.append({
                        "step": global_step,
                        "elapsed_s": round(
                            time.time() - self._run_start_time, 1,
                        ),
                        "tool": tc.name,
                        "args": {
                            k: (
                                v[:200] + "..."
                                if isinstance(v, str) and len(v) > 200
                                else v
                            )
                            for k, v in tc.arguments.items()
                        },
                        "result_preview": result_preview,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "burst_role": role,
                        "cycle": self._cycle,
                    })

                    if self._executor.submitted:
                        submitted = True
                        _early_stop = True
                        break

                    # Think loop detection: consecutive think-only steps
                    # indicate the agent is stuck repeating itself.
                    if tc.name == "think":
                        _consecutive_thinks += 1
                    else:
                        _consecutive_thinks = 0

                if submitted or _burst_done:
                    break

                # Force-break if agent is looping on think()
                if _consecutive_thinks >= _MAX_CONSECUTIVE_THINKS:
                    logger.info(
                        f"Burst {role} C{self._cycle}: "
                        f"{_consecutive_thinks} consecutive thinks — "
                        f"force-ending burst (likely stuck)."
                    )
                    break

            else:
                # No tool calls → sub-agent considers assignment complete
                if response.content:
                    # Save the report in memory so it persists
                    memory.add_assistant_message(content=response.content)
                    trace.append({
                        "step": global_step,
                        "elapsed_s": round(
                            time.time() - self._run_start_time, 1,
                        ),
                        "tool": "_text_only",
                        "args": {},
                        "result_preview": (response.content or "")[:300],
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "burst_role": role,
                        "cycle": self._cycle,
                    })
                break  # Early termination — agent is done with assignment
            _early_stop = True  # text-only = done signal

        return BurstResult(
            role=role,
            steps=total_steps,
            submitted=submitted,
            early_stop=_early_stop,
            answer=self._executor.final_answer if submitted else None,
            trace=trace,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )

    # ------------------------------------------------------------------
    # Task message builders
    # ------------------------------------------------------------------
    def _build_initial_task(
        self, task, instruction, initial_obs, env_info, burst_budget,
        context_refs=None,
    ) -> str:
        """Build the first user message for a sub-agent (full context)."""
        parts = [
            f"## Task\n{task.strip()}",
            f"\n\n## Environment\n```\n"
            f"{initial_obs.strip() if initial_obs.strip() else '(unknown)'}"
            f"\n```",
        ]

        if env_info and env_info.strip():
            parts.append(f"\n```\n{env_info.strip()}\n```")

        parts.append(
            f"\n\n## Your Assignment from the Orchestrator\n{instruction.strip()}"
        )

        parts.append(
            f"\n\n## Budget\n"
            f"You have up to {burst_budget} steps for this assignment. "
            f"Return early if you finish sooner.\n"
            f"Overall task progress: {self._global_step}/{self._config.agent.max_steps} steps used."
        )

        if self._use_kb:
            parts.append(
                f"\n\n{self._knowledge_base.format_for_agent(context_refs)}"
            )

        parts.append(
            "\n\n## Reporting\n"
            "When you've completed your assignment, call submit() with a summary "
            "of your findings or what you did. This signals you are done."
        )

        return "\n".join(parts)

    def _build_orchestrator_injection(
        self, instruction, burst_budget, context_refs=None,
    ) -> str:
        """Build the injection message for a sub-agent on re-invocation."""
        parts = [
            f"## New Assignment from the Orchestrator\n{instruction.strip()}",
            f"\n## Budget\n"
            f"You have up to {burst_budget} steps. Return early if you finish sooner.\n"
            f"Overall: {self._global_step}/{self._config.agent.max_steps} steps used.",
        ]

        if self._use_kb:
            parts.append(
                f"\n{self._knowledge_base.format_for_agent(context_refs)}"
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Burst summarization
    # ------------------------------------------------------------------
    @staticmethod
    def _summarize_burst(result: BurstResult) -> str:
        """Create a summary of a burst's actions for the orchestrator.

        Includes bash output snippets so the orchestrator can see what
        was actually found/produced — not just which commands were run.
        """
        if not result.trace:
            return "No actions taken."

        action_parts: list[str] = []
        agent_report = ""
        last_think = ""

        for e in result.trace:
            tool = e.get("tool", "?")
            args = e.get("args", {})
            preview = e.get("result_preview", "")

            if tool == "bash":
                cmd = args.get("command", "")[:120]
                # Include output snippet so orchestrator sees WHAT happened
                output = preview[:200].strip()
                if "[ERROR]" in preview or "FAIL" in preview.upper():
                    action_parts.append(f"bash({cmd}) ⚠FAILED → {output}")
                elif output:
                    action_parts.append(f"bash({cmd}) → {output}")
                else:
                    action_parts.append(f"bash({cmd})")
            elif tool in ("write_file", "patch_file"):
                action_parts.append(f"{tool}({args.get('path', '?')})")
            elif tool == "read_file":
                # Include content snippet for files read
                content_snip = preview[:150].strip()
                if content_snip:
                    action_parts.append(
                        f"read({args.get('path', '?')}) → {content_snip}"
                    )
                else:
                    action_parts.append(f"read({args.get('path', '?')})")
            elif tool == "think":
                last_think = args.get("content", "")[:400]
            elif tool == "note":
                action_parts.append(f"note({args.get('title', '?')})")
            elif tool == "submit":
                action_parts.append(
                    f"submit({args.get('answer', '?')[:60]})"
                )
            elif tool == "_agent_done":
                agent_report = args.get("answer", "")[:500]
            elif tool == "_text_only":
                agent_report = preview[:500]

        summary = "\n".join(action_parts) if action_parts else ""

        # Append agent's concluding report (from text-only response)
        if agent_report:
            summary += f"\nReport: {agent_report}"

        # Append last reasoning if no text report
        if not agent_report and last_think:
            summary += f"\nAnalysis: {last_think}"

        return summary or "No actions taken."

    # ------------------------------------------------------------------
    # Force-submit
    # ------------------------------------------------------------------
    def _force_submit(self, reason: str) -> None:
        """Force-submit if the orchestrator ran out of budget."""
        terminal = self._executor._terminal
        try:
            has_tests = terminal.file_exists("/tests/run-tests.sh")
            if has_tests:
                test_result = terminal.exec(
                    "cd /app && bash /tests/run-tests.sh 2>&1",
                    timeout=60.0,
                )
                output = test_result.output.upper()
                if "FAIL" not in output and "PASS" in output:
                    self._executor._submitted = True
                    self._executor._final_answer = (
                        "Task completed (auto-submitted: all tests pass)."
                    )
                    logger.info("Force-submit: tests pass")
                    return

            self._executor._submitted = True
            self._executor._final_answer = (
                f"Best effort (budget exhausted: {reason})"
            )
            logger.info(f"Force-submit: {reason}")
        except Exception as e:
            self._executor._submitted = True
            self._executor._final_answer = f"Budget exhausted: {reason}"
            logger.warning(f"Force-submit error: {e}")
