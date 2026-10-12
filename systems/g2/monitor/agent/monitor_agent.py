"""
Monitor orchestrator v2 — Executor + Monitor interleaved guidance.

Changes from v1:
  1. Richer context: 500-char stdout + full test output verbatim.
  2. Review interval=10, silent-when-fine (no injection if "on track").
  3. CAMEL-style inception injection: monitor feedback is injected as
     an assistant think() call + tool result, not a user message.
     This makes the executor treat it as its own reasoning (inception).
  4. Single-shot monitor with full action log (no rolling window).
  5. Escalating authority via prior-advice tracking in monitor.py.

Architecture:
        ┌──────────┐  actions    ┌─────────┐
        │ Executor │ ──────────► │ Monitor │
        │ (ReAct)  │             │ (LLM)   │
        │          │  inception  │         │
        │          │ ◄────────── │         │
        └──────────┘             └─────────┘
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from agent.react import ReActAgent, AgentResult
from agent.monitor import Monitor, ON_TRACK_SENTINEL
from agent.prompts import get_monitor_prompt

logger = logging.getLogger(__name__)


class MonitorAgent(ReActAgent):
    """
    Two-agent system: Executor + Monitor.

    Extends ReActAgent with periodic monitor feedback injection.
    The monitor sees the full action history (single-shot) and its
    feedback is injected via CAMEL inception (think + tool_result).
    """

    def __init__(
        self,
        llm,
        executor,
        config,
        review_interval: int = 10,
        supervisor_llm=None,   # heterogeneous supervisor (Monitor role); None => same as llm
    ):
        super().__init__(llm, executor, config)
        self._sup_llm = supervisor_llm if supervisor_llm is not None else llm
        self._review_interval = review_interval
        self._monitor: Optional[Monitor] = None
        self._review_count = 0
        self._monitor_input_tokens = 0
        self._monitor_output_tokens = 0
        self._run_start_time: float = 0.0
        self._skipped_injections = 0

    def run(
        self,
        instruction: str,
        initial_observation: str = "",
        env_info: str = "",
    ) -> AgentResult:
        """Execute with periodic monitor feedback."""
        self._monitor = Monitor(
            llm=self._sup_llm,
            task_description=instruction,
            system_prompt=get_monitor_prompt(),
        )
        self._run_start_time = time.time()

        logger.info(
            f"Monitor agent v2: review_interval={self._review_interval}"
        )

        result = super().run(instruction, initial_observation, env_info)

        # Add monitor tokens to totals (for fair comparison)
        result.total_input_tokens += self._monitor_input_tokens
        result.total_output_tokens += self._monitor_output_tokens

        # Add monitor stats to tool_stats
        result.tool_stats["monitor_calls"] = self._review_count
        result.tool_stats["monitor_injections"] = self._review_count - self._skipped_injections
        result.tool_stats["monitor_skipped"] = self._skipped_injections
        result.tool_stats["monitor_input_tokens"] = self._monitor_input_tokens
        result.tool_stats["monitor_output_tokens"] = self._monitor_output_tokens

        return result

    # ------------------------------------------------------------------
    # Hook: called after every step by ReActAgent's loop
    # ------------------------------------------------------------------
    def _post_step_hook(self, step: int) -> None:
        """Inject monitor feedback every ``review_interval`` steps."""
        if step % self._review_interval != 0:
            return
        if self._executor.submitted:
            return

        self._review_count += 1

        # -- Build full action log (1 line per step, all steps) --
        full_action_log = self._build_full_action_log()

        # -- Build enriched status context --
        enriched_context = self._build_enriched_context(step)

        # -- Call monitor (single-shot with full history) --
        guidance, in_tok, out_tok = self._monitor.review(
            executor_context=enriched_context,
            full_action_log=full_action_log,
            current_step=step,
        )
        self._monitor_input_tokens += in_tok
        self._monitor_output_tokens += out_tok

        # -- Silent-when-fine: skip injection if on track --
        if guidance == ON_TRACK_SENTINEL:
            self._skipped_injections += 1
            self._conversation_trace.append({
                "step": step,
                "elapsed_s": round(time.time() - self._run_start_time, 1),
                "tool": "_monitor",
                "args": {"check_in_number": self._review_count, "injected": False},
                "result_preview": "ON_TRACK — no injection",
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            })
            logger.info(
                f"Monitor check-in #{self._review_count} at step {step}: "
                f"on track — skipped injection [{in_tok}in/{out_tok}out]"
            )
            return

        # -- CAMEL inception injection --
        # Inject as: assistant thinks → tool returns monitor assessment.
        # The executor sees this as its OWN reasoning, not external advice.
        # This dramatically increases compliance vs user-message injection.
        self._memory.add_assistant_message(
            content=None,
            tool_calls=[{
                "id": f"monitor_{self._review_count}",
                "type": "function",
                "function": {
                    "name": "think",
                    "arguments": '{"content": "Let me reconsider my approach based on the progress monitor\'s assessment..."}',
                },
            }],
        )
        self._memory.add_tool_result(
            tool_call_id=f"monitor_{self._review_count}",
            tool_name="think",
            result=(
                f"[PROGRESS ASSESSMENT — Check-in #{self._review_count}]\n\n"
                f"{guidance}\n\n"
                f"[Step {step}/{self._config.agent.max_steps} | "
                f"Budget info above. Consider this assessment and adjust approach if needed.]"
            ),
        )

        # Log in conversation trace
        self._conversation_trace.append({
            "step": step,
            "elapsed_s": round(time.time() - self._run_start_time, 1),
            "tool": "_monitor",
            "args": {"check_in_number": self._review_count, "injected": True},
            "result_preview": guidance[:200],
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })

        logger.info(
            f"Monitor check-in #{self._review_count} INJECTED at step {step} "
            f"(inception) [{in_tok}in/{out_tok}out]"
        )

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------
    def _build_full_action_log(self) -> str:
        """Build compact 1-line-per-step log of ALL executor actions."""
        lines: list[str] = []
        for e in self._conversation_trace:
            if e.get("tool") == "_monitor":
                continue  # skip monitor's own entries
            tool = e.get("tool", "?")
            step = e.get("step", "?")
            args = e.get("args", {})
            # Richer preview: 500 chars instead of 150
            result_preview = (e.get("result_preview", "") or "")[:500]

            if tool == "bash":
                cmd = args.get("command", "")[:120]
                line = f"Step {step}: bash({cmd})"
            elif tool in ("write_file", "patch_file", "read_file"):
                path = args.get("path", "?")
                line = f"Step {step}: {tool}({path})"
            elif tool == "think":
                thought = (args.get("content", "") or "")[:80]
                line = f"Step {step}: think({thought})"
            elif tool == "submit":
                answer = (args.get("answer", "") or "")[:60]
                line = f"Step {step}: submit({answer})"
            else:
                line = f"Step {step}: {tool}"

            if result_preview:
                # Show up to 500 chars (vs 150 in v1) — critical for diagnosis
                line += f"\n  → {result_preview}"

            lines.append(line)

        return "\n".join(lines) if lines else "(no actions yet)"

    def _build_enriched_context(self, step: int) -> str:
        """Build budget + test + file modification context."""
        max_steps = self._config.agent.max_steps
        elapsed = time.time() - self._run_start_time
        max_wallclock = self._config.agent.max_wallclock_seconds

        budget_info = (
            f"Steps used: {step}/{max_steps} "
            f"({max_steps - step} remaining)\n"
            f"Time elapsed: {elapsed:.0f}s / {max_wallclock}s "
            f"({max(0, max_wallclock - elapsed):.0f}s remaining)"
        )

        # Collect file modifications from full trace
        files_modified = set()
        # Collect ALL test outputs verbatim (not truncated)
        test_results: list[str] = []
        for e in self._conversation_trace:
            if e.get("tool") == "_monitor":
                continue
            tool = e.get("tool", "")
            if tool in ("write_file", "patch_file"):
                path = e.get("args", {}).get("path", "")
                if path:
                    files_modified.add(path)
            if tool == "bash":
                cmd = e.get("args", {}).get("command", "")
                preview = e.get("result_preview", "") or ""
                # Include full test results verbatim (up to 500 chars each)
                if any(kw in cmd.lower() for kw in (
                    "test", "pytest", "run-tests", "check",
                )) or any(kw in preview.lower() for kw in (
                    "passed", "failed", "error", "assert", "traceback",
                )):
                    test_results.append(
                        f"Step {e.get('step', '?')}: {cmd[:100]}\n"
                        f"  → {preview[:500]}"
                    )

        parts = [f"### Budget status:\n{budget_info}"]

        if files_modified:
            parts.append(
                "\n### Files modified:\n"
                + "\n".join(f"- {f}" for f in sorted(files_modified))
            )

        if test_results:
            parts.append(
                "\n### Test execution history (verbatim):\n"
                + "\n\n".join(test_results[-5:])  # Last 5 test runs, full output
            )

        return "\n".join(parts)
