"""
Monitor agent for the G2 monitor system (v2).

Single-shot architecture: every check-in is an independent LLM call
with the FULL action history (no persistent chat, no rolling window).
The monitor sees a panoramic view of everything the executor has done.

Escalating authority: the monitor tracks its own previous advice and
whether the executor followed it.  Repeated ignored advice triggers
increasingly urgent directives.

Design:
  - No tools — purely observational / advisory.
  - Same LLM as the Executor (different system prompt).
  - Single-shot calls with full cumulative context.
  - Returns "ON_TRACK" sentinel when everything is fine (caller skips injection).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llm.base import BaseLLM, Message

logger = logging.getLogger(__name__)

# Sentinel returned when the monitor has nothing to add.
# The caller (monitor_agent.py) checks for this and skips injection.
ON_TRACK_SENTINEL = "ON_TRACK"


@dataclass
class _PriorAdvice:
    """Record of a single prior check-in."""
    check_in: int
    guidance: str
    step: int = 0


class Monitor:
    """
    Single-shot monitor with escalating authority.

    Every review() call builds a fresh message list containing:
      1. System prompt (with escalation instructions)
      2. Task description
      3. Full cumulative action log (1 line per step)
      4. Summary of all prior advice given + whether followed
      5. Current check-in context (budget, recent test output, etc.)

    Returns ON_TRACK_SENTINEL when there is nothing meaningful to inject.
    """

    def __init__(
        self,
        llm: BaseLLM,
        task_description: str,
        system_prompt: str,
    ):
        self._llm = llm
        self._system_prompt = system_prompt
        self._task_description = task_description

        self._review_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Track all prior advice for escalation awareness
        self._prior_advice: list[_PriorAdvice] = []

    def review(
        self,
        executor_context: str,
        full_action_log: str,
        current_step: int,
    ) -> tuple[str, int, int]:
        """
        Provide guidance based on executor's full history.

        Single-shot: no persistent conversation.  Every call gets the
        full picture so the monitor can detect long-range patterns.

        Args:
            executor_context: Budget status, test results, files modified.
            full_action_log:  Compact 1-line-per-step log of ALL actions so far.
            current_step:     The step number at which this review fires.

        Returns:
            Tuple of (guidance_text, input_tokens, output_tokens).
            guidance_text is ON_TRACK_SENTINEL if nothing to inject.
        """
        self._review_count += 1

        # -- Build prior-advice summary for escalation awareness --
        advice_summary = self._build_advice_summary()

        # -- Compose single-shot message list --
        user_content_parts = [
            f"## Task\n{self._task_description}",
            f"\n## Full action log (all steps so far)\n{full_action_log}",
            f"\n## Current status\n{executor_context}",
        ]
        if advice_summary:
            user_content_parts.append(
                f"\n## Your prior advice history\n{advice_summary}"
            )
        user_content_parts.append(
            f"\n## Check-in #{self._review_count} — provide your assessment."
        )

        messages: list[Message] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ]

        try:
            response = self._llm.chat(
                messages=messages,
                tools=None,
                temperature=0.0,
                max_tokens=350,
            )
        except Exception as e:
            logger.warning(f"Monitor call #{self._review_count} failed: {e}")
            return ON_TRACK_SENTINEL, 0, 0

        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens

        raw = response.text.strip() if response.text else ""

        # Detect "on track" responses → return sentinel (no injection)
        if self._is_on_track(raw):
            logger.info(
                f"Monitor check-in #{self._review_count}: on track — skipping injection "
                f"[{response.input_tokens}in/{response.output_tokens}out]"
            )
            return ON_TRACK_SENTINEL, response.input_tokens, response.output_tokens

        # Record this advice for future escalation context
        self._prior_advice.append(_PriorAdvice(
            check_in=self._review_count,
            guidance=raw[:300],
            step=current_step,
        ))

        logger.info(
            f"Monitor check-in #{self._review_count}: {raw[:100]}... "
            f"[{response.input_tokens}in/{response.output_tokens}out]"
        )

        return raw, response.input_tokens, response.output_tokens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_advice_summary(self) -> str:
        """Summarize prior advice for escalation awareness."""
        if not self._prior_advice:
            return ""

        lines: list[str] = []
        for pa in self._prior_advice:
            lines.append(
                f"- Check-in #{pa.check_in} (step {pa.step}): {pa.guidance[:200]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _is_on_track(text: str) -> bool:
        """Detect 'on track' responses that don't need injection."""
        lower = text.lower().strip()
        # Short affirmative responses
        if len(lower) < 80:
            on_track_phrases = [
                "on track", "no issues", "continue", "looking good",
                "good progress", "proceeding well", "no concerns",
            ]
            return any(phrase in lower for phrase in on_track_phrases)
        return False
