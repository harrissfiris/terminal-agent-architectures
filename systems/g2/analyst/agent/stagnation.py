"""
Stagnation detection and progress tracking for Stage 2.

Two independent, zero-LLM-cost mechanisms that run after each tool call:
  - StagnationDetector: flags repeated similar commands (SequenceMatcher)
  - ProgressTracker: flags lack of new unique errors over N steps
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger(__name__)

# Reuse error keywords from the existing codebase pattern
ERROR_KEYWORDS = re.compile(
    r"(traceback|error:|exception:|syntaxerror|nameerror|typeerror|valueerror|"
    r"filenotfounderror|permissionerror|command not found|no such file|"
    r"killed|segfault|fatal error|errno)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# StagnationDetector
# ─────────────────────────────────────────────────────────────────────────────

class StagnationDetector:
    """
    Detects when the agent repeats similar tool calls.

    After each tool call, computes SequenceMatcher similarity against the last
    N calls.  If similarity > threshold for `streak_trigger` consecutive calls,
    returns a context-aware warning string.
    """

    def __init__(
        self,
        window: int = 5,
        threshold: float = 0.80,
        streak_trigger: int = 3,
    ):
        self._window = window
        self._threshold = threshold
        self._streak_trigger = streak_trigger

        self._history: list[str] = []   # last N (tool_name + args) strings
        self._streak: int = 0           # consecutive high-similarity calls
        self._warnings_issued: int = 0
        self._total_checks: int = 0
        self._events: list[dict] = []

    def check(self, tool_name: str, args_str: str) -> Optional[str]:
        """
        Record a tool call and check for stagnation.

        Returns a warning string if stagnation detected, else None.
        """
        self._total_checks += 1
        current = f"{tool_name}:{args_str}"

        # Compare against recent history
        is_similar = False
        if self._history:
            best_ratio = max(
                SequenceMatcher(None, current, prev).ratio()
                for prev in self._history[-self._window:]
            )
            is_similar = best_ratio > self._threshold

        # Update history (keep bounded)
        self._history.append(current)
        if len(self._history) > self._window * 2:
            self._history = self._history[-self._window:]

        # Update streak
        if is_similar:
            self._streak += 1
        else:
            self._streak = 0

        # Check trigger
        if self._streak >= self._streak_trigger:
            self._streak = 0  # reset to allow re-triggering later
            self._warnings_issued += 1
            warning = self._build_warning(tool_name, args_str)
            self._events.append({
                "check_number": self._total_checks,
                "tool": tool_name,
                "warning": warning,
            })
            logger.info(f"Stagnation detected at check {self._total_checks}: {warning[:100]}")
            return warning

        return None

    def _build_warning(self, tool_name: str, args_str: str) -> str:
        """Build a pattern-specific warning message with escalation tiers."""
        if self._warnings_issued >= 4:
            base = (
                f"CRITICAL: You have received {self._warnings_issued} stagnation warnings "
                "and continue repeating the same approach. STOP. "
                "Your current strategy has failed. "
                "You MUST try a fundamentally different approach on your next step. "
                "If you cannot make progress, call submit() with your best answer."
            )
        elif self._warnings_issued >= 2:
            base = (
                f"STRONG WARNING: This is stagnation warning #{self._warnings_issued}. "
                "Your current approach is clearly not working. "
                "Change your strategy immediately — try a completely different method."
            )
        else:
            base = (
                "WARNING: You are repeating very similar commands. "
                "Try a completely different approach."
            )

        # Pattern-specific suggestions
        if "python3 -c" in args_str or "python -c" in args_str:
            return (
                f"{base}\n"
                "SUGGESTION: Write your code to a file with write_file() instead "
                "of using inline python3 -c commands."
            )

        if tool_name == "bash" and ("pip install" in args_str or "apt" in args_str):
            return (
                f"{base}\n"
                "SUGGESTION: The same installation keeps failing. "
                "Try an alternative library or approach."
            )

        return base

    @property
    def stats(self) -> dict:
        return {
            "stagnation_warnings_issued": self._warnings_issued,
            "total_checks": self._total_checks,
        }

    @property
    def events(self) -> list[dict]:
        return list(self._events)


# ─────────────────────────────────────────────────────────────────────────────
# ProgressTracker
# ─────────────────────────────────────────────────────────────────────────────

class ProgressTracker:
    """
    Tracks whether the agent is making progress based on unique errors.

    If no new unique error appears in `stale_threshold` consecutive steps,
    injects a warning that the current approach is not working.
    """

    def __init__(self, stale_threshold: int = 5):
        self._stale_threshold = stale_threshold
        self._seen_errors: set[str] = set()
        self._steps_without_new_error: int = 0
        self._warnings_issued: int = 0
        self._total_checks: int = 0

    def check(self, tool_name: str, result: str) -> Optional[str]:
        """
        Record a tool result and check for lack of progress.

        Returns a warning string if no progress detected, else None.
        """
        self._total_checks += 1

        # Only track bash results (where errors appear)
        if tool_name != "bash":
            return None

        # Extract error fingerprints: first 100 chars of each line containing
        # an error keyword.  This distinguishes "SyntaxError line 5" from
        # "SyntaxError line 20" instead of collapsing them into one keyword.
        new_errors: set[str] = set()
        for line in result.split("\n"):
            if ERROR_KEYWORDS.search(line):
                new_errors.add(line.strip()[:100])
        novel = new_errors - self._seen_errors

        if novel:
            self._seen_errors.update(novel)
            self._steps_without_new_error = 0
        else:
            self._steps_without_new_error += 1

        if self._steps_without_new_error >= self._stale_threshold:
            self._steps_without_new_error = 0  # reset
            self._warnings_issued += 1
            return (
                "WARNING: No progress detected in the last "
                f"{self._stale_threshold} bash commands. "
                "Your current approach is not working. "
                "Reconsider your strategy entirely."
            )

        return None

    @property
    def stats(self) -> dict:
        return {
            "progress_warnings_issued": self._warnings_issued,
            "total_checks": self._total_checks,
            "unique_errors_seen": len(self._seen_errors),
        }
