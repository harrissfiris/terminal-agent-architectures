"""
Command safety guard for the ReAct agent.

Provides pre-execution filtering of bash commands to:
  - Block catastrophically dangerous commands (rm -rf /, etc.)
  - Warn about risky patterns (overwrites, system-level changes)
  - Sanitize common LLM mistakes (e.g., interactive commands)

Design principles:
  - Err on the side of ALLOWING — the agent is inside a Docker container,
    so most damage is contained. We only block truly catastrophic patterns.
  - Return structured verdicts so the caller can decide what to do.
  - Never silently modify commands — either allow, warn, or block.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Verdict(Enum):
    """Safety check outcome."""
    ALLOW = "allow"       # Safe to execute
    WARN = "warn"         # Execute but inject a warning
    BLOCK = "block"       # Do NOT execute, return error to agent
    REWRITE = "rewrite"   # Replace command with a safer version


@dataclass
class SafetyResult:
    """Result of a command safety check."""
    verdict: Verdict
    original_command: str
    rewritten_command: Optional[str] = None
    message: str = ""

    @property
    def command(self) -> str:
        """The command to actually execute (may be rewritten)."""
        if self.verdict == Verdict.REWRITE and self.rewritten_command:
            return self.rewritten_command
        return self.original_command


# ----------------------------------------------------------------
# Blocked patterns — truly catastrophic commands
# ----------------------------------------------------------------
BLOCKED_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Recursive delete of root or critical system dirs
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?(-[a-zA-Z]*r[a-zA-Z]*\s+)?/\s*$"), "rm of / (root filesystem)"),
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/\s*$"), "rm -rf /"),
    (re.compile(r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+/\s*$"), "rm -fr /"),
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+/\s*$"), "rm -r /"),
    (re.compile(r"\bmkfs\b"), "mkfs (filesystem format)"),
    (re.compile(r"\bdd\b.*\bof=/dev/[sh]d"), "dd to disk device"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;"), "fork bomb"),
    (re.compile(r"\b(halt|poweroff|shutdown|reboot)\b"), "system shutdown/reboot"),
]

# ----------------------------------------------------------------
# Rewrite patterns — fix common LLM mistakes
# ----------------------------------------------------------------
REWRITE_RULES: list[tuple[re.Pattern, str, str]] = [
    # Interactive editors — BLOCK since they hang the terminal
    (re.compile(r"^\s*(vim?|nano|emacs|vi)\s+"), "__BLOCK__", "Interactive editors (vim/nano/emacs) will hang the terminal. Use the write_file or patch_file tool instead."),
    # Interactive pagers — disable
    (re.compile(r"\|\s*(less|more)\s*$"), "| cat", "Replaced interactive pager with cat."),
    # apt without -y — would hang waiting for confirmation
    (re.compile(r"(\bapt(?:-get)?\s+install)\b(?!.*-y)"), r"\1 -y", "Auto-added -y flag to prevent interactive apt prompt."),
]

# ----------------------------------------------------------------
# Auto-fix patterns — silently fix common LLM mistakes
# ----------------------------------------------------------------

def _auto_fix_pip(command: str) -> str | None:
    """Auto-inject --break-system-packages into pip install commands in Docker."""
    if not re.search(r"\bpip3?\s+install\b", command):
        return None
    if "--break-system-packages" in command or "--user" in command or "-t " in command:
        return None
    # Insert --break-system-packages after 'pip install' or 'pip3 install'
    fixed = re.sub(
        r"(\bpip3?\s+install)\b",
        r"\1 --break-system-packages",
        command,
    )
    return fixed if fixed != command else None

# ----------------------------------------------------------------
# Warning patterns — risky but allowed
# ----------------------------------------------------------------
WARNING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f"), "Recursive force delete — make sure the path is correct"),
    (re.compile(r"\bchmod\s+777"), "chmod 777 — very permissive"),
    (re.compile(r"\bcurl\b.*\|\s*(ba)?sh"), "Piping curl to shell — potentially dangerous"),
    (re.compile(r"\bwget\b.*\|\s*(ba)?sh"), "Piping wget to shell — potentially dangerous"),
    (re.compile(r">\s*/etc/"), "Writing directly to /etc/"),
    (re.compile(r"\bsudo\b"), "Using sudo"),
    (re.compile(r"\bkill\s+-9"), "Sending SIGKILL"),
]


class CommandGuard:
    """
    Pre-execution safety filter for bash commands.

    Usage:
        guard = CommandGuard()
        result = guard.check("rm -rf /app/build")
        if result.verdict == Verdict.BLOCK:
            ... refuse execution ...
        else:
            ... execute result.command ...
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, WARN verdicts become BLOCK verdicts.
        """
        self._strict = strict
        self._blocked_count = 0
        self._warned_count = 0
        self._rewritten_count = 0

    def check(self, command: str) -> SafetyResult:
        """
        Check a command for safety issues.

        Returns:
            SafetyResult with verdict and optional rewrite/message.
        """
        cmd_stripped = command.strip()

        # 1. Check blocked patterns
        for pattern, reason in BLOCKED_PATTERNS:
            if pattern.search(cmd_stripped):
                self._blocked_count += 1
                logger.warning(f"BLOCKED command: {reason} | cmd={cmd_stripped[:100]}")
                return SafetyResult(
                    verdict=Verdict.BLOCK,
                    original_command=command,
                    message=f"[BLOCKED] Dangerous command detected: {reason}. "
                            f"This command was NOT executed for safety reasons.",
                )

        # 1.5 Auto-fix pip install (silently add --break-system-packages)
        pip_fixed = _auto_fix_pip(cmd_stripped)
        if pip_fixed:
            self._rewritten_count += 1
            logger.info(f"AUTO-FIX pip: {cmd_stripped[:60]} → {pip_fixed[:80]}")
            cmd_stripped = pip_fixed
            command = pip_fixed

        # 2. Check rewrite rules
        for pattern, replacement, reason in REWRITE_RULES:
            match = pattern.search(cmd_stripped)
            if match:
                if replacement == "__BLOCK__":
                    # Hard block — command would hang the terminal
                    self._blocked_count += 1
                    return SafetyResult(
                        verdict=Verdict.BLOCK,
                        original_command=command,
                        message=f"[BLOCKED] {reason}",
                    )
                elif not replacement:
                    # No auto-rewrite — just warn the agent to fix it
                    if reason:
                        self._warned_count += 1
                        return SafetyResult(
                            verdict=Verdict.WARN,
                            original_command=command,
                            message=f"[WARNING] {reason}",
                        )
                else:
                    # Auto-rewrite
                    rewritten = pattern.sub(replacement, cmd_stripped)
                    self._rewritten_count += 1
                    logger.info(f"REWRITE: {cmd_stripped[:60]} → {rewritten[:60]}")
                    return SafetyResult(
                        verdict=Verdict.REWRITE,
                        original_command=command,
                        rewritten_command=rewritten,
                        message=f"[REWRITTEN] {reason}" if reason else "",
                    )

        # 3. Check warning patterns
        for pattern, reason in WARNING_PATTERNS:
            if pattern.search(cmd_stripped):
                self._warned_count += 1
                if self._strict:
                    return SafetyResult(
                        verdict=Verdict.BLOCK,
                        original_command=command,
                        message=f"[BLOCKED (strict mode)] {reason}",
                    )
                return SafetyResult(
                    verdict=Verdict.WARN,
                    original_command=command,
                    message=f"[WARNING] {reason}",
                )

        # 4. All clear
        return SafetyResult(
            verdict=Verdict.ALLOW,
            original_command=command,
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "blocked": self._blocked_count,
            "warned": self._warned_count,
            "rewritten": self._rewritten_count,
        }
