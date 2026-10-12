"""
Tool executor — bridges tool calls from the LLM to actual actions.

Maps tool names to execution functions that operate on a SafeTerminal.
Returns structured results as strings for the LLM's context.
"""
from __future__ import annotations

import logging
import shlex
from typing import Any, Optional

from terminal.session import SafeTerminal
from agent.safety import CommandGuard, Verdict

logger = logging.getLogger(__name__)

# Fast content hash for dedup
import hashlib

def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


class ToolExecutor:
    """
    Executes tool calls against a SafeTerminal.

    Thread-safe: all terminal operations are sequential.
    """

    def __init__(
        self,
        terminal: SafeTerminal,
        default_timeout: float = 30.0,
        long_timeout: float = 120.0,
        command_guard: Optional[CommandGuard] = None,
    ):
        self._terminal = terminal
        self._default_timeout = default_timeout
        self._long_timeout = long_timeout
        self._guard = command_guard or CommandGuard()

        # Track state for submit
        self._submitted = False
        self._final_answer: Optional[str] = None

        # Stats
        self._bash_calls = 0
        self._write_calls = 0
        self._patch_calls = 0
        self._read_calls = 0
        self._think_calls = 0
        self._last_exit_code: Optional[int] = None

        # File-write dedup: path → hash of last-written content
        self._write_hashes: dict[str, str] = {}

        # Budget awareness (set externally per step by the ReAct loop)
        self._current_step: int = 0
        self._max_steps: int = 60
        self._elapsed_seconds: float = 0.0
        self._max_wallclock: float = 600.0

    @property
    def submitted(self) -> bool:
        return self._submitted

    @property
    def final_answer(self) -> Optional[str]:
        return self._final_answer

    @property
    def stats(self) -> dict[str, int]:
        return {
            "bash_calls": self._bash_calls,
            "write_calls": self._write_calls,
            "patch_calls": self._patch_calls,
            "read_calls": self._read_calls,
            "think_calls": self._think_calls,
            **{f"guard_{k}": v for k, v in self._guard.stats.items()},
        }

    @property
    def last_bash_exit_code(self) -> Optional[int]:
        """Exit code from the most recent bash command (for loop detection)."""
        return self._last_exit_code

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool call and return the result as a string.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments (parsed from LLM output).

        Returns:
            Result string to feed back to the LLM.
        """
        try:
            if tool_name == "bash":
                return self._exec_bash(arguments)
            elif tool_name == "write_file":
                return self._exec_write_file(arguments)
            elif tool_name == "patch_file":
                return self._exec_patch_file(arguments)
            elif tool_name == "read_file":
                return self._exec_read_file(arguments)
            elif tool_name == "search_files":
                return self._exec_search_files(arguments)
            elif tool_name == "list_dir":
                return self._exec_list_dir(arguments)
            elif tool_name == "think":
                return self._exec_think(arguments)
            elif tool_name == "submit":
                return self._exec_submit(arguments)
            else:
                return f"[ERROR] Unknown tool: {tool_name}. Available: bash, write_file, patch_file, read_file, think, submit"

        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return f"[ERROR] Tool '{tool_name}' raised an exception: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _exec_bash(self, args: dict[str, Any]) -> str:
        """Execute a shell command."""
        command = args.get("command", "")
        if not command or not command.strip():
            return "[ERROR] Empty command. Provide a non-empty 'command' argument."

        # Safety check
        safety = self._guard.check(command)
        if safety.verdict == Verdict.BLOCK:
            return safety.message

        # Use rewritten command if applicable
        command = safety.command
        warning_prefix = f"{safety.message}\n" if safety.message else ""

        # Determine timeout — use long timeout for known slow commands
        timeout = args.get("timeout")
        if timeout is not None:
            timeout = float(timeout)
        else:
            timeout = self._classify_timeout(command)

        self._bash_calls += 1
        # Route fragile commands through the base64 transport so they reach the
        # shell intact.  Simple commands keep the plain line-by-line path
        # (no behaviour change for the common case).
        result = self._terminal.exec(
            command, timeout=timeout, robust=self._needs_robust_transport(command),
        )
        self._last_exit_code = result.exit_code

        # If the command timed out, recover the terminal to prevent
        # hung processes from polluting subsequent commands
        if result.timed_out:
            self._terminal.escape()

        # Build structured result
        parts: list[str] = []

        if result.timed_out:
            parts.append(f"[TIMEOUT after {timeout:.0f}s]")

        if result.output.strip():
            parts.append(result.output)
        else:
            parts.append("[No output]")

        if result.exit_code is not None and result.exit_code != 0:
            parts.append(f"\n[Exit code: {result.exit_code}]")

        if result.truncated:
            parts.append("[Output was truncated]")

        output = "\n".join(parts)
        return warning_prefix + output if warning_prefix else output

    @staticmethod
    def _needs_robust_transport(command: str) -> bool:
        """Heuristic: does this command risk corruption over the tmux layer?

        Simple single-line commands (the overwhelming majority, e.g. Qwen's)
        take the unchanged plain path.  We only switch to the base64 transport
        for the fragile cases that actually get mangled in transit: multi-line
        scripts, very long lines, backslash escapes, or heavy/mixed quoting.
        """
        if "\n" in command:
            return True
        if len(command) > 350:
            return True
        if "\\" in command:           # backslash escapes (awk \n, \", \$ …)
            return True
        # Mixed or heavy quoting — the case that broke DeepSeek's awk one-liner.
        if command.count('"') + command.count("'") >= 4:
            return True
        return False

    def _exec_write_file(self, args: dict[str, Any]) -> str:
        """Write content to a file."""
        path = args.get("path", "")
        content = args.get("content", "")

        if not path:
            return "[ERROR] Missing 'path' argument."

        if not path.startswith("/"):
            return f"[ERROR] Path must be absolute (starts with /). Got: {path}"

        # Dedup check: warn if writing identical content to same path
        new_hash = _content_hash(content)
        if path in self._write_hashes and self._write_hashes[path] == new_hash:
            return (
                f"[WARNING] You are writing IDENTICAL content to {path} "
                f"(same as your previous write). This won't fix anything. "
                f"You MUST change your approach — the bug is elsewhere or "
                f"your logic needs a different fix."
            )
        self._write_hashes[path] = new_hash

        self._write_calls += 1
        self._terminal.write_file(path, content)

        # Verify the write
        exists = self._terminal.file_exists(path)
        if exists:
            # Quick size check
            result = self._terminal.exec(
                f"wc -c < {shlex.quote(path)}", timeout=5.0
            )
            size = result.output.strip()
            return f"[OK] Wrote {size} bytes to {path}"
        else:
            return f"[WARNING] Write command sent for {path}, but file existence check failed. It may still have been written."

    def _exec_patch_file(self, args: dict[str, Any]) -> str:
        """Surgically replace a text snippet in a file."""
        path = args.get("path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")

        if not path:
            return "[ERROR] Missing 'path' argument."
        if not path.startswith("/"):
            return f"[ERROR] Path must be absolute (starts with /). Got: {path}"
        if not old_text:
            return "[ERROR] Missing 'old_text' argument. Provide the exact text to replace."

        # Check file exists
        if not self._terminal.file_exists(path):
            return f"[ERROR] File not found: {path}"

        self._patch_calls += 1

        # Read current file content
        current = self._terminal.read_file(path)

        # Count occurrences
        count = current.count(old_text)
        if count == 0:
            # Provide helpful feedback
            return (
                f"[ERROR] old_text not found in {path}. "
                f"Make sure it matches EXACTLY (including whitespace/indentation). "
                f"Use read_file to see the exact content first."
            )
        if count > 1:
            return (
                f"[WARNING] old_text found {count} times in {path}. "
                f"Include more context lines to make the match unique. "
                f"Only the FIRST occurrence was replaced."
            )

        # Perform replacement (first occurrence only for safety)
        patched = current.replace(old_text, new_text, 1)

        # Dedup check: resulting content identical to what was already written?
        patched_hash = _content_hash(patched)
        if path in self._write_hashes and self._write_hashes[path] == patched_hash:
            return (
                f"[WARNING] This patch produces IDENTICAL content to your previous write "
                f"to {path}. The patch is a no-op. Try a different fix."
            )
        self._write_hashes[path] = patched_hash

        # Write back
        self._terminal.write_file(path, patched)

        # Verify
        verified = self._terminal.read_file(path)
        if new_text in verified:
            lines_before = old_text.count("\n") + 1
            lines_after = new_text.count("\n") + 1
            return f"[OK] Patched {path}: replaced {lines_before} line(s) with {lines_after} line(s)."
        else:
            return f"[WARNING] Patch applied to {path} but verification uncertain. Check the file."

    def _exec_read_file(self, args: dict[str, Any]) -> str:
        """Read a file's contents."""
        path = args.get("path", "")

        if not path:
            return "[ERROR] Missing 'path' argument."

        self._read_calls += 1
        content = self._terminal.read_file(path)

        if not content.strip():
            # Check if file exists
            exists = self._terminal.file_exists(path)
            if not exists:
                return f"[ERROR] File not found: {path}"
            return f"[OK] File {path} exists but is empty."

        return content

    def _exec_think(self, args: dict[str, Any]) -> str:
        """Reasoning scratchpad — returns current budget status for situational awareness."""
        content = args.get("content", "")
        self._think_calls += 1
        # Return actionable step + time budget so the model can plan
        total_tool_calls = (
            self._bash_calls + self._write_calls + self._patch_calls
            + self._read_calls + self._think_calls
        )
        steps_remaining = max(0, self._max_steps - self._current_step)
        time_remaining = max(0, self._max_wallclock - self._elapsed_seconds)
        return (
            f"[OK] Reasoning noted. | "
            f"Step {self._current_step}/{self._max_steps} ({steps_remaining} remaining) | "
            f"Time: {self._elapsed_seconds:.0f}s/{self._max_wallclock:.0f}s ({time_remaining:.0f}s remaining) | "
            f"Tools: bash={self._bash_calls}, write={self._write_calls}, "
            f"patch={self._patch_calls}, read={self._read_calls}, think={self._think_calls} "
            f"(total={total_tool_calls})"
        )

    def _exec_search_files(self, args: dict[str, Any]) -> str:
        """Search for a pattern across files using grep."""
        pattern = args.get("pattern", "")
        if not pattern:
            return "[ERROR] Missing 'pattern' argument."

        directory = args.get("directory", "/app")
        # Sanitize: only allow absolute paths
        if not directory.startswith("/"):
            return f"[ERROR] Directory must be an absolute path. Got: {directory}"

        cmd = (
            f"grep -rn --include='*' -I "
            f"{shlex.quote(pattern)} {shlex.quote(directory)} "
            f"2>/dev/null | head -80"
        )
        result = self._terminal.exec(cmd, timeout=15.0)
        output = result.output.strip()
        if not output:
            return f"No matches found for {pattern!r} in {directory}"
        return output

    def _exec_list_dir(self, args: dict[str, Any]) -> str:
        """List directory contents."""
        path = args.get("path", "/app")
        if not path.startswith("/"):
            return f"[ERROR] Path must be absolute. Got: {path}"

        cmd = f"ls -la {shlex.quote(path)} 2>&1 | head -60"
        result = self._terminal.exec(cmd, timeout=10.0)
        return result.output.strip() or f"[OK] Directory {path} is empty."

    def _exec_submit(self, args: dict[str, Any]) -> str:
        """Mark task as complete."""
        answer = args.get("answer", "Task completed.")
        self._submitted = True
        self._final_answer = answer
        return f"[SUBMITTED] {answer}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_timeout(self, command: str) -> float:
        """Heuristically determine appropriate timeout for a command."""
        cmd_lower = command.lower().strip()

        # Long-running commands
        long_patterns = [
            "pip install", "pip3 install",
            "apt-get install", "apt install",
            "npm install", "yarn install",
            "make", "cmake", "gcc", "g++", "gfortran",
            "cargo build", "go build",
            "python3 -m pytest", "pytest",
            "mvn ", "gradle ",
            "docker ", "git clone",
            "wget ", "curl ", "tar ",
            "train", "epoch",
        ]

        for pattern in long_patterns:
            if pattern in cmd_lower:
                return self._long_timeout

        # Test runners
        if "test" in cmd_lower or "run-tests" in cmd_lower:
            return self._long_timeout

        return self._default_timeout
