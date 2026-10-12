"""
Tool executor — bridges tool calls from the LLM to actual actions.

Maps tool names to execution functions that operate on a SafeTerminal.
Returns structured results as strings for the LLM's context.
"""
from __future__ import annotations

import logging
import re as _re
import shlex
from typing import Any, Optional

from terminal.session import SafeTerminal
from agent.safety import CommandGuard, Verdict

logger = logging.getLogger(__name__)

# Fast content hash for dedup
import hashlib

def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


def _explorer_budget_nudge(steps_remaining: int, time_remaining: float) -> str:
    """
    Inline budget warning appended to tool outputs when the Explorer is near
    its step/time limit.  Placed inline in the observation (not as a system
    message) so the model sees it directly in its conversation context.

    Includes the exact Data Summary JSON schema so the model knows precisely
    what to write — preventing it from outputting reasoning text or a tool
    wrapper instead of the actual summary.
    """
    return (
        f"\n\n⚠️ BUDGET WARNING: You have {steps_remaining} step(s) and "
        f"{time_remaining:.0f}s remaining. "
        f"Your NEXT action MUST be think() with the complete Data Summary JSON. "
        f"Do NOT call bash or read_file again. "
        f"Use EXACTLY this JSON structure:\n"
        f'{{\n'
        f'  "task_directory": "/app",\n'
        f'  "files": [\n'
        f'    {{\n'
        f'      "path": "<relative path>",\n'
        f'      "type": "<CSV|DAT|PY|TXT|...>",\n'
        f'      "size_bytes": <int>,\n'
        f'      "sample_head": "<first few lines>",\n'
        f'      "notes": "<key observations about format/content>"\n'
        f'    }}\n'
        f'  ],\n'
        f'  "observations": ["<key finding 1>", "<key finding 2>"]\n'
        f'}}'
    )


def _extract_json_objects(text: str) -> list[dict]:
    """
    Scan text for all top-level JSON objects and return the ones that parse
    successfully as dicts.  Uses a brace-depth scanner so adjacent objects
    (e.g. a tool-wrapper block followed by a Data Summary block) are each
    returned individually — unlike a greedy regex which merges them.
    """
    import json as _json
    results: list[dict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start:i + 1]
                try:
                    obj = _json.loads(candidate)
                    if isinstance(obj, dict):
                        results.append(obj)
                except _json.JSONDecodeError:
                    pass
                start = -1
    return results


class ToolExecutor:
    """
    Executes tool calls against a SafeTerminal.

    Thread-safe: all terminal operations are sequential.
    """

    # Commands allowed in explorer (read-only) mode.  The check uses
    # startswith on the stripped command, which catches both bare commands
    # ("ls /app") and PATH-qualified ones ("/bin/ls /app").
    EXPLORER_READONLY_CMDS: frozenset[str] = frozenset({
        "ls", "cat", "head", "tail", "wc", "file", "stat",
        "find", "du", "grep", "echo", "pwd", "which", "type",
    })

    def __init__(
        self,
        terminal: SafeTerminal,
        default_timeout: float = 30.0,
        long_timeout: float = 120.0,
        command_guard: Optional[CommandGuard] = None,
        explorer_mode: bool = False,
        allowed_tools: Optional[list[str]] = None,
    ):
        self._terminal = terminal
        self._default_timeout = default_timeout
        self._long_timeout = long_timeout
        self._guard = command_guard or CommandGuard()
        self._explorer_mode = explorer_mode
        # When set, any tool call not in this list is blocked at execution time.
        # This enforces role boundaries (e.g. Planner can only think()).
        self._allowed_tools: Optional[list[str]] = allowed_tools

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
        # Tracks how many consecutive think() calls the executor has made
        # without any action tool in between. Used to inject a nudge.
        self._consecutive_think_calls: int = 0
        # Tracks repeated read_file calls on the same path (loop detection).
        self._last_read_path: Optional[str] = None
        self._consecutive_same_read: int = 0

        # Full think() history (not truncated — needed by ExplorerAgent to
        # extract the Data Summary JSON from the final think call).
        self._think_history: list[str] = []

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
    def think_history(self) -> list[str]:
        """Full content of every think() call (not truncated)."""
        return list(self._think_history)

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
            # Role enforcement: block tools outside the allowed subset.
            # This is the hard guardrail that prevents the Planner from
            # accidentally calling bash/submit/write_file despite only
            # receiving think() schemas.
            if self._allowed_tools is not None and tool_name not in self._allowed_tools:
                logger.warning(
                    f"Tool '{tool_name}' blocked by allowed_tools guard "
                    f"(allowed: {self._allowed_tools})"
                )
                return (
                    f"[BLOCKED] Tool '{tool_name}' is not available in this context. "
                    f"You may ONLY use: {', '.join(self._allowed_tools)}. "
                    f"Use think() to reason and produce your output."
                )

            if tool_name == "bash":
                return self._exec_bash(arguments)
            elif tool_name == "write_file":
                return self._exec_write_file(arguments)
            elif tool_name == "patch_file":
                return self._exec_patch_file(arguments)
            elif tool_name == "read_file":
                return self._exec_read_file(arguments)
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

        # Explorer-mode whitelist: only read-only commands are allowed.
        # This is an architectural guarantee — not just a prompt suggestion.
        if self._explorer_mode:
            first_token = command.strip().split()[0].split("/")[-1] if command.strip() else ""
            if first_token not in self.EXPLORER_READONLY_CMDS:
                logger.warning(
                    f"Explorer mode: blocked non-readonly command: {command[:80]!r}"
                )
                return (
                    f"[EXPLORER MODE] Only read-only commands are allowed "
                    f"(ls, cat, head, tail, wc, file, stat, find, du, grep). "
                    f"Command rejected: {command[:80]!r}"
                )
            # Block stdout/stderr redirects to actual files
            # (e.g. echo "data" > /app/f, echo x >> log)
            # Allow: 2>/dev/null, >/dev/null, cat x || cat y, ls | grep, ls; ls
            if _re.search(r'>{1,2}\s*(?!/dev/null)', command):
                logger.warning(
                    f"Explorer mode: blocked file-redirect in command: {command[:80]!r}"
                )
                return (
                    f"[EXPLORER MODE] File-write commands are not available to the "
                    f"Explorer agent — your role is read-only inspection only. "
                    f"The task environment IS writable; the Executor agent will "
                    f"perform all writes after your inspection is complete. "
                    f"Do NOT report the environment as read-only in your Data Summary. "
                    f"Command rejected: {command[:80]!r}"
                )

        # Determine timeout — use long timeout for known slow commands
        timeout = args.get("timeout")
        if timeout is not None:
            timeout = float(timeout)
        else:
            timeout = self._classify_timeout(command)

        self._consecutive_think_calls = 0  # action taken — reset
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
        final = warning_prefix + output if warning_prefix else output

        # Explorer budget nudge: append inline warning when nearing budget.
        # Inline in the tool observation (not a separate system message) so
        # the 32B model sees it directly in its conversation context.
        if self._explorer_mode:
            steps_remaining = max(0, self._max_steps - self._current_step)
            if steps_remaining <= 3 or self._elapsed_seconds > 0.55 * self._max_wallclock:
                time_remaining = max(0, self._max_wallclock - self._elapsed_seconds)
                final += _explorer_budget_nudge(steps_remaining, time_remaining)

        return final

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

        self._consecutive_think_calls = 0  # action taken — reset
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

        self._consecutive_think_calls = 0  # action taken — reset
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

        self._consecutive_think_calls = 0  # action taken — reset

        # Same-file read loop detection
        if path == self._last_read_path:
            self._consecutive_same_read += 1
        else:
            self._last_read_path = path
            self._consecutive_same_read = 1

        self._read_calls += 1
        content = self._terminal.read_file(path)

        if not content.strip():
            # Check if file exists
            exists = self._terminal.file_exists(path)
            if not exists:
                return f"[ERROR] File not found: {path}"
            return f"[OK] File {path} exists but is empty."

        result = content

        # Same-file read nudge: if the model keeps reading the same file,
        # it's not getting new info. Suggest head/tail with line offsets.
        if not self._explorer_mode and self._consecutive_same_read >= 2:
            result += (
                f"\n\n⚠️ REPEATED READ: You have called read_file('{path}') "
                f"{self._consecutive_same_read} times in a row and are seeing "
                f"the same content. read_file always returns the full file — "
                f"calling it again will NOT show more. "
                f"To read specific sections use bash:\n"
                f"  head -50 {path}          # first 50 lines\n"
                f"  tail -50 {path}          # last 50 lines\n"
                f"  sed -n '50,100p' {path}  # lines 50–100\n"
                f"  grep -n 'keyword' {path} # search for a term"
            )

        # Explorer budget nudge: same inline warning as in _exec_bash.
        if self._explorer_mode:
            steps_remaining = max(0, self._max_steps - self._current_step)
            if steps_remaining <= 3 or self._elapsed_seconds > 0.55 * self._max_wallclock:
                time_remaining = max(0, self._max_wallclock - self._elapsed_seconds)
                result += _explorer_budget_nudge(steps_remaining, time_remaining)

        return result

    def _exec_think(self, args: dict[str, Any]) -> str:
        """Reasoning scratchpad — returns current budget status for situational awareness."""
        import json as _json
        content = args.get("content", "")
        self._think_calls += 1
        # Store full content (used by ExplorerAgent to extract Data Summary)
        self._think_history.append(content)

        # Executor-mode consecutive think() detection: if the model has been
        # calling think() repeatedly without any action, it is stuck in a loop.
        if not self._explorer_mode:
            self._consecutive_think_calls += 1

        # Explorer mode auto-stop: if the think content contains a valid Data
        # Summary JSON (has a "files" key at top level), signal completion so
        # the ReAct loop stops immediately rather than iterating to max_steps.
        if self._explorer_mode and content:
            try:
                # Quick check: does any top-level JSON object in content have "files"?
                for obj in _extract_json_objects(content):
                    if "files" in obj and obj.get("tool") != "think":
                        logger.info(
                            "Explorer: valid Data Summary detected in think() — "
                            "signalling completion (auto-stop)"
                        )
                        self._submitted = True
                        self._final_answer = "Explorer complete: Data Summary produced."
                        break
                    # Also check double-wrapped inner content
                    if obj.get("tool") == "think" and isinstance(obj.get("args"), dict):
                        inner = obj["args"].get("content", "")
                        for inner_obj in _extract_json_objects(inner):
                            if "files" in inner_obj:
                                logger.info(
                                    "Explorer: valid Data Summary detected (double-wrapped) — "
                                    "signalling completion (auto-stop)"
                                )
                                self._submitted = True
                                self._final_answer = "Explorer complete: Data Summary produced."
                                break
                        if self._submitted:
                            break
            except Exception:
                pass  # never crash the think() tool

        # Return actionable step + time budget so the model can plan
        total_tool_calls = (
            self._bash_calls + self._write_calls + self._patch_calls
            + self._read_calls + self._think_calls
        )
        steps_remaining = max(0, self._max_steps - self._current_step)
        time_remaining = max(0, self._max_wallclock - self._elapsed_seconds)

        if self._submitted and self._explorer_mode:
            return (
                f"[OK] Data Summary accepted — Explorer phase complete. | "
                f"Step {self._current_step}/{self._max_steps} | "
                f"Time: {self._elapsed_seconds:.0f}s/{self._max_wallclock:.0f}s"
            )

        base = (
            f"[OK] Reasoning noted. | "
            f"Step {self._current_step}/{self._max_steps} ({steps_remaining} remaining) | "
            f"Time: {self._elapsed_seconds:.0f}s/{self._max_wallclock:.0f}s ({time_remaining:.0f}s remaining) | "
            f"Tools: bash={self._bash_calls}, write={self._write_calls}, "
            f"patch={self._patch_calls}, read={self._read_calls}, think={self._think_calls} "
            f"(total={total_tool_calls})"
        )

        # Consecutive think() nudge: fire after 2+ thinks with no action.
        # think() does NOT write files or run commands — the model must call
        # write_file / bash / patch_file to make any real progress.
        if not self._explorer_mode and self._consecutive_think_calls >= 2:
            base += (
                f"\n\n⚠️ ACTION REQUIRED: You have called think() "
                f"{self._consecutive_think_calls} times in a row without taking "
                f"any action (write={self._write_calls}, bash={self._bash_calls}, "
                f"patch={self._patch_calls}). "
                f"think() does NOT write files, run code, or make changes. "
                f"You MUST call write_file, bash, patch_file, or submit in your "
                f"NEXT step. Do NOT call think() again until you have taken action."
            )

        return base

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
