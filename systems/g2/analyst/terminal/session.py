"""
Safe terminal session wrapper with structured command execution.

Wraps TerminalBench's TmuxSession to provide:
  - Marker-based command execution with reliable output capture
  - Automatic output truncation to prevent context overflow
  - File read/write via tmux
  - Safety prelude (disable pagers) and escape hatch
  - ANSI escape code stripping
  - Exit code extraction

Usage:
    from terminal_bench.terminal.tmux_session import TmuxSession
    from terminal import SafeTerminal

    term = SafeTerminal(tmux_session)
    term.initialize()
    result = term.exec("ls /app")
    print(result.output, result.exit_code)
"""
from __future__ import annotations

import base64
import re
import shlex
import time
import uuid
from dataclasses import dataclass
from typing import Optional

try:
    from terminal_bench.terminal.tmux_session import TmuxSession
except ImportError:
    # Provide a stub for development/testing outside TerminalBench
    from typing import Protocol

    class TmuxSession(Protocol):  # type: ignore[no-redef]
        def send_keys(self, keys: list) -> None: ...
        def capture_pane(self) -> str: ...


# Regex to strip ANSI escape codes (colors, cursor movement, etc.)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# Also match OSC sequences and other escapes
ANSI_EXTENDED_RE = re.compile(r"\x1b(?:\[[0-9;]*[a-zA-Z]|\][^\x07]*\x07|\([AB012])")


@dataclass
class CommandResult:
    """Result of executing a terminal command."""

    output: str                       # Captured output (may be truncated)
    exit_code: Optional[int] = None   # Exit code if captured, None if unknown
    timed_out: bool = False           # True if command timed out
    truncated: bool = False           # True if output was truncated

    @property
    def success(self) -> bool:
        """True if exit code is 0 (or None = unknown, treated as success)."""
        return self.exit_code is None or self.exit_code == 0


class SafeTerminal:
    """
    Safe wrapper around TmuxSession with structured command execution.

    Key features:
    - Marker-based capture: unique markers bracket each command for reliable
      output extraction even when pane contains prior output.
    - Polling with backoff: waits for command completion via end marker.
    - Output truncation: prevents LLM context overflow.
    - File I/O: read_file() via cat, write_file() via heredoc.
    - Safety: disables pagers, provides escape hatch for stuck terminals.
    """

    def __init__(
        self,
        session: TmuxSession,
        max_output_chars: int = 16000,
        default_timeout: float = 30.0,
    ):
        self._session = session
        self._max_output = max_output_chars
        self._default_timeout = default_timeout
        self._counter = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Run safety prelude: disable pagers, configure shell for non-interactive use."""
        if self._initialized:
            return
        safety_commands = [
            "cd /app 2>/dev/null || true",       # Ensure we start in /app
            "export PAGER=cat",
            "export MANPAGER=cat",
            "export GIT_PAGER=cat",
            "export LESS='-FRX'",
            "export TERM=dumb",
            "alias less=cat 2>/dev/null || true",
            "alias more=cat 2>/dev/null || true",
            "stty -ixon 2>/dev/null || true",       # Disable Ctrl-S/Q flow control
            "set +o history 2>/dev/null || true",    # Cleaner output
        ]
        for cmd in safety_commands:
            self._send(cmd)
        time.sleep(0.3)
        self._initialized = True

    def escape(self) -> None:
        """Try to escape a stuck terminal state (pager, prompt, hung process)."""
        # Send 'q' (quit pagers like less/man)
        self._session.send_keys(["q", "Enter"])
        time.sleep(0.05)
        # Ctrl-C twice (interrupt hanging processes)
        self._session.send_keys(["\x03"])
        time.sleep(0.05)
        self._session.send_keys(["\x03"])
        time.sleep(0.05)
        # Ctrl-D (EOF, exits some interactive prompts)
        self._session.send_keys(["\x04"])
        time.sleep(0.05)
        # Wake the shell
        self._session.send_keys(["Enter"])
        time.sleep(0.15)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def exec(
        self,
        command: str,
        timeout: Optional[float] = None,
        robust: bool = False,
    ) -> CommandResult:
        """
        Execute a command and capture output with start/end markers.

        Uses polling with exponential backoff to wait for the end marker.
        Returns structured CommandResult with output, exit code, and status.

        Args:
            command: Shell command to execute (may be multi-line).
            timeout: Max seconds to wait for completion (default: 30s).
            robust: Send the command base64-encoded and ``eval`` it in the
                current shell instead of typing it line-by-line.  This makes
                complex one-liners (nested quotes, awk/sed with backslash
                escapes, embedded ``\\n``) survive the tmux keystroke layer
                intact — they otherwise get mangled in transit.  Semantically
                identical for well-formed commands; ``eval`` keeps cd/env
                persistence.  Used for model-issued bash.

        Returns:
            CommandResult with captured output.
        """
        timeout = timeout if timeout is not None else self._default_timeout
        marker = self._make_marker()
        end_marker = f"{marker}_END"

        # Echo start marker
        self._send(f"echo '{marker}'")

        if robust:
            # base64 is tmux-safe ([A-Za-z0-9+/=]) → no character gets eaten or
            # reinterpreted in transit.  Decoded + eval'd in the current shell.
            b64 = base64.b64encode(command.encode("utf-8", "surrogatepass")).decode("ascii")
            self._send(f"eval \"$(echo '{b64}' | base64 -d)\"")
        else:
            # Send command line(s)
            for line in command.splitlines():
                stripped = line.rstrip()
                if stripped:
                    self._send(stripped)

        # Echo end marker with exit code
        self._send(f"__tb_ec=$?; echo '{end_marker}':$__tb_ec")

        # Poll for end marker
        deadline = time.monotonic() + timeout
        wait = 0.15
        while time.monotonic() < deadline:
            time.sleep(wait)
            pane = self._capture_raw()

            if end_marker in pane:
                output = self._extract_between(pane, marker, end_marker)
                exit_code = self._extract_exit_code(pane, end_marker)
                output, truncated = self._truncate(output)
                return CommandResult(
                    output=self._clean(output),
                    exit_code=exit_code,
                    truncated=truncated,
                )

            # Exponential backoff, cap at 2s
            wait = min(wait * 1.5, 2.0)

        # Timeout — return whatever we captured
        pane = self._capture_raw()
        output = self._extract_after(pane, marker)
        output, truncated = self._truncate(output)

        return CommandResult(
            output=self._clean(output),
            exit_code=None,
            timed_out=True,
            truncated=truncated,
        )

    def exec_silent(self, command: str) -> None:
        """Execute a command without waiting for or capturing output."""
        self._send(command)
        time.sleep(0.1)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def read_file(self, path: str, max_chars: int = 50000) -> str:
        """
        Read file contents via cat.

        Returns file content as string. Truncates if larger than max_chars.
        Returns empty string if file doesn't exist or read fails.
        """
        result = self.exec(f"cat {shlex.quote(path)}", timeout=10.0)
        content = result.output
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...<truncated>..."
        return content

    def write_file(self, path: str, content: str) -> None:
        """
        Write file via heredoc with unique EOF delimiter.

        Uses quoted delimiter to prevent shell expansion within content.
        Automatically creates parent directories.
        """
        delim = f"EOF_{uuid.uuid4().hex[:10]}"

        # Ensure content ends with newline (heredoc requirement)
        if content and not content.endswith("\n"):
            content += "\n"

        # Create parent directory
        parent = "/".join(path.split("/")[:-1])
        if parent:
            self._send(f"mkdir -p {shlex.quote(parent)}")

        # Write via heredoc (quoted delimiter prevents expansion)
        self._send(f"cat << '{delim}' > {shlex.quote(path)}")
        for line in content.splitlines():
            self._session.send_keys([line, "Enter"])
        self._send(delim)
        time.sleep(0.15)

    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        result = self.exec(
            f"test -f {shlex.quote(path)} && echo EXISTS || echo NOTFOUND",
            timeout=5.0,
        )
        return "EXISTS" in result.output

    def dir_exists(self, path: str) -> bool:
        """Check if a directory exists."""
        result = self.exec(
            f"test -d {shlex.quote(path)} && echo EXISTS || echo NOTFOUND",
            timeout=5.0,
        )
        return "EXISTS" in result.output

    def list_dir(self, path: str = ".") -> str:
        """List directory contents."""
        result = self.exec(f"ls -la {shlex.quote(path)}", timeout=5.0)
        return result.output

    def get_cwd(self) -> str:
        """Get current working directory."""
        result = self.exec("pwd", timeout=5.0)
        return result.output.strip()

    # ------------------------------------------------------------------
    # Raw pane access
    # ------------------------------------------------------------------

    def capture_pane(self) -> str:
        """Get current visible pane contents (ANSI-cleaned)."""
        return self._clean(self._capture_raw())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, text: str) -> None:
        """Send a command line to the terminal."""
        self._session.send_keys([text, "Enter"])

    def _capture_raw(self) -> str:
        """Capture raw pane contents from tmux."""
        return self._session.capture_pane() or ""

    def _make_marker(self) -> str:
        """Generate a unique marker string."""
        self._counter += 1
        uid = uuid.uuid4().hex[:8]
        return f"__TB_{self._counter}_{uid}__"

    def _extract_between(
        self, text: str, start_marker: str, end_marker: str
    ) -> str:
        """Extract text between start and end markers."""
        start_idx = text.rfind(start_marker)
        end_idx = text.rfind(end_marker)

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return text[-self._max_output:]

        # Skip the start marker line
        after_start = text[start_idx + len(start_marker) :]
        first_newline = after_start.find("\n")
        if first_newline != -1:
            after_start = after_start[first_newline + 1 :]

        # Cut before end marker
        end_in_after = after_start.find(end_marker)
        if end_in_after != -1:
            content = after_start[:end_in_after]
        else:
            content = after_start

        # Strip echo lines of the command itself (first line might be cmd echo)
        lines = content.splitlines()
        # Remove trailing empty lines and prompt-like lines
        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)

    def _extract_after(self, text: str, marker: str) -> str:
        """Extract text after a marker (fallback for timeout/missing end marker)."""
        idx = text.rfind(marker)
        if idx == -1:
            return text[-self._max_output:]

        after = text[idx + len(marker) :]
        first_newline = after.find("\n")
        if first_newline != -1:
            after = after[first_newline + 1 :]

        return after.strip()

    def _extract_exit_code(self, text: str, end_marker: str) -> Optional[int]:
        """Extract exit code from end marker line (format: MARKER:N)."""
        idx = text.rfind(end_marker)
        if idx == -1:
            return None

        after = text[idx + len(end_marker) :]
        # Get everything until newline
        line_end = after.find("\n")
        tail = after[:line_end] if line_end != -1 else after

        # Parse :N format
        if tail.startswith(":"):
            try:
                return int(tail[1:].strip())
            except ValueError:
                pass
        return None

    def _truncate(self, text: str) -> tuple[str, bool]:
        """Truncate text if too long, preserving error context.

        Line-based truncation with smart error preservation:
        - Default: keep first 40 + last 40 lines, cap at 100 total.
        - pip/apt output: keep last 5 lines only (install output is noise).
        - Errors (Traceback, Error, FAILED) in the truncated middle
          are rescued with 1 line of surrounding context each.
        """
        if len(text) <= self._max_output:
            return text, False

        lines = text.splitlines()

        # ── Special case: pip / apt install → keep only tail ───────────
        first_line = lines[0].lower() if lines else ""
        if any(kw in first_line for kw in ("pip install", "apt-get install", "apt install")):
            tail = lines[-5:]
            truncated = (
                f"[{len(lines) - 5} install output lines truncated]\n"
                + "\n".join(tail)
            )
            return truncated, True

        # ── General case: first N + error rescue + last N ──────────────
        HEAD_LINES = 40
        TAIL_LINES = 40
        MAX_LINES = 100

        if len(lines) <= MAX_LINES:
            # Fits within line cap, but exceeded char cap — fall back to
            # char-based truncation.
            third = self._max_output // 3
            head = text[:third]
            tail = text[-third * 2:]
            dropped = len(text) - self._max_output
            return f"{head}\n\n... [{dropped} chars truncated] ...\n\n{tail}", True

        head = lines[:HEAD_LINES]
        tail = lines[-TAIL_LINES:]
        middle = lines[HEAD_LINES:-TAIL_LINES]

        # Scan middle for error lines and rescue them
        error_keywords = ("traceback", "error", "exception", "failed", "fatal")
        rescued: list[str] = []
        for i, line in enumerate(middle):
            low = line.lower()
            if any(kw in low for kw in error_keywords):
                # Keep the error line + 1 line of context above (if available)
                if i > 0 and middle[i - 1] not in rescued:
                    rescued.append(middle[i - 1])
                rescued.append(line)

        dropped = len(middle) - len(rescued)
        parts = head
        if rescued:
            parts.append(f"\n... [{dropped} lines truncated, {len(rescued)} error lines preserved] ...\n")
            parts.extend(rescued)
            parts.append(f"\n... [end of preserved error lines] ...\n")
        else:
            parts.append(f"\n... [{dropped} lines truncated] ...\n")
        parts.extend(tail)

        truncated = "\n".join(parts)
        return truncated, True

    @staticmethod
    def _clean(text: str) -> str:
        """Remove ANSI escape codes from text."""
        return ANSI_EXTENDED_RE.sub("", text)
