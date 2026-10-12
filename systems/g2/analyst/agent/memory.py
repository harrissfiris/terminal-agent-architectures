"""
Conversation memory management for the ReAct agent.

Handles:
  - Token-aware sliding window (estimate tokens, trim to budget)
  - Key facts extraction and pinning (important discoveries persist)
  - Smart output truncation (prioritize errors over generic output)
  - Dropped-context summaries (compact what was lost)

Design principle: the LLM should always have access to:
  1. System prompt (always)
  2. Task instruction (always, pinned)
  3. Key discoveries (pinned facts)
  4. Dropped-context summary (if anything was trimmed)
  5. Recent conversation turns (sliding window)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from llm.base import Message

# Type for a summarizer callback: takes a list of messages-to-drop,
# returns a compact summary string. Allows optional LLM-powered summaries
# without coupling Memory to the LLM directly.
SummarizerFn = Callable[[list[Message]], Optional[str]]

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Token estimation — fast heuristic, no tokenizer dependency
# ----------------------------------------------------------------
# Qwen2.5 uses a 151K-entry BPE vocabulary.  Empirical measurements on
# mixed code+English text give ~3.8-4.2 chars/token — significantly
# different from GPT-style (~3.2-3.5).  We use 4.0 as the best estimate
# for Qwen models on code-heavy prompts.
CHARS_PER_TOKEN = 4.0

# Per-message overhead in tokens (role, formatting, separators)
MESSAGE_OVERHEAD_TOKENS = 4

# Safety multiplier: even with CHARS_PER_TOKEN=4.0, code with lots of
# short identifiers / JSON / special chars can compress less.  1.20
# gives headroom to avoid context overflow.
TOKEN_ESTIMATE_SAFETY_FACTOR = 1.20


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count (fast heuristic)."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD_TOKENS


def estimate_message_tokens(msg: Message) -> int:
    """Estimate token count for a single message."""
    tokens = MESSAGE_OVERHEAD_TOKENS

    # Content
    content = msg.get("content") or ""
    if content:
        tokens += int(len(content) / CHARS_PER_TOKEN)

    # Tool calls (in assistant messages)
    tool_calls = msg.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        tokens += int(len(func.get("name", "")) / CHARS_PER_TOKEN)
        tokens += int(len(func.get("arguments", "")) / CHARS_PER_TOKEN)
        tokens += 10  # structural overhead per tool call

    return tokens


@dataclass
class KeyFact:
    """An important discovered fact that should persist across the sliding window."""

    content: str
    source: str = ""      # e.g., "step_3", "initial"
    importance: float = 1.0


class ConversationMemory:
    """
    Manages the conversation history for the ReAct agent.

    Architecture:
        ┌─────────────────────────────────────┐
        │  system prompt   (always present)    │
        │  task message    (always present)    │
        │  key facts       (pinned, compact)   │
        │  dropped summary (if turns trimmed)  │
        │  ...sliding window of recent turns...│
        │  latest turn     (current)           │
        └─────────────────────────────────────┘

    Trimming strategy:
      1. Primary gate: turn count (max_turns)
      2. Secondary gate: token budget (max_context_tokens)
      When trimming, the oldest complete assistant→tool pair is removed
      and a compact one-line summary is accumulated.
    """

    def __init__(
        self,
        system_prompt: str,
        task_message: str,
        max_turns: int = 40,
        max_output_chars: int = 16000,
        max_context_tokens: int = 28000,
        summarizer_fn: Optional[SummarizerFn] = None,
        overhead_tokens: int = 0,
    ):
        """
        Args:
            overhead_tokens: Tokens consumed OUTSIDE this memory object on every
                LLM call (tool schema descriptions injected into the prompt +
                the model's output budget ``max_tokens``).  The trim gate is
                ``estimated_input_tokens + overhead_tokens > max_context_tokens``
                so the memory automatically leaves room for both.
        """
        self._system_prompt = system_prompt
        self._task_message = task_message
        self._max_turns = max_turns
        self._max_output_chars = max_output_chars
        self._max_context_tokens = max_context_tokens
        # Chars/token used for trim estimates.  Starts as the heuristic but is
        # updated at runtime (set_chars_per_token) with the model's REAL ratio
        # learned from the server tokenizer — so trimming matches the actual
        # model, not a Qwen-tuned constant.  Prevents context-overflow 400s.
        self._chars_per_token = float(CHARS_PER_TOKEN)
        self._summarizer_fn = summarizer_fn
        self._overhead_tokens = overhead_tokens  # external: tool schemas + max_tokens output

        # Key facts that persist across window slides
        self._key_facts: list[KeyFact] = []

        # Dropped context summary lines
        self._dropped_summaries: list[str] = []

        # Conversation turns (assistant + tool messages interleaved)
        self._turns: list[Message] = []

        # Stats
        self._total_messages_added = 0
        self._messages_dropped = 0
        self._estimated_tokens = 0  # last computed estimate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[dict]] = None,
    ) -> None:
        """Add an assistant response to the conversation."""
        msg: Message = {"role": "assistant"}

        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
            if not content:
                msg["content"] = None  # OpenAI requires content field

        self._turns.append(msg)
        self._total_messages_added += 1
        self._maybe_trim()

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> None:
        """Add a tool execution result to the conversation."""
        # Truncate long outputs
        truncated_result = self._truncate_output(result)

        msg: Message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": truncated_result,
        }
        self._turns.append(msg)
        self._total_messages_added += 1
        # Trim after tool results too — large outputs can spike context in one step
        self._maybe_trim()

    def add_user_message(self, content: str) -> None:
        """Add a user/system injection message (e.g., budget warning)."""
        self._turns.append({"role": "user", "content": content})
        self._total_messages_added += 1
        self._maybe_trim()

    def add_key_fact(self, content: str, source: str = "", importance: float = 1.0) -> None:
        """Pin an important fact that persists across sliding window trims."""
        # Deduplicate
        for existing in self._key_facts:
            if existing.content == content:
                return

        self._key_facts.append(KeyFact(
            content=content,
            source=source,
            importance=importance,
        ))

        # Keep key facts bounded (max 20)
        if len(self._key_facts) > 20:
            self._key_facts.sort(key=lambda f: f.importance, reverse=True)
            self._key_facts = self._key_facts[:20]

    def get_messages(self) -> list[Message]:
        """
        Build the full message list for the LLM.

        Structure:
          1. System prompt
          2. User task message (with key facts + dropped summary injected)
          3. Conversation turns (sliding window)
        """
        messages: list[Message] = []

        # System prompt
        messages.append({"role": "system", "content": self._system_prompt})

        # Task message with key facts and dropped context
        task_content = self._task_message

        if self._key_facts:
            task_content += "\n\n" + self._format_key_facts()

        if self._dropped_summaries:
            task_content += "\n\n" + self._format_dropped_summary()

        messages.append({"role": "user", "content": task_content})

        # Conversation turns
        messages.extend(self._turns)

        # Update token estimate
        self._estimated_tokens = sum(self._est_msg(m) for m in messages)

        return messages

    @property
    def turn_count(self) -> int:
        """Number of messages in the sliding window."""
        return len(self._turns)

    @property
    def total_messages(self) -> int:
        """Total messages ever added."""
        return self._total_messages_added

    @property
    def messages_dropped(self) -> int:
        """Messages dropped by sliding window."""
        return self._messages_dropped

    @property
    def key_facts_count(self) -> int:
        return len(self._key_facts)

    @property
    def estimated_tokens(self) -> int:
        """Last computed token estimate for the full context."""
        return self._estimated_tokens

    def set_chars_per_token(self, value: float) -> None:
        """Update the chars/token ratio used for trim estimates (calibrated
        from the server's real tokenizer by the LLM provider)."""
        if value and value > 0:
            self._chars_per_token = float(value)

    def shrink_for_overflow(self) -> bool:
        """Aggressively drop oldest turns after a context-overflow rejection.

        Removes ~a third of the window (at least two turns) in one shot so the
        retry comfortably fits.  Returns False if there's nothing left to drop
        (only the protected system+task remain)."""
        if len(self._turns) <= 4:
            return False
        target = max(2, len(self._turns) // 3)
        before = len(self._turns)
        for _ in range(target):
            if len(self._turns) <= 4:
                break
            self._drop_oldest_pair()
        return len(self._turns) < before

    # ------------------------------------------------------------------
    # Internal — token estimation (instance-bound chars/token)
    # ------------------------------------------------------------------

    def _est(self, text: str) -> int:
        """Estimate tokens for a text fragment using the calibrated ratio."""
        if not text:
            return 0
        return int(len(text) / self._chars_per_token) + MESSAGE_OVERHEAD_TOKENS

    def _est_msg(self, msg: Message) -> int:
        """Estimate tokens for a message using the calibrated ratio."""
        tokens = MESSAGE_OVERHEAD_TOKENS
        content = msg.get("content") or ""
        if content:
            tokens += int(len(content) / self._chars_per_token)
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            tokens += int(len(func.get("name", "")) / self._chars_per_token)
            tokens += int(len(func.get("arguments", "")) / self._chars_per_token)
            tokens += 10  # structural overhead per tool call
        return tokens

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_trim(self) -> None:
        """Trim old turns if window is too large (by count or tokens)."""
        # Gate 1: Turn count
        while len(self._turns) > self._max_turns * 2:
            self._drop_oldest_pair()

        # Gate 2: Token budget (estimate without building full message list)
        estimated = self._quick_token_estimate()
        while estimated > self._max_context_tokens and len(self._turns) > 4:
            self._drop_oldest_pair()
            estimated = self._quick_token_estimate()

    def _drop_oldest_pair(self) -> None:
        """Remove the oldest assistant→tool pair and create a compact summary.

        Summaries include WHAT was done and WHETHER it succeeded/failed,
        so the model doesn't re-attempt approaches that already failed.

        If an LLM summarizer callback is configured, it is tried first
        for a richer summary.  Falls back to keyword extraction on failure.
        """
        if not self._turns:
            return

        # Collect the messages we're about to drop (assistant + its tool results)
        dropped_messages: list[Message] = []

        removed = self._turns.pop(0)
        self._messages_dropped += 1
        dropped_messages.append(removed)

        if removed.get("role") == "assistant":
            while self._turns and self._turns[0].get("role") == "tool":
                tool_msg = self._turns.pop(0)
                self._messages_dropped += 1
                dropped_messages.append(tool_msg)
        elif removed.get("role") == "user":
            pass  # Single user message, nothing else to collect

        # ------- Try LLM-powered summarization first -------
        if self._summarizer_fn is not None and len(dropped_messages) >= 2:
            try:
                llm_summary = self._summarizer_fn(dropped_messages)
                if llm_summary and llm_summary.strip():
                    self._dropped_summaries.append(llm_summary.strip())
                    if len(self._dropped_summaries) > 20:
                        self._dropped_summaries = self._dropped_summaries[-20:]
                    return  # Done — LLM summary succeeded
            except Exception as exc:
                logger.warning(f"LLM summarizer failed, falling back to keywords: {exc}")

        # ------- Keyword-based fallback -------
        summary_parts = self._keyword_summary(dropped_messages)

        if summary_parts:
            summary_line = " | ".join(summary_parts)
            self._dropped_summaries.append(summary_line)

        # Keep dropped summaries bounded (keep last 20)
        if len(self._dropped_summaries) > 20:
            self._dropped_summaries = self._dropped_summaries[-20:]

    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_summary(dropped_messages: list[Message]) -> list[str]:
        """Build a compact keyword-based summary from dropped messages."""
        summary_parts: list[str] = []

        for msg in dropped_messages:
            role = msg.get("role", "")

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    try:
                        import json as _json
                        args = _json.loads(func.get("arguments", "{}"))
                    except Exception:
                        args = {}

                    if name == "bash":
                        cmd = args.get("command", "")[:80]
                        summary_parts.append(f"bash: {cmd}")
                    elif name == "write_file":
                        summary_parts.append(f"write: {args.get('path', '?')}")
                    elif name == "patch_file":
                        summary_parts.append(f"patch: {args.get('path', '?')}")
                    elif name == "read_file":
                        summary_parts.append(f"read: {args.get('path', '?')}")
                    elif name == "think":
                        thought = args.get("content", "")[:80].replace("\n", " ")
                        summary_parts.append(f"think: {thought}")
                    else:
                        summary_parts.append(name)

            elif role == "tool":
                content = msg.get("content", "")
                tool_name = msg.get("name", "")

                if "[ERROR]" in content or "[FAIL" in content.upper():
                    error_snippet = content[:120].replace("\n", " ")
                    summary_parts.append(f"→ FAILED: {error_snippet}")
                elif "[TIMEOUT" in content:
                    summary_parts.append("→ TIMED OUT")
                elif "[Exit code:" in content:
                    import re as _re
                    ec_match = _re.search(r"\[Exit code:\s*(\d+)\]", content)
                    ec = ec_match.group(1) if ec_match else "?"
                    error_lines = [l.strip() for l in content.split("\n")
                                   if l.strip() and any(kw in l.lower() for kw in
                                   ["error", "traceback", "failed", "not found", "no such"])]
                    if error_lines:
                        summary_parts.append(f"→ exit={ec}: {error_lines[-1][:100]}")
                    else:
                        summary_parts.append(f"→ exit={ec}")
                elif "[OK]" in content or "[SUBMITTED]" in content:
                    summary_parts.append("→ OK")
                elif tool_name == "bash" and content.strip():
                    first_line = content.strip().split("\n")[0][:60]
                    summary_parts.append(f"→ ok: {first_line}")

            elif role == "user":
                content = msg.get("content", "")[:80]
                summary_parts.append(f"system msg: {content}")

        return summary_parts

    def _quick_token_estimate(self) -> int:
        """Fast token estimate without building the full message list.

        Includes:
          - All in-memory content (system prompt, task, key facts, dropped
            summaries, conversation turns)
          - External overhead (``_overhead_tokens``): tool schema descriptions
            injected per-call outside this object + the model's output budget
          - A 20 % safety factor to compensate for tokenizer variance
            (Qwen2.5 with 151K vocab averages ~4.0 chars/token for code).
        """
        tokens = self._est(self._system_prompt)
        tokens += self._est(self._task_message)

        # Key facts and dropped summary overhead (rough)
        tokens += sum(self._est(f.content) for f in self._key_facts)
        tokens += sum(self._est(s) for s in self._dropped_summaries)

        # Conversation turns
        tokens += sum(self._est_msg(m) for m in self._turns)

        # External overhead invisible to this object (tool schemas + max_tokens)
        tokens += self._overhead_tokens

        # Safety factor: accounts for tokenizer underestimation on code/JSON
        return int(tokens * TOKEN_ESTIMATE_SAFETY_FACTOR)

    def _truncate_output(self, text: str) -> str:
        """
        Truncate tool output if too long.

        Smart strategy: preserve error lines and keep head+tail.
        """
        if len(text) <= self._max_output_chars:
            return text

        # Try to extract and prioritize error lines
        error_lines = self._extract_error_lines(text)
        error_block = ""
        if error_lines:
            error_block = "\n[Key errors/warnings extracted:]\n" + "\n".join(error_lines[:10])

        # Compute remaining budget after error block
        error_block_len = len(error_block)
        remaining = self._max_output_chars - error_block_len - 60  # margin for markers

        if remaining < 200:
            # Error block alone is too large, just do simple truncation
            remaining = self._max_output_chars - 60

        # Head + tail split (40% head, 60% tail — tail usually has the useful info)
        head_budget = int(remaining * 0.4)
        tail_budget = remaining - head_budget

        head = text[:head_budget]
        tail = text[-tail_budget:]
        dropped = len(text) - head_budget - tail_budget

        parts = [head]
        parts.append(f"\n\n... [{dropped} chars truncated] ...")
        if error_block and remaining >= 200:
            parts.append(error_block)
        parts.append(f"\n\n{tail}")

        return "".join(parts)

    @staticmethod
    def _extract_error_lines(text: str) -> list[str]:
        """Extract lines that look like errors, warnings, or failures."""
        error_patterns = re.compile(
            r"(error|traceback|exception|failed|failure|fatal|"
            r"not found|no such|permission denied|syntax error|"
            r"import error|module.*not found|assert.*error|"
            r"FAIL|FAILED|ERROR)",
            re.IGNORECASE,
        )
        lines = text.split("\n")
        matches = []
        for line in lines:
            stripped = line.strip()
            if stripped and error_patterns.search(stripped):
                # Keep it short
                matches.append(stripped[:150])
        return matches

    def _format_key_facts(self) -> str:
        """Format key facts for injection into the task message."""
        if not self._key_facts:
            return ""

        lines = ["## Key Discoveries (from previous steps)"]
        for fact in self._key_facts:
            lines.append(f"- {fact.content}")
        return "\n".join(lines)

    def _format_dropped_summary(self) -> str:
        """Format dropped context summary with categorized structure."""
        if not self._dropped_summaries:
            return ""

        # Categorize summaries by content type
        discovered: list[str] = []
        created: list[str] = []
        failed: list[str] = []
        progress: list[str] = []

        for summary in self._dropped_summaries:
            lower = summary.lower()
            if any(w in lower for w in ("error", "fail", "traceback", "exit=1", "exit=2", "not found")):
                failed.append(summary)
            elif any(w in lower for w in ("write_file", "patch_file", "created", "wrote", "installed")):
                created.append(summary)
            elif any(w in lower for w in ("column", "schema", "found", "read_file", "discovered", "csv")):
                discovered.append(summary)
            else:
                progress.append(summary)

        lines = [
            f"## Earlier Actions ({self._messages_dropped} messages trimmed from context)",
            "DO NOT repeat failed approaches listed below.",
        ]
        if discovered:
            lines.append("### Discovered")
            for s in discovered:
                lines.append(f"- {s}")
        if created:
            lines.append("### Created/Modified")
            for s in created:
                lines.append(f"- {s}")
        if failed:
            lines.append("### Failed (do NOT retry)")
            for s in failed:
                lines.append(f"- {s}")
        if progress:
            lines.append("### Other Progress")
            for s in progress:
                lines.append(f"- {s}")
        return "\n".join(lines)
