"""
Shared Knowledge Base for the orchestra multi-agent system.

Inspired by Dan Austin's Context Store pattern [austin2025orchestrator]:
  - Structured entries with unique IDs and categories
  - Selective injection: orchestrator chooses which entries each sub-agent sees
  - Orchestrator sees compact summary (IDs + titles) for routing decisions
  - Sub-agents see full content only for orchestrator-selected entries
  - Size management with category-aware eviction

KB variant toggle:
  THESIS_USE_KNOWLEDGE_BASE=0 → KB disabled (vanilla orchestra)
  THESIS_USE_KNOWLEDGE_BASE=1 → KB enabled, note() tool available to all agents

When disabled, the orchestrator is the sole information bridge between agents.
When enabled, agents also share persistent knowledge via the KB.

Key differences from flat notepad:
  1. Entries have categories (discovery, implementation, error, test_result, requirement)
  2. Each entry gets a unique ID (kb_001, kb_002, ...)
  3. Orchestrator can selectively inject relevant entries via context_refs
  4. Size cap with oldest-first eviction prevents context overflow
  5. Compact orchestrator view vs full agent view
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Valid categories for knowledge base entries
CATEGORIES = {
    "discovery",        # File paths, schemas, data formats, environment findings
    "implementation",   # Code written, algorithms chosen, design decisions
    "error",            # Errors encountered and their analysis
    "test_result",      # Test outcomes (pass/fail counts, specific failures)
    "requirement",      # Task requirements extracted from README/tests/instructions
}

# Maximum entries before eviction kicks in
MAX_ENTRIES = 30

# Maximum tokens for KB content injected into agent context
MAX_AGENT_KB_TOKENS = 2000

# Approximate chars per token (Qwen/Gemini)
_CHARS_PER_TOKEN = 4.0


@dataclass
class ContextEntry:
    """A single structured entry in the knowledge base."""

    id: str                  # Unique ID: "kb_001", "kb_002", ...
    category: str            # One of CATEGORIES
    title: str               # Short descriptive title
    content: str             # Full content of the entry
    author: str              # Role that created it (explorer, coder, verifier)
    cycle: int = 0           # Orchestration cycle when created
    last_used: int = 0       # Monotonic tick — updated on every access (LRU)


class SharedKnowledgeBase:
    """
    Structured context store for inter-agent knowledge transfer.

    Follows Dan Austin's Context Store pattern:
      - Orchestrator-Directed: orchestrator selects which entries to inject
      - Knowledge Artifacts: sub-agents create discrete, reusable entries
      - Selective Injection: sub-agents receive only relevant entries
      - Persistent Memory: entries persist across all agent invocations

    Information flow:
      1. Sub-agent discovers something → note(category, title, content)
      2. Entry stored with unique ID (kb_NNN)
      3. Orchestrator routing call sees compact summary (IDs + titles)
      4. Orchestrator decides: {"context_refs": ["kb_001", "kb_003"], ...}
      5. Selected entries injected into next sub-agent's context
    """

    def __init__(self, max_entries: int = MAX_ENTRIES):
        self._entries: dict[str, ContextEntry] = {}  # id → entry
        self._counter: int = 0
        self._max_entries = max_entries
        self._access_tick: int = 0  # Monotonic counter for LRU tracking

    def add(
        self,
        category: str,
        title: str,
        content: str,
        author: str = "unknown",
        cycle: int = 0,
    ) -> str:
        """Add a structured entry to the KB.

        Returns confirmation message with the assigned ID.
        """
        # Validate category
        if category not in CATEGORIES:
            return (
                f"[ERROR] Invalid category '{category}'. "
                f"Choose from: {', '.join(sorted(CATEGORIES))}"
            )

        clean_title = title.strip()[:120]
        clean_content = content.strip()[:1500]

        # Dedup: if an entry with the same title exists, update it in-place
        for eid, existing in self._entries.items():
            if existing.title == clean_title:
                existing.content = clean_content
                existing.author = author
                existing.cycle = cycle
                existing.category = category
                self._touch(existing)
                return (
                    f"[OK] Updated {eid} [{category}]: '{clean_title}' "
                    f"(KB: {len(self._entries)}/{self._max_entries} entries)"
                )

        # Evict if at capacity
        self._evict_if_needed()

        # Assign unique ID
        self._counter += 1
        entry_id = f"kb_{self._counter:03d}"

        entry = ContextEntry(
            id=entry_id,
            category=category,
            title=clean_title,
            content=clean_content,
            author=author,
            cycle=cycle,
        )
        self._entries[entry_id] = entry
        self._touch(entry)

        return (
            f"[OK] Recorded as {entry_id} [{category}]: '{entry.title}' "
            f"(KB: {len(self._entries)}/{self._max_entries} entries)"
        )

    def _touch(self, *entries: ContextEntry) -> None:
        """Mark entries as recently used (LRU tracking)."""
        self._access_tick += 1
        for e in entries:
            e.last_used = self._access_tick

    def get_by_ids(self, ids: list[str]) -> list[ContextEntry]:
        """Retrieve entries by their IDs. Unknown IDs are silently skipped."""
        found = [self._entries[eid] for eid in ids if eid in self._entries]
        self._touch(*found)
        return found

    def get_by_category(self, category: str) -> list[ContextEntry]:
        """Retrieve all entries in a category."""
        return [e for e in self._entries.values() if e.category == category]

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[ContextEntry]:
        """All entries in insertion order."""
        return list(self._entries.values())

    # ------------------------------------------------------------------
    # Formatting for different consumers
    # ------------------------------------------------------------------

    def format_for_orchestrator(self) -> str:
        """Compact summary for the orchestrator routing decision.

        Shows IDs, categories, and titles — NOT full content.
        The orchestrator uses this to decide which entries to inject
        into the next sub-agent via context_refs.
        """
        if not self._entries:
            return (
                "## Knowledge Base\n"
                "(empty — sub-agents can record findings with note())"
            )

        parts = [
            "## Knowledge Base",
            f"({len(self._entries)} entries — "
            f"include relevant IDs in context_refs to inject into sub-agent)",
        ]

        # Group by category for readability
        by_cat: dict[str, list[ContextEntry]] = {}
        for entry in self._entries.values():
            by_cat.setdefault(entry.category, []).append(entry)

        for cat in sorted(by_cat):
            parts.append(f"\n**{cat}:**")
            for e in by_cat[cat]:
                # Compact: just ID + title + author
                parts.append(f"  {e.id}: {e.title} (by {e.author}, C{e.cycle})")

        return "\n".join(parts)

    def format_for_agent(self, context_refs: Optional[list[str]] = None) -> str:
        """Full content for selected entries, injected into sub-agent context.

        Args:
            context_refs: List of entry IDs to include. If None or empty,
                          returns a minimal message. If ["*"], includes all.
        """
        # Fallback: if no context_refs provided but KB has entries,
        # inject the 5 most recent entries (prevents information blackout
        # when orchestrator omits context_refs due to format issues)
        _FALLBACK_RECENT = 5

        if not context_refs:
            if not self._entries:
                return ""
            # Auto-select most recent entries as fallback
            recent = list(self._entries.values())[-_FALLBACK_RECENT:]
            context_refs = [e.id for e in recent]

        # Special wildcard: include everything (fallback for compatibility)
        if context_refs == ["*"]:
            selected = list(self._entries.values())
        else:
            selected = self.get_by_ids(context_refs)

        if not selected:
            return (
                "## Knowledge Base\n"
                "(no matching entries for the provided context_refs)"
            )

        # Token budget: truncate if too large
        parts = ["## Knowledge Base (selected entries)"]
        total_chars = 0
        max_chars = int(MAX_AGENT_KB_TOKENS * _CHARS_PER_TOKEN)

        for entry in selected:
            entry_text = (
                f"\n**[{entry.id}] {entry.category}: {entry.title}** "
                f"(by {entry.author}, cycle {entry.cycle})\n"
                f"{entry.content}"
            )
            if total_chars + len(entry_text) > max_chars and total_chars > 0:
                remaining = len(selected) - len(parts) + 1
                parts.append(f"\n... ({remaining} more entries truncated for token budget)")
                break
            parts.append(entry_text)
            total_chars += len(entry_text)

        return "\n".join(parts)

    # Backward compatibility alias
    def format_for_context(self) -> str:
        """Legacy method — returns orchestrator view."""
        return self.format_for_orchestrator()

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Remove least-recently-used entry when at capacity.

        Strategy (LRU + age):
          Primary:   lowest last_used (least recently accessed)
          Secondary: lowest cycle (oldest creation time)
        """
        while len(self._entries) >= self._max_entries:
            victim_id = min(
                self._entries,
                key=lambda eid: (
                    self._entries[eid].last_used,
                    self._entries[eid].cycle,
                ),
            )
            del self._entries[victim_id]
