"""
Analyst Agent — Stage 2.5 merged Explorer + Planner read-only phase.

Runs BEFORE the main Executor in the "explorer+selfval+react" pipeline.
Combines two responsibilities in a single phase:
  1. Data inspection (Explorer): read-only file inspection → Data Summary JSON
  2. Execution planning (Planner): data-grounded plan → execution_plan JSON

This replaces the separate Explorer (Stage 0.5) and Planner (Stage 1) with
a single agent, avoiding the budget overhead and context fragmentation of
two separate phases.  The plan is naturally data-grounded because the
Analyst produces it AFTER inspecting the actual files.

Design notes:
- ExplorerAgent is a thin wrapper around the existing ReActAgent.
  No new loop logic — just a restricted config + explorer_mode executor.
- Tool access is restricted to [bash, read_file, think], enforced at two
  levels: (1) only those schemas are passed to the LLM, so it cannot
  generate calls for unknown tools; (2) the executor rejects any bash
  command that is not in the readonly whitelist.
- Data Summary + execution_plan are extracted from the executor's
  think_history (full content, not the 200-char-truncated trace).
- Failure is non-fatal: if the Analyst produces malformed JSON or
  crashes, the Executor still runs — just without a Data Summary/plan.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from llm.base import BaseLLM
from terminal.session import SafeTerminal
from agent.executor import ToolExecutor
from agent.safety import CommandGuard
from agent.react import ReActAgent

logger = logging.getLogger(__name__)

# Tools the Explorer is allowed to use (schema-level + executor-level enforcement)
EXPLORER_TOOL_SUBSET: list[str] = ["bash", "read_file", "think"]


@dataclass
class ExplorerResult:
    """Outcome of an Analyst agent run (Explorer + Planner merged)."""

    # Core output
    data_summary: dict = field(default_factory=dict)
    data_summary_valid: bool = False   # True = parsed JSON with "files" key
    execution_plan: Optional[str] = None  # Formatted plan text for Executor injection

    # Derived metrics (for evaluation / logging)
    files_inspected: int = 0           # len(data_summary["files"]) if valid

    # Budget usage
    steps_used: int = 0
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    # Full think() history (untruncated) — kept for debugging
    think_history: list[str] = field(default_factory=list)

    # Step-by-step trace (same format as AgentResult.conversation_trace)
    conversation_trace: list[dict] = field(default_factory=list)

    # Termination reason from inner ReActAgent
    termination_reason: str = ""


def _extract_json_objects(text: str) -> list[dict]:
    """
    Scan text for all top-level JSON objects ({...}) and return any that
    parse successfully as dicts, in order of appearance.

    Uses a simple brace-depth scanner so adjacent or nested objects are
    handled correctly — unlike a greedy regex which merges them.
    """
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
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    return results


_READONLY_POISON_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"read[\-\s]?only",
        r"cannot\s+write",
        r"not\s+writable",
        r"no\s+write\s+access",
        r"inspection[\-\s]?only",
        r"environment\s+is\s+restricted",
    ]
]

_READONLY_REPLACEMENT = (
    "Note: The Explorer phase was read-only. "
    "The Executor has FULL write access to all files and directories."
)


def _sanitize_readonly_poison(data_summary: dict) -> dict:
    """
    Remove or rewrite any 'read-only' / 'cannot write' mentions from the
    Explorer's Data Summary before it reaches the Executor.

    The Explorer's tools are restricted, but the task environment is fully
    writable.  If the Explorer incorrectly claims the environment is read-only,
    the Executor may refuse to write files — causing task failures.
    """
    def _is_poisoned(text: str) -> bool:
        return any(p.search(text) for p in _READONLY_POISON_PATTERNS)

    changed = False

    # Sanitize top-level "observations" list
    observations = data_summary.get("observations")
    if isinstance(observations, list):
        cleaned = []
        has_poison = False
        for obs in observations:
            if isinstance(obs, str) and _is_poisoned(obs):
                has_poison = True
                logger.debug(f"Explorer sanitize: removed poisoned observation: {obs[:80]!r}")
            else:
                cleaned.append(obs)
        if has_poison:
            cleaned.append(_READONLY_REPLACEMENT)
            data_summary["observations"] = cleaned
            changed = True

    # Sanitize per-file "notes" fields
    files = data_summary.get("files")
    if isinstance(files, list):
        for f in files:
            notes = f.get("notes", "")
            if isinstance(notes, str) and _is_poisoned(notes):
                f["notes"] = _READONLY_REPLACEMENT
                changed = True
                logger.debug(f"Explorer sanitize: rewrote poisoned notes for {f.get('path', '?')}")

    if changed:
        logger.info("Explorer: sanitized read-only poison from Data Summary")

    return data_summary


def _extract_execution_plan(data_summary: dict) -> Optional[str]:
    """
    Extract the execution_plan from the Analyst's Data Summary JSON and
    format it as readable text for injection into the Executor's user message.

    Returns None if no valid execution_plan is present.
    """
    plan = data_summary.get("execution_plan")
    if not isinstance(plan, dict):
        return None

    approach = plan.get("approach", "")
    challenges = plan.get("key_challenges", [])
    criteria = plan.get("success_criteria", [])
    # Backwards compat: old format had "steps"
    steps = plan.get("steps", [])

    if not approach and not challenges and not steps:
        return None

    parts: list[str] = []

    if approach:
        parts.append(f"**Recommended approach:** {approach}")
        parts.append("")

    if challenges:
        parts.append("**Key challenges & warnings:**")
        for ch in challenges:
            parts.append(f"- {ch}")
        parts.append("")

    if criteria:
        parts.append("**Success criteria:**")
        for cr in criteria:
            parts.append(f"- {cr}")
        parts.append("")

    # Backwards compat: render old-format steps if present
    if steps:
        parts.append("**Suggested steps:**")
        for i, step in enumerate(steps, 1):
            parts.append(f"{i}. {step}")
        parts.append("")

    formatted = "\n".join(parts).strip()
    logger.info(f"Analyst: extracted strategic guidance ({len(challenges)} challenges, {len(criteria)} criteria)")
    return formatted if formatted else None


def extract_data_summary(think_history: list[str]) -> tuple[dict, bool]:
    """
    Extract the Data Summary JSON from the Explorer's think() call history.

    Searches backward from the last think() call (the Explorer is instructed
    to put its final summary in the last think), trying:
      1. Scan ALL top-level JSON objects in the content (handles the common
         pattern where the LLM writes a tool-wrapper object followed by the
         actual Data Summary object as separate adjacent blocks)
      2. Direct json.loads() on the full content as a single object
      3. Fallback: wrap raw text in {"raw_summary": ...}

    Returns:
        (data_summary_dict, is_valid)
        is_valid=True means we got a proper JSON dict with a "files" key.
    """
    if not think_history:
        logger.warning("Explorer: no think() calls found — returning empty summary")
        return {"error": "no_think_calls"}, False

    # Search from last to first
    for content in reversed(think_history):
        if not content or not content.strip():
            continue

        stripped = content.strip()

        # --- Attempt 1: scan ALL top-level JSON objects in the content ---
        # This handles the common LLM pattern of writing a tool-wrapper object
        # immediately followed by the actual Data Summary object:
        #   {"tool": "think", "args": {...}}\n\n{"task_directory": ..., "files": [...]}
        all_objects = _extract_json_objects(stripped)
        if all_objects:
            # Search from last to first for a dict with "files" key.
            # Also unwrap any double-wrapped think() calls.
            for obj in reversed(all_objects):
                # Unwrap {"tool":"think","args":{"content":"...JSON..."}}
                if obj.get("tool") == "think" and isinstance(obj.get("args"), dict):
                    inner = obj["args"].get("content", "")
                    if inner:
                        inner_objects = _extract_json_objects(inner)
                        for inner_obj in reversed(inner_objects):
                            if "files" in inner_obj:
                                logger.debug("Explorer: unwrapped double-wrapped think() call")
                                return inner_obj, True
                        # direct parse of inner string
                        try:
                            inner_parsed = json.loads(inner.strip())
                            if isinstance(inner_parsed, dict) and "files" in inner_parsed:
                                logger.debug("Explorer: unwrapped double-wrapped think() (direct)")
                                return inner_parsed, True
                        except json.JSONDecodeError:
                            pass
                    continue  # skip the tool-wrapper itself
                if "files" in obj:
                    logger.debug(
                        f"Explorer: found valid Data Summary in multi-object scan "
                        f"({len(all_objects)} objects found)"
                    )
                    return obj, True
            # No object had "files" — return the last non-wrapper object
            for obj in reversed(all_objects):
                if obj.get("tool") != "think":
                    logger.debug("Explorer: JSON parsed but missing 'files' key")
                    return obj, False

        # --- Attempt 2: direct JSON parse of full content ---
        try:
            result = json.loads(stripped)
            if isinstance(result, dict):
                valid = "files" in result
                if not valid:
                    logger.debug("Explorer: JSON parsed but missing 'files' key")
                return result, valid
        except json.JSONDecodeError:
            pass

        # This think() had content but no parseable JSON — try the previous one
        logger.debug(
            f"Explorer: think() content not parseable as JSON, "
            f"trying earlier think calls... (preview: {stripped[:80]!r})"
        )

    # --- Fallback: return raw last think content ---
    last = think_history[-1] if think_history else ""
    logger.warning(
        "Explorer: could not extract JSON from any think() call. "
        "Passing raw text as fallback summary."
    )
    return {"raw_summary": last}, False


def extract_data_summary_with_fallback(think_history, raw_outputs=None):
    """Additive robustness wrapper around extract_data_summary (primary path UNCHANGED).
    1) Try the intended think() channel (original behaviour, identical for Qwen).
    2) If that yields no VALID summary, re-run the SAME extraction over the raw model
       outputs — this recovers a valid Data Summary that a model (e.g. Devstral) emitted
       as a bare/fenced object which the main tool-call parser flagged as <parse_error>
       and therefore never recorded as a think() call.
    Returns the original (invalid) result when neither channel yields a summary."""
    summary, valid = extract_data_summary(think_history)
    if valid or not raw_outputs:
        return summary, valid
    r_summary, r_valid = extract_data_summary(raw_outputs)
    if r_valid:
        logger.debug("Explorer: recovered Data Summary from raw-output fallback")
        return r_summary, True
    return summary, valid


class ExplorerAgent:
    """
    Read-only inspection agent for Stage 0.5.

    Wraps the existing ReActAgent with:
    - A restricted tool subset [bash, read_file, think]
    - An executor in explorer_mode (bash whitelist enforced)
    - The explorer_v1 system prompt
    - A smaller step/wallclock budget

    The original config and executor are NOT modified — ExplorerAgent
    creates its own internal copies so it is safe to run before the
    Executor which operates on the full config.
    """

    def __init__(
        self,
        llm: BaseLLM,
        terminal: SafeTerminal,
        config: Config,
    ):
        self._llm = llm
        self._terminal = terminal
        self._config = config

    def run(
        self,
        instruction: str,
        initial_observation: str = "",
        env_info: str = "",
    ) -> ExplorerResult:
        """
        Run the Explorer phase for a task.

        Args:
            instruction:        The task description (same as for the Executor).
            initial_observation: Initial pwd/ls output (pre-collected).
            env_info:           Environment discovery string (tests, README, etc.)

        Returns:
            ExplorerResult with data_summary and metrics.
        """
        # ------------------------------------------------------------------
        # Build explorer-specific config (does not mutate original)
        # ------------------------------------------------------------------
        explorer_config = copy.deepcopy(self._config)
        explorer_config.agent.max_steps = self._config.agent.explorer_max_steps

        # Dynamic wallclock: honour explicit override (> 0), otherwise
        # use max(180s, 30% of total budget).  30% leaves 70% for the
        # Executor — enough for complex debug cycles on 600s total.
        configured_wc = self._config.agent.explorer_max_wallclock_seconds
        if configured_wc > 0:
            explorer_wallclock = configured_wc
        else:
            explorer_wallclock = max(180, int(self._config.agent.max_wallclock_seconds * 0.30))
        explorer_config.agent.max_wallclock_seconds = explorer_wallclock

        explorer_config.agent.prompt_version = "analyst_v1"

        logger.info(
            f"Explorer starting | "
            f"max_steps={explorer_config.agent.max_steps} | "
            f"max_wallclock={explorer_config.agent.max_wallclock_seconds}s"
        )

        # ------------------------------------------------------------------
        # Create dedicated executor in explorer_mode
        # (shares the same SafeTerminal — same tmux session, no env loss)
        # ------------------------------------------------------------------
        explorer_executor = ToolExecutor(
            terminal=self._terminal,
            default_timeout=explorer_config.agent.command_timeout,
            long_timeout=explorer_config.agent.long_command_timeout,
            command_guard=CommandGuard(),   # same safety layer as Executor
            explorer_mode=True,
        )

        # ------------------------------------------------------------------
        # Build inner ReActAgent and run with restricted tool subset
        # ------------------------------------------------------------------
        inner_agent = ReActAgent(
            llm=self._llm,
            executor=explorer_executor,
            config=explorer_config,
        )

        react_result = inner_agent.run(
            instruction=instruction,
            initial_observation=initial_observation,
            env_info=env_info,
            tool_subset=EXPLORER_TOOL_SUBSET,
        )

        # ------------------------------------------------------------------
        # Extract Data Summary + Execution Plan from full think() history
        # ------------------------------------------------------------------
        think_hist = explorer_executor.think_history
        _raw_outputs = [e.get("raw_output", "") for e in (react_result.conversation_trace or []) if e.get("raw_output")]
        data_summary, is_valid = extract_data_summary_with_fallback(think_hist, _raw_outputs)

        # Sanitize read-only poison before passing to Executor
        if is_valid:
            data_summary = _sanitize_readonly_poison(data_summary)

        # Extract and format the execution plan from the data summary JSON
        execution_plan = _extract_execution_plan(data_summary)

        files_inspected = 0
        if is_valid and isinstance(data_summary.get("files"), list):
            files_inspected = len(data_summary["files"])

        logger.info(
            f"Analyst finished | "
            f"steps={react_result.steps} | "
            f"elapsed={react_result.elapsed_seconds:.1f}s | "
            f"summary_valid={is_valid} | "
            f"files_inspected={files_inspected} | "
            f"has_plan={execution_plan is not None} | "
            f"tokens={react_result.total_input_tokens}in/"
            f"{react_result.total_output_tokens}out"
        )

        if not is_valid:
            logger.warning(
                "Analyst: Data Summary JSON could not be parsed. "
                "Executor will run without structured grounding data."
            )

        return ExplorerResult(
            data_summary=data_summary,
            data_summary_valid=is_valid,
            execution_plan=execution_plan,
            files_inspected=files_inspected,
            steps_used=react_result.steps,
            elapsed_seconds=react_result.elapsed_seconds,
            input_tokens=react_result.total_input_tokens,
            output_tokens=react_result.total_output_tokens,
            think_history=think_hist,
            conversation_trace=react_result.conversation_trace,
            termination_reason=react_result.termination_reason,
        )
