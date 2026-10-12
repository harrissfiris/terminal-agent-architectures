"""
Planner Agent — Stage 1 plan generation phase.

Runs AFTER the Explorer and BEFORE the per-subtask Executor loop in the
"explorer+planner+react" pipeline.  Its sole job is to decompose the task
into 1–4 sequential subtasks and output them as a JSON plan.

Design notes:
- PlannerAgent is a thin wrapper around the existing ReActAgent, analogous
  to ExplorerAgent.  No new loop logic — just a restricted config + think()-only
  tool access.
- The Planner has NO side-effect tools: only think() is available.  It cannot
  run commands, write files, or submit answers.
- The plan is extracted from the Planner's think() history using the same
  JSON-scanning pattern as ExplorerAgent.extract_data_summary().
- If the Planner produces invalid JSON, the pipeline falls back to
  single-subtask mode (= Stage 0.5 behavior), ensuring zero regression risk.
- Budget: 5 steps / 90s (32b profile).  Planner has room to analyze → draft → review.

Pipeline position:
    Explorer (read-only) → **Planner (think-only)** → Executor (per subtask)
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from config import Config
from llm.base import BaseLLM
from agent.executor import ToolExecutor
from agent.react import ReActAgent

logger = logging.getLogger(__name__)

# The Planner is restricted to think() only — no tools, no side effects.
PLANNER_TOOL_SUBSET: list[str] = ["think"]


@dataclass
class PlanResult:
    """Outcome of a Planner agent run."""

    # Core output
    plan: dict = field(default_factory=dict)
    plan_valid: bool = False           # True = parsed JSON with "subtasks" key
    subtasks: list[dict] = field(default_factory=list)

    # Budget usage
    steps_used: int = 0
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    # Full think() history (for debugging)
    think_history: list[str] = field(default_factory=list)

    # Step-by-step trace (same format as AgentResult.conversation_trace)
    conversation_trace: list[dict] = field(default_factory=list)

    # Termination reason from inner ReActAgent
    termination_reason: str = ""


def _extract_json_objects(text: str) -> list[dict]:
    """
    Scan text for all top-level JSON objects ({...}) and return those that
    parse successfully as dicts.  Uses a brace-depth scanner so adjacent
    objects are handled correctly.
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


def extract_plan(think_history: list[str]) -> tuple[dict, bool, list[dict]]:
    """
    Extract the Plan JSON from the Planner's think() call history.

    Searches backward from the last think() call, looking for a JSON object
    with a "subtasks" key containing a non-empty list.

    Returns:
        (plan_dict, is_valid, subtasks_list)
        is_valid=True means we got a proper JSON dict with a "subtasks" list
        containing at least one subtask.
    """
    if not think_history:
        logger.warning("Planner: no think() calls found — returning empty plan")
        return {"error": "no_think_calls"}, False, []

    for content in reversed(think_history):
        if not content or not content.strip():
            continue

        stripped = content.strip()

        # Scan all top-level JSON objects
        all_objects = _extract_json_objects(stripped)
        if all_objects:
            for obj in reversed(all_objects):
                # Unwrap double-wrapped think() calls
                if obj.get("tool") == "think" and isinstance(obj.get("args"), dict):
                    inner = obj["args"].get("content", "")
                    if inner:
                        inner_objects = _extract_json_objects(inner)
                        for inner_obj in reversed(inner_objects):
                            subtasks = inner_obj.get("subtasks", [])
                            if isinstance(subtasks, list) and len(subtasks) > 0:
                                logger.debug("Planner: unwrapped double-wrapped think()")
                                return inner_obj, True, subtasks
                    continue

                subtasks = obj.get("subtasks", [])
                if isinstance(subtasks, list) and len(subtasks) > 0:
                    logger.debug(
                        f"Planner: found valid plan with {len(subtasks)} subtasks"
                    )
                    return obj, True, subtasks

        # Direct JSON parse
        try:
            result = json.loads(stripped)
            if isinstance(result, dict):
                subtasks = result.get("subtasks", [])
                valid = isinstance(subtasks, list) and len(subtasks) > 0
                if not valid:
                    logger.debug("Planner: JSON parsed but missing/empty 'subtasks'")
                return result, valid, subtasks if valid else []
        except json.JSONDecodeError:
            pass

        logger.debug(
            f"Planner: think() content not parseable as plan JSON, "
            f"trying earlier think calls... (preview: {stripped[:80]!r})"
        )

    # Fallback
    last = think_history[-1] if think_history else ""
    logger.warning(
        "Planner: could not extract plan JSON from any think() call. "
        "Falling back to single-subtask mode."
    )
    return {"raw_content": last}, False, []


def extract_plan_with_fallback(think_history, raw_outputs=None):
    """Additive robustness wrapper around extract_plan (primary path UNCHANGED).
    Recovers a valid plan that a model emitted as a bare/fenced object (parse_error'd,
    never recorded as a think() call) by re-running extract_plan over raw model outputs
    when the think() channel yields no valid plan. Identical behaviour for Qwen."""
    plan, valid, subtasks = extract_plan(think_history)
    if valid or not raw_outputs:
        return plan, valid, subtasks
    r_plan, r_valid, r_subtasks = extract_plan(raw_outputs)
    if r_valid:
        logger.debug("Planner: recovered plan from raw-output fallback")
        return r_plan, True, r_subtasks
    return plan, valid, subtasks


def build_fallback_subtasks(instruction: str) -> list[dict]:
    """
    Build a single-subtask fallback plan when the Planner fails to produce
    valid JSON.  This ensures the pipeline degrades gracefully to Stage 0.5
    behavior rather than crashing.
    """
    return [
        {
            "id": 1,
            "description": (
                f"Complete the task: {instruction[:200]}. "
                "Use write_file() for any code longer than one expression. "
                "Run tests before submitting."
            ),
            "success_criteria": "All tests pass",
            "budget_hint": "large",
        }
    ]


def allocate_budgets(
    subtasks: list[dict],
    remaining_wallclock: float,
    remaining_steps: int,
) -> list[dict]:
    """
    Allocate wall-clock and step budgets to each subtask based on budget_hint.

    Budget hints map to proportional shares:
        small  → 15%
        medium → 25%
        large  → 50%

    Shares are normalized so they sum to 100% of the remaining budget.
    Each subtask gets a per-hint minimum floor (soft target):
        small  → 8 steps / 60s
        medium → 12 steps / 90s
        large  → 20 steps / 120s

    If the sum of floors exceeds the remaining budget, floors are scaled
    proportionally to fit — they act as soft targets, not hard minimums.

    Special case: single-subtask plans get 100% of the remaining budget
    (no allocation overhead).
    Returns:
        List of dicts with 'wallclock' and 'steps' for each subtask.
    """
    # Single-subtask plan → give it everything
    if len(subtasks) == 1:
        return [{"wallclock": remaining_wallclock, "steps": remaining_steps}]

    HINT_WEIGHTS = {
        "small": 0.15,
        "medium": 0.25,
        "large": 0.50,
    }
    DEFAULT_WEIGHT = 0.25  # unknown hints → medium

    HINT_STEP_FLOORS = {
        "small": 8,
        "medium": 12,
        "large": 20,
    }
    HINT_WC_FLOORS = {
        "small": 60.0,
        "medium": 90.0,
        "large": 120.0,
    }

    hints = []
    weights = []
    for st in subtasks:
        hint = st.get("budget_hint", "medium").lower().strip()
        hints.append(hint)
        weights.append(HINT_WEIGHTS.get(hint, DEFAULT_WEIGHT))

    # Normalize
    total_weight = sum(weights) or 1.0
    normalized = [w / total_weight for w in weights]

    # Compute raw floors and proportional allocations
    raw_step_floors = [HINT_STEP_FLOORS.get(h, 12) for h in hints]
    raw_wc_floors = [HINT_WC_FLOORS.get(h, 90.0) for h in hints]

    # Scale floors down if they exceed the budget (soft targets)
    total_step_floor = sum(raw_step_floors)
    total_wc_floor = sum(raw_wc_floors)

    step_floor_scale = min(1.0, remaining_steps / total_step_floor) if total_step_floor > 0 else 1.0
    wc_floor_scale = min(1.0, remaining_wallclock / total_wc_floor) if total_wc_floor > 0 else 1.0

    budgets = []
    for i, frac in enumerate(normalized):
        step_floor = int(raw_step_floors[i] * step_floor_scale)
        wc_floor = raw_wc_floors[i] * wc_floor_scale

        steps = max(step_floor, int(remaining_steps * frac))
        wc = max(wc_floor, remaining_wallclock * frac)
        budgets.append({"wallclock": wc, "steps": steps})

    # Cap: if total allocation exceeds remaining budget, trim from the
    # largest subtasks first.  This preserves the floors for smaller
    # subtasks (which need every step) and takes from those that can
    # most afford to lose a step.
    total_steps = sum(b["steps"] for b in budgets)
    if total_steps > remaining_steps:
        excess = total_steps - remaining_steps
        # Sort indices by steps descending; trim from the biggest first
        order = sorted(range(len(budgets)), key=lambda i: budgets[i]["steps"], reverse=True)
        for idx in order:
            if excess <= 0:
                break
            can_trim = max(0, budgets[idx]["steps"] - 3)  # never go below 3
            trim = min(excess, can_trim)
            budgets[idx]["steps"] -= trim
            excess -= trim

    total_wc = sum(b["wallclock"] for b in budgets)
    if total_wc > remaining_wallclock:
        excess_wc = total_wc - remaining_wallclock
        order = sorted(range(len(budgets)), key=lambda i: budgets[i]["wallclock"], reverse=True)
        for idx in order:
            if excess_wc <= 0:
                break
            can_trim = max(0.0, budgets[idx]["wallclock"] - 30.0)
            trim = min(excess_wc, can_trim)
            budgets[idx]["wallclock"] -= trim
            excess_wc -= trim

    return budgets


def compress_result(
    subtask_id: int,
    result,
    key_findings: str = "",
) -> str:
    """
    Compress an executor subtask result into a concise string for injection
    into the next subtask's context.

    Args:
        subtask_id: The subtask number (1-based).
        result: AgentResult from the subtask executor run.
        key_findings: Optional key findings from the subtask execution
                      (extracted from think history or final output).

    Returns:
        A compressed ~100-token summary string.
    """
    status = "completed" if result.submitted else "budget_exhausted"

    # Extract files written from conversation trace
    files_written = set()
    for entry in result.conversation_trace:
        if entry.get("tool") in ("write_file", "patch_file"):
            path = entry.get("args", {}).get("path", "")
            if path:
                files_written.add(path)

    # Extract last exit code — prefer the dedicated field (reliable),
    # fall back to parsing result_preview (legacy traces).
    last_exit = None
    for entry in reversed(result.conversation_trace):
        if entry.get("tool") == "bash":
            if "exit_code" in entry:
                last_exit = entry["exit_code"]
            else:
                # Legacy fallback: parse from truncated preview
                preview = entry.get("result_preview", "")
                if "Exit code:" in preview:
                    try:
                        last_exit = int(preview.split("Exit code:")[1].strip().rstrip("]"))
                    except (ValueError, IndexError):
                        pass
                elif "[SUBMITTED]" not in preview:
                    last_exit = 0
            break

    parts = [
        f"Subtask {subtask_id}: {status}.",
        f"Steps: {result.steps}.",
        f"Files: {sorted(files_written) if files_written else 'none'}.",
    ]

    if last_exit is not None and last_exit != 0:
        parts.append(f"Last exit code: {last_exit}.")

    if result.answer and result.submitted:
        parts.append(f"Output: {result.answer[:150]}")

    if key_findings:
        parts.append(f"Findings: {key_findings[:150]}")

    return " ".join(parts)


class PlannerAgent:
    """
    Plan generation agent for Stage 1.

    Wraps the existing ReActAgent with:
    - A restricted tool subset [think] only
    - The planner_v1 system prompt
    - A small step/wallclock budget

    The Planner receives the task description + Data Summary and produces
    a structured JSON plan decomposing the task into sequential subtasks.
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: Config,
    ):
        self._llm = llm
        self._config = config

    def run(
        self,
        instruction: str,
        data_summary: Optional[dict] = None,
        env_info: str = "",
        executor_steps: int = 0,
    ) -> PlanResult:
        """
        Run the Planner phase for a task.

        Args:
            instruction:   The task description.
            data_summary:  Data Summary from Explorer (may be None).
            env_info:      Environment discovery (tests, python ver, etc.)

        Returns:
            PlanResult with plan JSON and metrics.
        """
        # ------------------------------------------------------------------
        # Build planner-specific config (does not mutate original)
        # ------------------------------------------------------------------
        planner_config = copy.deepcopy(self._config)
        planner_config.agent.max_steps = self._config.agent.planner_max_steps

        # Wallclock: honour explicit override (> 0), otherwise use default
        configured_wc = self._config.agent.planner_max_wallclock_seconds
        if configured_wc > 0:
            planner_wallclock = configured_wc
        else:
            planner_wallclock = 60  # default 60s
        planner_config.agent.max_wallclock_seconds = planner_wallclock

        planner_config.agent.prompt_version = "planner_v1"

        logger.info(
            f"Planner starting | "
            f"max_steps={planner_config.agent.max_steps} | "
            f"max_wallclock={planner_config.agent.max_wallclock_seconds}s"
        )

        # ------------------------------------------------------------------
        # Create dedicated executor (think-only, no terminal needed).
        # allowed_tools=["think"] is the hard guardrail: any bash/write/submit
        # call the model generates is blocked before reaching the terminal,
        # returning a [BLOCKED] error message instead of crashing.
        # ------------------------------------------------------------------
        planner_executor = ToolExecutor(
            terminal=None,  # type: ignore  — Planner never uses terminal tools
            default_timeout=planner_config.agent.command_timeout,
            long_timeout=planner_config.agent.long_command_timeout,
            allowed_tools=["think"],
        )

        # ------------------------------------------------------------------
        # Build inner ReActAgent and run with think()-only tools
        # ------------------------------------------------------------------
        from agent.prompts import build_planner_user_message, get_planner_system_prompt

        inner_agent = ReActAgent(
            llm=self._llm,
            executor=planner_executor,
            config=planner_config,
        )

        # Build planner-specific first message
        max_subtasks = self._config.agent.planner_max_subtasks
        task_message = build_planner_user_message(
            instruction=instruction,
            data_summary=data_summary,
            env_info=env_info,
            executor_steps=executor_steps,
            max_subtasks=max_subtasks,
        )

        # Dynamic system prompt: matches max_subtasks cap so the system
        # prompt and user message are consistent (no "2-4" vs "1-2" clash).
        planner_system_prompt = get_planner_system_prompt(max_subtasks)

        # Pass the planner-formatted message directly and set
        # data_summary=None so react.py uses it as-is (no executor builder).
        react_result = inner_agent.run(
            instruction=task_message,
            initial_observation="",
            env_info="",
            tool_subset=PLANNER_TOOL_SUBSET,
            data_summary=None,
            system_prompt_override=planner_system_prompt,
        )

        # ------------------------------------------------------------------
        # Extract Plan from think() history
        # ------------------------------------------------------------------
        think_hist = planner_executor.think_history
        _raw_outputs = [e.get("raw_output", "") for e in (react_result.conversation_trace or []) if e.get("raw_output")]
        plan_dict, is_valid, subtasks = extract_plan_with_fallback(think_hist, _raw_outputs)

        logger.info(
            f"Planner finished | "
            f"steps={react_result.steps} | "
            f"elapsed={react_result.elapsed_seconds:.1f}s | "
            f"plan_valid={is_valid} | "
            f"subtasks={len(subtasks)} | "
            f"tokens={react_result.total_input_tokens}in/"
            f"{react_result.total_output_tokens}out"
        )

        if not is_valid:
            logger.warning(
                "Planner: Plan JSON could not be parsed. "
                "Falling back to single-subtask mode."
            )
            subtasks = build_fallback_subtasks(instruction)

        return PlanResult(
            plan=plan_dict,
            plan_valid=is_valid,
            subtasks=subtasks,
            steps_used=react_result.steps,
            elapsed_seconds=react_result.elapsed_seconds,
            input_tokens=react_result.total_input_tokens,
            output_tokens=react_result.total_output_tokens,
            think_history=think_hist,
            conversation_trace=react_result.conversation_trace,
            termination_reason=react_result.termination_reason,
        )
