"""
TerminalBench entry point — v2 ReAct agent.

Usage with tb CLI:
    AGENT_IMPORT_PATH=tbench_agent_v2:ReActTerminalAgent tb run ...

Or in run scripts:
    poetry run tb run --agent-import-path tbench_agent_v2:ReActTerminalAgent ...
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Any

from terminal_bench.agents import BaseAgent
from terminal_bench.terminal.tmux_session import TmuxSession

from config import load_config
from llm import create_llm
from terminal.session import SafeTerminal
from agent.executor import ToolExecutor
from agent.react import ReActAgent
from agent.explorer import ExplorerAgent, ExplorerResult
from agent.planner import PlannerAgent, PlanResult, allocate_budgets, compress_result, build_fallback_subtasks
from model_profiles import get_profile, apply_profile, apply_supervisor_profile


# Configure logging
log_level = os.environ.get("THESIS_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress TerminalBench harness DEBUG spam (tmux keys, docker compose cmds, etc.)
# Everything is already saved to runs/ — these add no value in the terminal.
logging.getLogger("terminal_bench").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class AgentResult:
    """TerminalBench-compatible result object."""

    def __init__(
        self,
        final_answer: str,
        failure_mode: Optional[str] = None,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        total_tokens: int = 0,
        timestamped_markers: Optional[List[Any]] = None,
    ):
        self.final_answer = final_answer
        self.failure_mode = failure_mode
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens
        self.total_tokens = total_tokens
        self.timestamped_markers = timestamped_markers or []


class ReActTerminalAgent(BaseAgent):
    """
    Minimal ReAct agent for TerminalBench.

    No classifier, no strategy hints, no adaptive budgets.
    The LLM decides everything.  The harness only provides:
    tools, memory management, safety guardrails, and budget enforcement.
    """

    @staticmethod
    def name() -> str:
        return "thesis-react-agent-v2"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Optional[Path] = None,
    ) -> AgentResult:
        """
        Execute the task using the ReAct agent.

        Args:
            instruction: Natural language task description.
            session: tmux session connected to the Docker container.
            logging_dir: Optional directory for writing agent logs.

        Returns:
            AgentResult with final answer and token stats.
        """
        # ----------------------------------------------------------
        # Setup
        # ----------------------------------------------------------
        config = load_config()

        # Apply model profile if set (THESIS_MODEL_PROFILE env var)
        profile_name = os.environ.get("THESIS_MODEL_PROFILE")
        active_profile = None
        if profile_name:
            active_profile = get_profile(profile_name)
            config = apply_profile(config, active_profile)
            logger.info(
                f"Model profile: {active_profile.name} "
                f"({active_profile.description})"
            )

        config = apply_supervisor_profile(config)   # heterogeneous supervisor (no-op unless THESIS_SUPERVISOR_PROFILE set)
        llm = create_llm(config.llm)                # worker/executor LLM
        llm_sup = create_llm(config.supervisor) if config.supervisor else llm  # supervisor LLM (same object if homogeneous)
        terminal = SafeTerminal(session)

        executor = ToolExecutor(
            terminal=terminal,
            default_timeout=config.agent.command_timeout,
            long_timeout=config.agent.long_command_timeout,
        )
        agent = ReActAgent(llm=llm, executor=executor, config=config)

        # ----------------------------------------------------------
        # Get initial observation + environment discovery
        # ----------------------------------------------------------
        terminal.initialize()
        initial_obs = terminal.exec("pwd && ls -la", timeout=10.0).output

        # Front-load environment discovery
        env_parts: list[str] = []

        # Check for tests (critical: defines success criterion)
        tests_result = terminal.exec(
            "ls /tests/ 2>/dev/null && echo '---TESTS_EXIST---' "
            "|| echo '---NO_TESTS---'",
            timeout=5.0,
        )
        if "---TESTS_EXIST---" in tests_result.output:
            env_parts.append(
                f"Tests directory: /tests/ exists\n"
                f"{tests_result.output.split('---TESTS_EXIST---')[0].strip()}"
            )
        else:
            env_parts.append("Tests directory: /tests/ does NOT exist")

        # Python version
        py_ver = terminal.exec("python3 --version 2>&1", timeout=5.0)
        if py_ver.output.strip():
            env_parts.append(f"Python: {py_ver.output.strip()}")

        # Check for README or instructions
        readme_result = terminal.exec(
            "cat /app/README* /app/readme* /app/INSTRUCTIONS* "
            "2>/dev/null | head -40",
            timeout=5.0,
        )
        if readme_result.output.strip():
            env_parts.append(
                f"README/Instructions:\n{readme_result.output.strip()}"
            )

        env_info = "\n".join(env_parts)

        logger.info(f"Task: {instruction[:100]}...")
        logger.info(f"LLM: {config.llm.provider}/{config.llm.model}")
        logger.info(
            f"Budget: {config.agent.max_steps} steps, "
            f"{config.agent.max_wallclock_seconds}s wall-clock"
        )
        logger.info(
            f"Tests exist: "
            f"{'yes' if '---TESTS_EXIST---' in tests_result.output else 'no'}"
        )

        # ----------------------------------------------------------
        # Run agent (pipeline-aware)
        # ----------------------------------------------------------
        explorer_result: Optional[ExplorerResult] = None
        planner_result: Optional[PlanResult] = None

        if config.agent.pipeline == "explorer+planner+react":
            # -------------------------------------------------------
            # Stage 1: Explorer → Planner → per-subtask Executor
            # -------------------------------------------------------
            logger.info(
                f"Pipeline: explorer+planner+react | "
                f"explorer_steps={config.agent.explorer_max_steps} | "
                f"planner_steps={config.agent.planner_max_steps}"
            )

            import time as _time
            pipeline_start = _time.time()

            # Phase 1: Explorer (read-only)
            explorer_agent = ExplorerAgent(
                llm=llm, terminal=terminal, config=config
            )
            explorer_result = explorer_agent.run(
                instruction=instruction,
                initial_observation=initial_obs,
                env_info=env_info,
            )

            data_summary = (
                explorer_result.data_summary
                if explorer_result.data_summary_valid
                else None
            )

            # Phase 2: Planner (think-only)
            # Pre-compute executor budget so the planner can make informed decisions.
            executor_steps_estimate = max(
                3,
                config.agent.max_steps
                - config.agent.explorer_max_steps
                - config.agent.planner_max_steps,
            )
            planner_agent = PlannerAgent(llm=llm_sup, config=config)
            planner_result = planner_agent.run(
                instruction=instruction,
                data_summary=data_summary,
                env_info=env_info,
                executor_steps=executor_steps_estimate,
            )

            subtasks = planner_result.subtasks

            # Apply hard subtask cap (from model profile / env var).
            # When capping, merge overflow subtask descriptions into the last
            # subtask so no work is silently dropped (subtask C may reference
            # files that subtask B was supposed to create).
            max_sub = config.agent.planner_max_subtasks
            if max_sub > 0 and len(subtasks) > max_sub:
                logger.warning(
                    f"Planner produced {len(subtasks)} subtasks but cap is {max_sub} "
                    f"— merging overflow into last subtask."
                )
                keep = subtasks[: max_sub - 1]
                overflow = subtasks[max_sub - 1 :]  # all remaining subtasks
                # Merge: combine descriptions, use last subtask's success_criteria
                merged_desc = "\n\n".join(
                    f"Phase {st.get('id', i+max_sub)}: {st.get('description', '')}"
                    for i, st in enumerate(overflow)
                )
                merged_subtask = {
                    "id": max_sub,
                    "description": merged_desc,
                    "success_criteria": overflow[-1].get("success_criteria", ""),
                    "budget_hint": "large",  # merged work → always large
                }
                subtasks = keep + [merged_subtask]

            logger.info(
                f"Planner produced {len(planner_result.subtasks)} subtasks "
                f"(valid={planner_result.plan_valid})"
                + (f" → capped to {len(subtasks)}" if len(subtasks) != len(planner_result.subtasks) else "")
            )

            # Phase 3: Execute subtasks
            elapsed_so_far = _time.time() - pipeline_start
            remaining_wallclock = max(
                60, config.agent.max_wallclock_seconds - elapsed_so_far
            )
            remaining_steps = executor_steps_estimate

            if len(subtasks) == 1:
                # ---------------------------------------------------------
                # Single-subtask bypass: run as Stage 0.5 with plan guidance
                # No decomposition overhead — full budget, full context.
                # The subtask description is injected as an "Execution Plan"
                # section so the Executor gets plan-as-guidance without the
                # per-subtask machinery.
                # ---------------------------------------------------------
                logger.info(
                    "Planner produced 1 subtask — bypassing decomposition, "
                    "running as Stage 0.5 with plan guidance"
                )

                from agent.prompts import build_executor_user_message

                plan_description = subtasks[0].get("description", "")
                plan_criteria = subtasks[0].get("success_criteria", "")
                if plan_criteria:
                    plan_description += f"\n\n**Success criteria:** {plan_criteria}"

                executor_config = copy.deepcopy(config)
                executor_config.agent.max_steps = remaining_steps
                executor_config.agent.max_wallclock_seconds = int(remaining_wallclock)
                executor_config.agent.prompt_version = "v1"

                bypass_executor = ToolExecutor(
                    terminal=terminal,
                    default_timeout=config.agent.command_timeout,
                    long_timeout=config.agent.long_command_timeout,
                )
                bypass_agent = ReActAgent(
                    llm=llm,
                    executor=bypass_executor,
                    config=executor_config,
                )

                result = bypass_agent.run(
                    instruction=instruction,
                    initial_observation=initial_obs,
                    env_info=env_info,
                    data_summary=data_summary,
                    plan_guidance=plan_description,
                )

            else:
                # ---------------------------------------------------------
                # Multi-subtask execution: per-subtask Executors with fresh
                # context and allocated budgets.
                # ---------------------------------------------------------
                budgets = allocate_budgets(
                    subtasks, remaining_wallclock, remaining_steps
                )

                previous_results: List[str] = []
                subtask_traces: List[dict] = []
                result = None  # will hold the last subtask result

                from agent.prompts import build_subtask_message
                from agent.tools import get_tool_names

                for i, subtask in enumerate(subtasks):
                    is_last = (i == len(subtasks) - 1)

                    # Check global wall-clock budget before starting subtask
                    elapsed_so_far = _time.time() - pipeline_start
                    if elapsed_so_far > config.agent.max_wallclock_seconds - 10:
                        logger.warning(
                            f"Global wall-clock budget nearly exhausted "
                            f"({elapsed_so_far:.0f}s) — skipping subtask {i+1}"
                        )
                        break

                    # Subtask budget
                    subtask_budget = budgets[i]
                    subtask_wc = min(
                        subtask_budget["wallclock"],
                        config.agent.max_wallclock_seconds - elapsed_so_far,
                    )
                    subtask_steps = subtask_budget["steps"]

                    logger.info(
                        f"Subtask {i+1}/{len(subtasks)}: "
                        f"{subtask.get('description', '?')[:80]} | "
                        f"budget: {subtask_steps} steps / {subtask_wc:.0f}s"
                    )

                    # Build subtask config (fresh context)
                    subtask_config = copy.deepcopy(config)
                    subtask_config.agent.max_steps = subtask_steps
                    subtask_config.agent.max_wallclock_seconds = int(subtask_wc)
                    subtask_config.agent.prompt_version = "v1"

                    # Submit gating: only the last subtask gets submit()
                    all_tools = get_tool_names()
                    if is_last:
                        tool_subset = None  # all tools including submit
                    else:
                        tool_subset = [t for t in all_tools if t != "submit"]

                    # Build subtask-specific first message
                    task_message = build_subtask_message(
                        instruction=instruction,
                        subtask=subtask,
                        data_summary=data_summary,
                        previous_results=previous_results,
                        is_last_subtask=is_last,
                        step_budget=subtask_steps,
                        initial_observation=initial_obs if i == 0 else None,
                    )

                    # Fresh executor + agent for this subtask.
                    subtask_executor = ToolExecutor(
                        terminal=terminal,
                        default_timeout=config.agent.command_timeout,
                        long_timeout=config.agent.long_command_timeout,
                        allowed_tools=tool_subset,
                    )
                    subtask_agent = ReActAgent(
                        llm=llm,
                        executor=subtask_executor,
                        config=subtask_config,
                    )

                    # Run the subtask with fresh context
                    result = subtask_agent.run(
                        instruction=task_message,
                        initial_observation="",
                        env_info="",
                        tool_subset=tool_subset,
                    )

                    # Compress result for next subtask
                    compressed = compress_result(
                        subtask_id=subtask.get("id", i + 1),
                        result=result,
                    )
                    previous_results.append(compressed)

                    # Track subtask trace for logging
                    subtask_traces.append({
                        "subtask_id": subtask.get("id", i + 1),
                        "description": subtask.get("description", ""),
                        "success_criteria": subtask.get("success_criteria", ""),
                        "budget_hint": subtask.get("budget_hint", "medium"),
                        "steps": result.steps,
                        "elapsed_s": round(result.elapsed_seconds, 1),
                        "submitted": result.submitted,
                        "input_tokens": result.total_input_tokens,
                        "output_tokens": result.total_output_tokens,
                        "tool_stats": result.tool_stats,
                        "compressed_result": compressed,
                        "trace": result.conversation_trace,
                        "task_message": task_message,
                    })

                    logger.info(
                        f"Subtask {i+1} done | "
                        f"steps={result.steps} | "
                        f"elapsed={result.elapsed_seconds:.1f}s | "
                        f"submitted={result.submitted}"
                    )

                    if is_last and result.submitted:
                        break

                # If no subtask ran (all skipped due to wall-clock), run a
                # last-ditch executor with the correct config (v1 prompt,
                # remaining budget) to force-submit something.
                if result is None:
                    logger.warning("All subtasks skipped — running last-ditch executor")
                    fallback_config = copy.deepcopy(config)
                    fallback_config.agent.max_steps = 5
                    fallback_config.agent.max_wallclock_seconds = 30
                    fallback_config.agent.prompt_version = "v1"
                    fallback_executor = ToolExecutor(
                        terminal=terminal,
                        default_timeout=config.agent.command_timeout,
                        long_timeout=config.agent.long_command_timeout,
                    )
                    fallback_agent = ReActAgent(
                        llm=llm,
                        executor=fallback_executor,
                        config=fallback_config,
                    )
                    result = fallback_agent.run(
                        instruction=instruction,
                        initial_observation=initial_obs,
                        env_info=env_info,
                        data_summary=data_summary,
                    )

                # Merge token stats from all subtask runs
                total_subtask_in = sum(
                    st["input_tokens"] for st in subtask_traces
                )
                total_subtask_out = sum(
                    st["output_tokens"] for st in subtask_traces
                )
                result.total_input_tokens = total_subtask_in
                result.total_output_tokens = total_subtask_out

                # Merge conversation traces
                merged_trace = []
                for st in subtask_traces:
                    merged_trace.extend(st.get("trace", []))
                result.conversation_trace = merged_trace

                # Stash subtask_traces for logging
                result._subtask_traces = subtask_traces  # type: ignore[attr-defined]

        elif config.agent.pipeline == "explorer+react":
            # -------------------------------------------------------
            # Stage 0.5: Explorer (read-only) → Executor (full tools)
            # -------------------------------------------------------
            logger.info(
                f"Pipeline: explorer+react | "
                f"explorer_steps={config.agent.explorer_max_steps} | "
                f"executor_steps={max(1, config.agent.max_steps - config.agent.explorer_max_steps)}"
            )

            # Phase 1: Explorer
            explorer_agent = ExplorerAgent(
                llm=llm, terminal=terminal, config=config
            )
            explorer_result = explorer_agent.run(
                instruction=instruction,
                initial_observation=initial_obs,
                env_info=env_info,
            )

            # Phase 2: Executor — remaining fixed step + wallclock budget
            executor_max_steps = max(
                1,
                config.agent.max_steps - config.agent.explorer_max_steps,
            )
            executor_wallclock = max(
                60,  # floor so executor always gets at least 60s
                config.agent.max_wallclock_seconds
                - explorer_result.elapsed_seconds,
            )
            executor_config = copy.deepcopy(config)
            executor_config.agent.max_steps = executor_max_steps
            executor_config.agent.max_wallclock_seconds = int(executor_wallclock)

            # Rebuild executor+agent with executor-specific config.
            # Shares the same SafeTerminal — same tmux session, no env loss.
            executor = ToolExecutor(
                terminal=terminal,
                default_timeout=config.agent.command_timeout,
                long_timeout=config.agent.long_command_timeout,
            )
            agent = ReActAgent(llm=llm, executor=executor, config=executor_config)

            result = agent.run(
                instruction=instruction,
                initial_observation=initial_obs,
                env_info=env_info,
                data_summary=(
                    explorer_result.data_summary
                    if explorer_result.data_summary_valid
                    else None
                ),
            )

        else:
            # -------------------------------------------------------
            # Original single-agent pipeline (Stage 0 / baseline)
            # -------------------------------------------------------
            result = agent.run(
                instruction=instruction,
                initial_observation=initial_obs,
                env_info=env_info,
            )

        # ----------------------------------------------------------
        # Write logs
        # ----------------------------------------------------------
        if logging_dir is not None:
            self._write_logs(logging_dir, result, config, active_profile, explorer_result, planner_result)

        # ----------------------------------------------------------
        # Build TerminalBench result
        # ----------------------------------------------------------
        failure_mode = None
        if not result.submitted:
            failure_mode = result.termination_reason or "did_not_submit"

        # Token accounting: include explorer + planner overhead when applicable
        explorer_in = explorer_result.input_tokens if explorer_result else 0
        explorer_out = explorer_result.output_tokens if explorer_result else 0
        planner_in = planner_result.input_tokens if planner_result else 0
        planner_out = planner_result.output_tokens if planner_result else 0
        total_in = result.total_input_tokens + explorer_in + planner_in
        total_out = result.total_output_tokens + explorer_out + planner_out

        return AgentResult(
            final_answer=result.answer or "",
            failure_mode=failure_mode,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_tokens=total_in + total_out,
        )

    @staticmethod
    def _write_logs(
        logging_dir: Path,
        result,
        config,
        profile=None,
        explorer_result: Optional[ExplorerResult] = None,
        planner_result: Optional["PlanResult"] = None,
    ) -> None:
        """Write agent execution logs for thesis auditing."""
        try:
            agent_logs_dir = logging_dir
            agent_logs_dir.mkdir(parents=True, exist_ok=True)

            # Summary
            summary = {
                "submitted": result.submitted,
                "answer": result.answer,
                "steps": result.steps,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "termination_reason": result.termination_reason,
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
                "tool_stats": result.tool_stats,
                "messages_dropped": result.messages_dropped,
                "config": {
                    "pipeline": config.agent.pipeline,
                    "provider": config.llm.provider,
                    "model": config.llm.model,
                    "supervisor_model": (config.supervisor.model if config.supervisor else None),
                    "tool_calling": config.llm.tool_calling,
                    "temperature": config.llm.temperature,
                    "max_tokens": config.llm.max_tokens,
                    "max_steps": config.agent.max_steps,
                    "max_wallclock": config.agent.max_wallclock_seconds,
                    "max_context_tokens": config.agent.max_context_tokens,
                    "explorer_max_steps": config.agent.explorer_max_steps,
                },
            }

            # Add model profile info if used
            if profile is not None:
                summary["model_profile"] = {
                    "name": profile.name,
                    "description": profile.description,
                    "model_id": profile.model_id,
                    "quantization": profile.quantization,
                    "gpu": profile.gpu,
                    "max_model_len": profile.max_model_len,
                }

            # Add Explorer metrics (Stage 0.5+ only)
            if explorer_result is not None:
                summary["explorer"] = {
                    "data_summary_valid": explorer_result.data_summary_valid,
                    "files_inspected": explorer_result.files_inspected,
                    "steps_used": explorer_result.steps_used,
                    "elapsed_seconds": round(explorer_result.elapsed_seconds, 2),
                    "input_tokens": explorer_result.input_tokens,
                    "output_tokens": explorer_result.output_tokens,
                    "termination_reason": explorer_result.termination_reason,
                }

            # Add Planner metrics (Stage 1 only)
            if planner_result is not None:
                summary["planner"] = {
                    "plan_valid": planner_result.plan_valid,
                    "num_subtasks": len(planner_result.subtasks),
                    "subtasks": [
                        {
                            "id": st.get("id"),
                            "description": st.get("description", ""),
                            "success_criteria": st.get("success_criteria", ""),
                            "budget_hint": st.get("budget_hint", "medium"),
                        }
                        for st in planner_result.subtasks
                    ],
                    "steps_used": planner_result.steps_used,
                    "elapsed_seconds": round(planner_result.elapsed_seconds, 2),
                    "input_tokens": planner_result.input_tokens,
                    "output_tokens": planner_result.output_tokens,
                    "termination_reason": planner_result.termination_reason,
                }

            # Add per-subtask execution metrics (Stage 1 only)
            subtask_traces_data = getattr(result, '_subtask_traces', None)
            if subtask_traces_data:
                summary["subtask_execution"] = [
                    {
                        "subtask_id": st["subtask_id"],
                        "description": st.get("description", "")[:200],
                        "steps": st["steps"],
                        "elapsed_s": st["elapsed_s"],
                        "submitted": st["submitted"],
                        "input_tokens": st["input_tokens"],
                        "output_tokens": st["output_tokens"],
                        "tool_stats": st["tool_stats"],
                        "compressed_result": st["compressed_result"],
                    }
                    for st in subtask_traces_data
                ]

            (agent_logs_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Explorer trace (Stage 0.5+ only)
            if explorer_result is not None and explorer_result.conversation_trace:
                (agent_logs_dir / "explorer_trace.json").write_text(
                    json.dumps(
                        explorer_result.conversation_trace,
                        indent=2, ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                # Also write the raw Data Summary JSON for quick inspection
                (agent_logs_dir / "data_summary.json").write_text(
                    json.dumps(
                        explorer_result.data_summary,
                        indent=2, ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            # Planner trace & plan (Stage 1 only)
            if planner_result is not None and planner_result.conversation_trace:
                (agent_logs_dir / "planner_trace.json").write_text(
                    json.dumps(
                        planner_result.conversation_trace,
                        indent=2, ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            if planner_result is not None and planner_result.subtasks:
                (agent_logs_dir / "plan.json").write_text(
                    json.dumps(
                        planner_result.subtasks,
                        indent=2, ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            # Conversation trace — step-by-step tool calls for debugging
            if hasattr(result, "conversation_trace") and result.conversation_trace:
                (agent_logs_dir / "conversation_trace.json").write_text(
                    json.dumps(
                        result.conversation_trace,
                        indent=2, ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                # Per-subtask executor traces (Stage 1 only)
                subtask_traces_data = getattr(result, '_subtask_traces', None)
                if subtask_traces_data:
                    for st in subtask_traces_data:
                        st_id = st.get("subtask_id", "?")
                        st_trace_data = {
                            "subtask_id": st_id,
                            "description": st.get("description", ""),
                            "success_criteria": st.get("success_criteria", ""),
                            "budget_hint": st.get("budget_hint", "medium"),
                            "steps": st.get("steps", 0),
                            "elapsed_s": st.get("elapsed_s", 0),
                            "submitted": st.get("submitted", False),
                            "task_message": st.get("task_message", ""),
                            "trace": st.get("trace", []),
                        }
                        (agent_logs_dir / f"executor_trace_{st_id}.json").write_text(
                            json.dumps(st_trace_data, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )

                # Human-readable conversation trace
                trace_lines = []
                for entry in result.conversation_trace:
                    tool = entry.get("tool", "?")
                    step = entry.get("step", "?")
                    elapsed = entry.get("elapsed_s", "?")
                    in_tok = entry.get("input_tokens", 0)
                    out_tok = entry.get("output_tokens", 0)

                    args_str = ""
                    args = entry.get("args", {})
                    if tool == "bash":
                        args_str = args.get("command", "")[:120]
                    elif tool in ("write_file", "patch_file", "read_file"):
                        args_str = args.get("path", "")
                    elif tool == "think":
                        args_str = (args.get("content", "") or "")[:100]
                    elif tool == "submit":
                        args_str = (args.get("answer", "") or "")[:80]

                    result_preview = (
                        entry.get("result_preview", "") or ""
                    )[:120]

                    line = (
                        f"[Step {step}] {elapsed}s | {tool}({args_str})\n"
                        f"  tokens: {in_tok}in/{out_tok}out\n"
                        f"  → {result_preview}\n"
                    )
                    if tool == "<parse_error>":
                        raw = entry.get("raw_output") or ""
                        line += f"  RAW OUTPUT ({len(raw)} chars):\n{raw}\n"
                    trace_lines.append(line)

                (agent_logs_dir / "conversation_trace.txt").write_text(
                    "\n".join(trace_lines), encoding="utf-8",
                )

            # One-line summary
            stats_str = " | ".join(
                f"{k}={v}" for k, v in result.tool_stats.items()
            )
            profile_str = f" | profile={profile.name}" if profile else ""
            explorer_str = ""
            if explorer_result is not None:
                explorer_str = (
                    f" | explorer_steps={explorer_result.steps_used}"
                    f" | explorer_valid={explorer_result.data_summary_valid}"
                    f" | explorer_files={explorer_result.files_inspected}"
                )
            planner_str = ""
            if planner_result is not None:
                planner_str = (
                    f" | planner_steps={planner_result.steps_used}"
                    f" | planner_valid={planner_result.plan_valid}"
                    f" | subtasks={len(planner_result.subtasks)}"
                )
            summary_line = (
                f"submitted={result.submitted} | "
                f"steps={result.steps} | "
                f"elapsed={result.elapsed_seconds:.1f}s | "
                f"tokens={result.total_input_tokens}"
                f"+{result.total_output_tokens} | "
                f"{stats_str}{profile_str}{explorer_str}{planner_str}"
            )
            (agent_logs_dir / "summary.txt").write_text(
                summary_line, encoding="utf-8",
            )

        except Exception as e:
            logger.warning(f"Failed to write logs: {e}")
