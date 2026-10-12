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

        if config.agent.pipeline == "explorer+react":
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
                llm=llm_sup, terminal=terminal, config=config
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
            executor_config.agent.prompt_version = "v1"  # Always baseline v1

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
            self._write_logs(logging_dir, result, config, active_profile, explorer_result)

        # ----------------------------------------------------------
        # Build TerminalBench result
        # ----------------------------------------------------------
        failure_mode = None
        if not result.submitted:
            failure_mode = result.termination_reason or "did_not_submit"

        # Token accounting: include explorer overhead when applicable
        explorer_in = explorer_result.input_tokens if explorer_result else 0
        explorer_out = explorer_result.output_tokens if explorer_result else 0
        total_in = result.total_input_tokens + explorer_in
        total_out = result.total_output_tokens + explorer_out

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

            # Conversation trace — step-by-step tool calls for debugging
            if hasattr(result, "conversation_trace") and result.conversation_trace:
                (agent_logs_dir / "conversation_trace.json").write_text(
                    json.dumps(
                        result.conversation_trace,
                        indent=2, ensure_ascii=False,
                    ),
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
            summary_line = (
                f"submitted={result.submitted} | "
                f"steps={result.steps} | "
                f"elapsed={result.elapsed_seconds:.1f}s | "
                f"tokens={result.total_input_tokens}"
                f"+{result.total_output_tokens} | "
                f"{stats_str}{profile_str}{explorer_str}"
            )
            (agent_logs_dir / "summary.txt").write_text(
                summary_line, encoding="utf-8",
            )

        except Exception as e:
            logger.warning(f"Failed to write logs: {e}")
