"""
TerminalBench entry point — v2 ReAct agent.

Usage with tb CLI:
    AGENT_IMPORT_PATH=tbench_agent_v2:ReActTerminalAgent tb run ...

Or in run scripts:
    poetry run tb run --agent-import-path tbench_agent_v2:ReActTerminalAgent ...
"""
from __future__ import annotations

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
from model_profiles import get_profile, apply_profile


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

        llm = create_llm(config.llm)
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
        # Run agent
        # ----------------------------------------------------------
        result = agent.run(
            instruction=instruction,
            initial_observation=initial_obs,
            env_info=env_info,
        )

        # ----------------------------------------------------------
        # Write logs
        # ----------------------------------------------------------
        if logging_dir is not None:
            self._write_logs(logging_dir, result, config, active_profile)

        # ----------------------------------------------------------
        # Build TerminalBench result
        # ----------------------------------------------------------
        failure_mode = None
        if not result.submitted:
            failure_mode = result.termination_reason or "did_not_submit"

        return AgentResult(
            final_answer=result.answer or "",
            failure_mode=failure_mode,
            total_input_tokens=result.total_input_tokens,
            total_output_tokens=result.total_output_tokens,
            total_tokens=result.total_input_tokens + result.total_output_tokens,
        )

    @staticmethod
    def _write_logs(logging_dir: Path, result, config, profile=None) -> None:
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
                    "provider": config.llm.provider,
                    "model": config.llm.model,
                    "tool_calling": config.llm.tool_calling,
                    "temperature": config.llm.temperature,
                    "max_tokens": config.llm.max_tokens,
                    "max_steps": config.agent.max_steps,
                    "max_wallclock": config.agent.max_wallclock_seconds,
                    "max_context_tokens": config.agent.max_context_tokens,
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

            (agent_logs_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
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
                    # For parse failures, show the FULL raw model output — the
                    # primary material the parser choked on.
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
            summary_line = (
                f"submitted={result.submitted} | "
                f"steps={result.steps} | "
                f"elapsed={result.elapsed_seconds:.1f}s | "
                f"tokens={result.total_input_tokens}"
                f"+{result.total_output_tokens} | "
                f"{stats_str}{profile_str}"
            )
            (agent_logs_dir / "summary.txt").write_text(
                summary_line, encoding="utf-8",
            )

        except Exception as e:
            logger.warning(f"Failed to write logs: {e}")
