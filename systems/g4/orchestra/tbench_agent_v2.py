"""
TerminalBench entry point — G4 Orchestra agent.

Strategist orchestrator with 3 persistent specialized sub-agents.
Optional shared knowledge base (THESIS_USE_KNOWLEDGE_BASE=1).

Usage:
    AGENT_IMPORT_PATH=tbench_agent_v2:OrchestraTerminalAgent tb run ...

Variants (same code, toggle via env var):
    THESIS_USE_KNOWLEDGE_BASE=0  ./scripts/run.sh 32b core_64  # vanilla
    THESIS_USE_KNOWLEDGE_BASE=1  ./scripts/run.sh 32b core_64  # + KB
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
from agent.orchestra import OrchestraAgent
from agent.knowledge_base import SharedKnowledgeBase
from model_profiles import get_profile, apply_profile, apply_supervisor_profile


# Configure logging
log_level = os.environ.get("THESIS_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
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


class OrchestraTerminalAgent(BaseAgent):
    """
    G4 Orchestra agent for TerminalBench.

    Strategist orchestrator with 3 persistent sub-agents (explorer, coder, verifier).
    Optionally uses a shared knowledge base for inter-agent knowledge transfer.
    """

    @staticmethod
    def name() -> str:
        use_kb = os.environ.get("THESIS_USE_KNOWLEDGE_BASE", "0") == "1"
        return f"thesis-orchestra{'_kb' if use_kb else ''}"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Optional[Path] = None,
    ) -> AgentResult:
        # ----------------------------------------------------------
        # Setup
        # ----------------------------------------------------------
        config = load_config()

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

        # Knowledge base toggle
        use_kb = os.environ.get("THESIS_USE_KNOWLEDGE_BASE", "0") == "1"
        knowledge_base = SharedKnowledgeBase() if use_kb else None

        agent = OrchestraAgent(
            llm=llm,
            supervisor_llm=llm_sup,
            executor=executor,
            config=config,
            burst_steps=config.agent.burst_steps,
            knowledge_base=knowledge_base,
        )

        # ----------------------------------------------------------
        # Environment discovery
        # ----------------------------------------------------------
        terminal.initialize()
        initial_obs = terminal.exec("pwd && ls -la", timeout=10.0).output

        env_parts: list[str] = []

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

        py_ver = terminal.exec("python3 --version 2>&1", timeout=5.0)
        if py_ver.output.strip():
            env_parts.append(f"Python: {py_ver.output.strip()}")

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
            f"Orchestra | KB={'enabled' if use_kb else 'disabled'} | "
            f"Budget: {config.agent.max_steps} steps, "
            f"{config.agent.max_wallclock_seconds}s wall-clock | "
            f"Burst: {config.agent.burst_steps} steps"
        )
        logger.info(
            f"Tests exist: "
            f"{'yes' if '---TESTS_EXIST---' in tests_result.output else 'no'}"
        )

        # ----------------------------------------------------------
        # Run orchestrated agent
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
            self._write_logs(
                logging_dir, result, config, active_profile,
                use_kb=use_kb,
                knowledge_base=knowledge_base,
                agent=agent,
            )

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
    def _write_logs(
        logging_dir: Path,
        result,
        config,
        profile=None,
        use_kb: bool = False,
        knowledge_base=None,
        agent=None,
    ) -> None:
        """Write orchestra-specific logs for thesis auditing."""
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
                "orchestra": {
                    "variant": "kb" if use_kb else "vanilla",
                    "burst_steps": config.agent.burst_steps,
                    "cycles": result.tool_stats.get("cycles", 0),
                    "kb_entries": result.tool_stats.get("kb_entries", 0),
                },
                "config": {
                    "provider": config.llm.provider,
                    "model": config.llm.model,
                    "supervisor_model": (config.supervisor.model if config.supervisor else None),
                    "tool_calling": config.llm.tool_calling,
                    "temperature": config.llm.temperature,
                    "max_tokens": config.llm.max_tokens,
                    "max_steps": config.agent.max_steps,
                    "max_wallclock": config.agent.max_wallclock_seconds,
                    "max_context_tokens": config.agent.max_context_tokens,
                    "burst_steps": config.agent.burst_steps,
                    "use_knowledge_base": use_kb,
                },
            }

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

            # Conversation trace
            if hasattr(result, "conversation_trace") and result.conversation_trace:
                (agent_logs_dir / "conversation_trace.json").write_text(
                    json.dumps(
                        result.conversation_trace,
                        indent=2, ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

                # Human-readable trace
                trace_lines = []
                for entry in result.conversation_trace:
                    tool = entry.get("tool", "?")
                    step = entry.get("step", "?")
                    elapsed = entry.get("elapsed_s", "?")
                    in_tok = entry.get("input_tokens", 0)
                    out_tok = entry.get("output_tokens", 0)
                    role = entry.get("burst_role", "?")
                    cycle = entry.get("cycle", "?")

                    args = entry.get("args", {})
                    if tool == "bash":
                        args_str = args.get("command", "")[:120]
                    elif tool in ("write_file", "patch_file", "read_file"):
                        args_str = args.get("path", "")
                    elif tool == "think":
                        args_str = (args.get("content", "") or "")[:100]
                    elif tool == "submit":
                        args_str = (args.get("answer", "") or "")[:80]
                    elif tool == "note":
                        args_str = args.get("title", "")
                    elif tool == "_orchestrator_route":
                        args_str = (
                            f"→ {args.get('next', '?')}: "
                            f"{args.get('instruction', '')[:100]}"
                        )
                    else:
                        args_str = str(args)[:100]

                    result_preview = (
                        entry.get("result_preview", "") or ""
                    )[:120]

                    line = (
                        f"[Step {step}] C{cycle} {role} | "
                        f"{elapsed}s | {tool}({args_str})\n"
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

            # Knowledge base snapshot (if enabled)
            if knowledge_base is not None and knowledge_base.size > 0:
                kb_data = [
                    {
                        "id": e.id,
                        "category": e.category,
                        "title": e.title,
                        "content": e.content,
                        "author": e.author,
                        "cycle": e.cycle,
                    }
                    for e in knowledge_base.entries
                ]
                (agent_logs_dir / "knowledge_base.json").write_text(
                    json.dumps(kb_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            # One-line summary for terminal
            stats_str = " | ".join(
                f"{k}={v}" for k, v in result.tool_stats.items()
            )
            profile_str = f" | profile={profile.name}" if profile else ""
            kb_str = " | KB" if use_kb else ""
            logger.info(
                f"RESULT: submitted={result.submitted} "
                f"| steps={result.steps} "
                f"| {result.elapsed_seconds:.1f}s "
                f"| {result.total_input_tokens}in/"
                f"{result.total_output_tokens}out "
                f"| {stats_str}{profile_str}{kb_str}"
            )

        except Exception as e:
            logger.error(f"Failed to write logs: {e}", exc_info=True)
