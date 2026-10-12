"""
TerminalBench entry point — G4 Dialogue agent.

Two equal agents (A and B) take round-robin turns on the same task.
Variant selection via THESIS_DIALOGUE_VARIANT:
  cooperation — Agent B continues Agent A's work (relay)
  debate      — Agent B critiques Agent A, then tries differently

Usage:
    AGENT_IMPORT_PATH=tbench_agent_v2:DialogueTerminalAgent tb run ...
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
from agent.dialogue import DialogueOrchestrator
from model_profiles import get_profile, apply_profile


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


class DialogueTerminalAgent(BaseAgent):
    """
    G4 Dialogue agent for TerminalBench.

    Two equal agents alternate turns on the same Docker session.
    Controlled by THESIS_DIALOGUE_VARIANT (cooperation | debate).
    """

    @staticmethod
    def name() -> str:
        variant = os.environ.get("THESIS_DIALOGUE_VARIANT", "cooperation")
        return f"thesis-dialogue-{variant}"

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

        llm = create_llm(config.llm)
        terminal = SafeTerminal(session)

        # ----------------------------------------------------------
        # Environment discovery (shared across all turns)
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

        variant = config.agent.dialogue_variant
        logger.info(f"Task: {instruction[:100]}...")
        logger.info(f"LLM: {config.llm.provider}/{config.llm.model}")
        logger.info(
            f"Dialogue variant: {variant} | "
            f"Budget: {config.agent.max_steps} steps, "
            f"{config.agent.max_wallclock_seconds}s wall-clock | "
            f"Steps/turn: {config.agent.dialogue_steps_per_turn}"
        )
        logger.info(
            f"Tests exist: "
            f"{'yes' if '---TESTS_EXIST---' in tests_result.output else 'no'}"
        )

        # ----------------------------------------------------------
        # Run dialogue orchestrator
        # ----------------------------------------------------------
        orchestrator = DialogueOrchestrator(
            llm=llm,
            terminal=terminal,
            config=config,
        )

        dialogue_result = orchestrator.run(
            instruction=instruction,
            initial_observation=initial_obs,
            env_info=env_info,
        )

        # ----------------------------------------------------------
        # Write logs
        # ----------------------------------------------------------
        if logging_dir is not None:
            self._write_logs(
                logging_dir, dialogue_result, config, active_profile,
            )

        # ----------------------------------------------------------
        # Build TerminalBench result
        # ----------------------------------------------------------
        final = dialogue_result.agent_result
        failure_mode = None
        if not final.submitted:
            failure_mode = final.termination_reason or "did_not_submit"

        return AgentResult(
            final_answer=final.answer or "",
            failure_mode=failure_mode,
            total_input_tokens=final.total_input_tokens,
            total_output_tokens=final.total_output_tokens,
            total_tokens=final.total_input_tokens + final.total_output_tokens,
        )

    @staticmethod
    def _write_logs(
        logging_dir: Path, dialogue_result, config, profile=None,
    ) -> None:
        """Write dialogue-specific logs for thesis auditing."""
        try:
            agent_logs_dir = logging_dir
            agent_logs_dir.mkdir(parents=True, exist_ok=True)

            final = dialogue_result.agent_result

            # ----------------------------------------------------------
            # Summary (dialogue-enriched)
            # ----------------------------------------------------------
            turn_summaries = []
            for t in dialogue_result.turns:
                turn_summaries.append({
                    "turn": t.turn_number,
                    "agent": t.agent_label,
                    "steps": t.agent_result.steps,
                    "elapsed_s": round(t.agent_result.elapsed_seconds, 2),
                    "submitted": t.agent_result.submitted,
                    "tokens_in": t.agent_result.total_input_tokens,
                    "tokens_out": t.agent_result.total_output_tokens,
                })

            summary = {
                "submitted": final.submitted,
                "answer": final.answer,
                "steps": dialogue_result.total_steps,
                "elapsed_seconds": round(dialogue_result.total_elapsed, 2),
                "termination_reason": final.termination_reason,
                "total_input_tokens": final.total_input_tokens,
                "total_output_tokens": final.total_output_tokens,
                "dialogue": {
                    "variant": dialogue_result.variant,
                    "total_turns": dialogue_result.total_turns,
                    "steps_per_turn": config.agent.dialogue_steps_per_turn,
                    "turns": turn_summaries,
                },
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

            # ----------------------------------------------------------
            # Per-turn conversation traces
            # ----------------------------------------------------------
            for t in dialogue_result.turns:
                tr = t.agent_result
                turn_tag = f"turn{t.turn_number:02d}_{t.agent_label.replace(' ', '_').lower()}"

                if hasattr(tr, "conversation_trace") and tr.conversation_trace:
                    (agent_logs_dir / f"{turn_tag}_trace.json").write_text(
                        json.dumps(
                            tr.conversation_trace,
                            indent=2, ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

                    # Human-readable per-turn trace
                    trace_lines = []
                    for entry in tr.conversation_trace:
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
                            f"[Step {step}] {elapsed}s | "
                            f"{tool}({args_str})\n"
                            f"  tokens: {in_tok}in/{out_tok}out\n"
                            f"  → {result_preview}\n"
                        )
                        if tool == "<parse_error>":
                            raw = entry.get("raw_output") or ""
                            line += f"  RAW OUTPUT ({len(raw)} chars):\n{raw}\n"
                        trace_lines.append(line)

                    (agent_logs_dir / f"{turn_tag}_trace.txt").write_text(
                        "\n".join(trace_lines), encoding="utf-8",
                    )

                # Per-turn action log (the compact handoff text)
                if t.action_log:
                    (agent_logs_dir / f"{turn_tag}_action_log.txt").write_text(
                        t.action_log, encoding="utf-8",
                    )

            # ----------------------------------------------------------
            # One-line summary
            # ----------------------------------------------------------
            profile_str = f" | profile={profile.name}" if profile else ""
            turn_info = " → ".join(
                f"T{t.turn_number}({t.agent_label}:{t.agent_result.steps}s)"
                for t in dialogue_result.turns
            )
            summary_line = (
                f"variant={dialogue_result.variant} | "
                f"submitted={final.submitted} | "
                f"steps={dialogue_result.total_steps} | "
                f"turns={dialogue_result.total_turns} | "
                f"elapsed={dialogue_result.total_elapsed:.1f}s | "
                f"tokens={final.total_input_tokens}"
                f"+{final.total_output_tokens} | "
                f"flow=[{turn_info}]{profile_str}"
            )
            (agent_logs_dir / "summary.txt").write_text(
                summary_line, encoding="utf-8",
            )

        except Exception as e:
            logger.warning(f"Failed to write logs: {e}")
