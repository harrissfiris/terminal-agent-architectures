#!/usr/bin/env bash
# G4 Orchestra: Strategist orchestrator with 3 persistent sub-agents.
#
# Variants (set via THESIS_USE_KNOWLEDGE_BASE):
#   0 (default) — Vanilla: orchestrator is sole information bridge
#   1           — KB: + shared notepad with note() tool
#
# Usage:
#   THESIS_USE_KNOWLEDGE_BASE=0 ./scripts/run.sh 32b core_64  # vanilla
#   THESIS_USE_KNOWLEDGE_BASE=1 ./scripts/run.sh 32b core_64  # + KB

configure_budgets() {
  local profile="${THESIS_MODEL_PROFILE:-32b}"

  export THESIS_AGENT_PIPELINE=orchestra
  export THESIS_AGENT_PROMPT_VERSION=orchestrator
  export THESIS_AGENT_BURST_STEPS=6
  export THESIS_USE_KNOWLEDGE_BASE="${THESIS_USE_KNOWLEDGE_BASE:-0}"
  export AGENT_IMPORT_PATH="tbench_agent_v2:OrchestraTerminalAgent"

  # Standardised budget: same 60 steps / 1800s as single-agent for fair comparison
  # Routing steps are FREE — only specialist burst steps count toward budget
  # 60 ÷ 6 steps/burst = 10 full bursts (routing excluded from budget)
  export THESIS_AGENT_MAX_STEPS="${THESIS_AGENT_MAX_STEPS:-60}"
  export THESIS_AGENT_MAX_WALLCLOCK="${THESIS_AGENT_MAX_WALLCLOCK:-1800}"

  case "$profile" in
    7b|14b|32b|72b)
      # Context/token budgets come from model_profiles.py defaults
      ;;
  esac
}
