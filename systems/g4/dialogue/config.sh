#!/usr/bin/env bash
# G4 Dialogue: two equal agents taking turns on the same task.
#
# Variants (set via THESIS_DIALOGUE_VARIANT):
#   cooperation — Agent B continues Agent A's work (relay/tag-team)
#   debate      — Agent B critiques Agent A's approach, then tries differently
#
# Usage:
#   THESIS_DIALOGUE_VARIANT=cooperation ./scripts/run.sh 32b core_64
#   THESIS_DIALOGUE_VARIANT=debate      ./scripts/run.sh 32b core_64

configure_budgets() {
  local profile="${THESIS_MODEL_PROFILE:-32b}"

  export THESIS_AGENT_PIPELINE=dialogue
  export THESIS_AGENT_PROMPT_VERSION=v1
  export THESIS_DIALOGUE_VARIANT="${THESIS_DIALOGUE_VARIANT:-cooperation}"
  export THESIS_DIALOGUE_STEPS_PER_TURN="${THESIS_DIALOGUE_STEPS_PER_TURN:-7}"
  export AGENT_IMPORT_PATH="tbench_agent_v2:DialogueTerminalAgent"

  # Standardised budget: same 60 steps / 1800s as single-agent for fair comparison
  # 60 ÷ 7 steps/turn ≈ 8 full turns (4 per agent)
  export THESIS_AGENT_MAX_STEPS="${THESIS_AGENT_MAX_STEPS:-60}"
  export THESIS_AGENT_MAX_WALLCLOCK="${THESIS_AGENT_MAX_WALLCLOCK:-1800}"

  case "$profile" in
    7b|14b|32b|72b)
      # Context/token budgets come from model_profiles.py defaults
      ;;
  esac
}
