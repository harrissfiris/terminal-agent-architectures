#!/usr/bin/env bash
# Stage 2: submit-intercept experiments (variants a/b/c/d).
# Set THESIS_STAGE2_VARIANT to choose the experiment.
#
# Usage:
#   THESIS_STAGE2_VARIANT=b ./scripts/run.sh 32b solvable_37

configure_budgets() {
  local profile="${THESIS_MODEL_PROFILE:-32b}"

  export THESIS_AGENT_PIPELINE=stage2
  export THESIS_STAGE2_VARIANT="${THESIS_STAGE2_VARIANT:-a}"
  export AGENT_IMPORT_PATH="tbench_agent_v2:Stage2TerminalAgent"

  case "$profile" in
    7b|14b|32b|72b)
      # All budgets come from model_profiles.py defaults
      ;;
  esac
}
