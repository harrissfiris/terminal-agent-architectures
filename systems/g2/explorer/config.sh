#!/usr/bin/env bash
# Stage 0.5: Explorer (read-only) + Executor.
# Budget config per model profile.

configure_budgets() {
  local profile="${THESIS_MODEL_PROFILE:-14b}"
  export THESIS_AGENT_PIPELINE=explorer+react
  export THESIS_AGENT_PROMPT_VERSION=v1

  case "$profile" in
    14b)
      export THESIS_AGENT_EXPLORER_MAX_STEPS=10
      export THESIS_AGENT_EXPLORER_MAX_WALLCLOCK=150
      ;;
    32b)
      export THESIS_AGENT_EXPLORER_MAX_STEPS=5
      export THESIS_AGENT_EXPLORER_MAX_WALLCLOCK=90
      ;;
    7b|72b)
      # Use model_profiles.py defaults
      ;;
  esac
}
