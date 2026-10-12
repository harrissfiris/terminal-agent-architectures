#!/usr/bin/env bash
# G2 Planner: Explorer + Planner + per-subtask Executors.
#   - Executor prompt: v1 (baseline — plan context comes from subtask messages)
#   - Budget: CARVED, not additive. executor_steps = max_steps - explorer - planner
#     (tbench_agent_v2.py:210-214), so total steps are CONSERVED at 60 (e.g. 5 + 5 + 50).
# Budget config per model profile.

configure_budgets() {
  local profile="${THESIS_MODEL_PROFILE:-14b}"
  export THESIS_AGENT_PIPELINE=explorer+planner+react
  export THESIS_AGENT_PROMPT_VERSION=v1

  case "$profile" in
    14b)
      export THESIS_AGENT_EXPLORER_MAX_STEPS=10
      export THESIS_AGENT_EXPLORER_MAX_WALLCLOCK=150
      export THESIS_AGENT_PLANNER_MAX_STEPS=5
      export THESIS_AGENT_PLANNER_MAX_WALLCLOCK=90
      ;;
    32b)
      export THESIS_AGENT_EXPLORER_MAX_STEPS=5
      export THESIS_AGENT_EXPLORER_MAX_WALLCLOCK=90
      export THESIS_AGENT_PLANNER_MAX_STEPS=5
      export THESIS_AGENT_PLANNER_MAX_WALLCLOCK=90
      ;;
    gemini-flash|gemini)
      export THESIS_AGENT_EXPLORER_MAX_STEPS=5
      export THESIS_AGENT_EXPLORER_MAX_WALLCLOCK=90
      export THESIS_AGENT_PLANNER_MAX_STEPS=5
      export THESIS_AGENT_PLANNER_MAX_WALLCLOCK=90
      ;;
    7b|72b)
      # Use model_profiles.py defaults
      ;;
  esac
}
