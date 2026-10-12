#!/usr/bin/env bash
# G2/Monitor v2: Executor + Monitor interleaved guidance.
# A monitor agent provides strategic coaching every N steps.
# v2 changes: interval=10, CAMEL inception injection, single-shot, escalation.

configure_budgets() {
  local profile="${THESIS_MODEL_PROFILE:-14b}"
  export THESIS_AGENT_PIPELINE=monitor
  export THESIS_AGENT_PROMPT_VERSION=v1
  export THESIS_AGENT_REVIEW_INTERVAL=10

  case "$profile" in
    7b|14b|32b|72b)
      # All budgets come from model_profiles.py defaults
      ;;
  esac
}
