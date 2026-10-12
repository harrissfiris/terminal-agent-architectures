#!/usr/bin/env bash
# Phase I: single ReAct agent — no explorer, no planner.
# Budget config per model profile.

configure_budgets() {
  local profile="${THESIS_MODEL_PROFILE:-14b}"
  export THESIS_AGENT_PIPELINE=single
  export THESIS_AGENT_PROMPT_VERSION="${THESIS_AGENT_PROMPT_VERSION:-v1}"

  case "$profile" in
    7b|14b|32b|72b)
      # All budgets come from model_profiles.py defaults
      ;;
  esac
}
