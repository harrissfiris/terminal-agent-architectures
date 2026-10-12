#!/usr/bin/env bash
# ============================================================
# Centralized configuration for all run scripts.
#
# Usage (from any run script):
#   source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
#
# Budget parameters live in systems/<group>/<variant>/config.sh.
# This file resolves the active group/variant and sources the right one.
#
# The .env file does NOT reach the Docker container, so all
# THESIS_* variables must be exported here as shell env vars.
# ============================================================

# Resolve paths — works whether sourced or executed directly
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _CONFIG_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _CONFIG_SH_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
PROJECT_DIR="$(dirname "$_CONFIG_SH_DIR")"

# ── Connection: backend preset selection ─────────────────────────────────────
# BACKEND picks a connection preset (URL + API key + provider) in one shot:
#   ngrok  (default) — Colab vLLM exposed via an ngrok tunnel
#   runpod           — RunPod pod OpenAI-compatible proxy
# Edit the *_API_BASE values below whenever the tunnel/pod changes.
# An explicitly exported THESIS_LLM_API_BASE / THESIS_LLM_API_KEY always wins
# over the preset (e.g. set them in .env or inline for a one-off endpoint).
BACKEND="${BACKEND:-ngrok}"

NGROK_API_BASE="https://<your-ngrok-subdomain>.ngrok-free.dev/v1"
NGROK_API_KEY="sk-local"   # vLLM ignores the value; any non-empty string works
RUNPOD_API_BASE="https://<your-pod-id>-8000.proxy.runpod.net/v1"
RUNPOD_API_KEY="sk-<your-pod-id>"   # RunPod proxy uses the pod id as the key

case "$BACKEND" in
  ngrok)  _BE_API_BASE="$NGROK_API_BASE";  _BE_API_KEY="$NGROK_API_KEY"  ;;
  runpod) _BE_API_BASE="$RUNPOD_API_BASE"; _BE_API_KEY="$RUNPOD_API_KEY" ;;
  *) echo "⚠ Unknown BACKEND='$BACKEND' (expected: ngrok | runpod); defaulting to ngrok" >&2
     BACKEND="ngrok"; _BE_API_BASE="$NGROK_API_BASE"; _BE_API_KEY="$NGROK_API_KEY" ;;
esac

export THESIS_LLM_BACKEND="$BACKEND"
export THESIS_LLM_API_BASE="${THESIS_LLM_API_BASE:-$_BE_API_BASE}"
export THESIS_LLM_API_KEY="${THESIS_LLM_API_KEY:-$_BE_API_KEY}"
export THESIS_LLM_PROVIDER="${THESIS_LLM_PROVIDER:-vllm}"
# Per-call LLM read-timeout: NOT hardcoded here. Each model profile declares its
# own llm_timeout_seconds (model_profiles.py) so slower models (e.g. Devstral)
# get a fair ceiling instead of a one-size-fits-all 120s that unfairly killed
# them mid-generation. Export THESIS_LLM_TIMEOUT only to force an override.
[[ -n "${THESIS_LLM_TIMEOUT:-}" ]] && export THESIS_LLM_TIMEOUT
export THESIS_LOG_LEVEL="${THESIS_LOG_LEVEL:-INFO}"

# ── Resolve active group/variant from symlink ────────────────────────────────
_resolve_active_config() {
  if [[ -L "$PROJECT_DIR/agent" ]]; then
    readlink "$PROJECT_DIR/agent" | sed 's|systems/||; s|/agent||'
  else
    echo "g1/baseline"  # fallback
  fi
}

ACTIVE_STAGE="$(_resolve_active_config)"
STAGE_CONFIG="$PROJECT_DIR/systems/$ACTIVE_STAGE/config.sh"

# ── Source group/variant-specific budgets ────────────────────────────────────
if [[ -f "$STAGE_CONFIG" ]]; then
  source "$STAGE_CONFIG"
else
  # Fallback: no stage config → define a no-op configure_budgets
  configure_budgets() { :; }
fi

# ── Helper: print budget summary ────────────────────────────────────────────
print_budget_summary() {
  local profile="${THESIS_MODEL_PROFILE:-?}"
  echo "  Stage:     $ACTIVE_STAGE"
  echo "  Profile:   $profile"
  echo "  Explorer:  ${THESIS_AGENT_EXPLORER_MAX_STEPS:-auto} steps / ${THESIS_AGENT_EXPLORER_MAX_WALLCLOCK:-auto}s"
  case "${THESIS_AGENT_PIPELINE:-}" in
    explorer+planner+react)
      echo "  Planner:   ${THESIS_AGENT_PLANNER_MAX_STEPS:-auto} steps / ${THESIS_AGENT_PLANNER_MAX_WALLCLOCK:-auto}s"
      ;;
    explorer+selfval+react)
      echo "  Retry min: ${THESIS_AGENT_SELFVAL_RETRY_MIN_STEPS:-5} steps / ${THESIS_AGENT_SELFVAL_RETRY_MIN_WALLCLOCK:-60}s"
      ;;
  esac
  echo "  Pipeline:  ${THESIS_AGENT_PIPELINE:-<from profile>}"
  echo "  Provider:  ${THESIS_LLM_PROVIDER:-vllm}"
  echo "  API Base:  ${THESIS_LLM_API_BASE:-<not set>}"
}
