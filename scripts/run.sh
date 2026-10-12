#!/usr/bin/env bash
# ============================================================
# Unified benchmark runner — replaces run_benchmark.sh,
# run_subset.sh, and run_single.sh.
#
# Task lists live in scripts/task_subsets/*.json as:
#   { "name": "...", "description": "...", "tasks": ["task1", ...] }
#
# Usage:
#   ./scripts/run.sh <profile> [subset] [range]
#   ./scripts/run.sh <profile> single:<task-id>
#   ./scripts/run.sh --list
#
# Examples:
#   ./scripts/run.sh 32b                       # all 64 tasks (core_64)
#   ./scripts/run.sh 32b core_64               # same, explicit
#   ./scripts/run.sh 32b core_64 1-20          # tasks 1–20 within subset
#   ./scripts/run.sh 32b solvable_35           # 35 solvable tasks
#   ./scripts/run.sh 32b subset_20_32b         # curated 20-task subset
#   ./scripts/run.sh 32b interesting_20        # variance study
#   ./scripts/run.sh 32b single:hello-world    # single task
#   ./scripts/run.sh --list                    # list available subsets
#
# Profiles: 7b | 14b | 32b | 32b-fp16 | 72b | qwen3-32b | qwen3.6-27b
#           devstral | phi4 | deepseek32b | llama8b | mistral-nemo | gemini
#
# Environment variables:
#   N_CONCURRENT        — parallel tasks (default: 1)
#   BACKEND             — connection preset: ngrok (default) | runpod
#                          Sets API base + key + provider in one shot.
#                          e.g. BACKEND=runpod ./scripts/run.sh llama70b-awq ...
#   THESIS_LLM_API_BASE — override the backend preset's URL (always wins)
#   THESIS_LLM_PROVIDER — vllm | ollama (default: vllm)
#   SPLIT_TASKS=1       — run each task as a separate tb run, then merge
#                          individual run folders via _merge_runs.py.
#                          Useful for isolation without Docker overhead.
#   DOCKER_CLEAN=1      — like SPLIT_TASKS=1 but also prunes Docker containers
#                          and dangling images between tasks.
#                          Prevents Docker disk exhaustion on long runs.
#   For both: individual runs are preserved — if interrupted, merge
#             manually: python scripts/_merge_runs.py runs/<d1> ... -o runs/<out>
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SUBSETS_DIR="$SCRIPT_DIR/task_subsets"

# ── --list: show available subsets and exit ──────────────────────────────────
if [[ "${1:-}" == "--list" ]]; then
  echo "Available task subsets (scripts/task_subsets/):"
  echo ""
  for f in "$SUBSETS_DIR"/*.json; do
    [[ -f "$f" ]] || continue
    basename="${f##*/}"
    name="${basename%.json}"
    desc=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f'{len(d[\"tasks\"]):>3} tasks — {d.get(\"description\",\"\")}' )" "$f" 2>/dev/null || echo "  ? tasks")
    printf "  %-25s %s\n" "$name" "$desc"
  done
  echo ""
  echo "Usage: ./scripts/run.sh <profile> <subset> [range]"
  echo "       ./scripts/run.sh <profile> single:<task-id>"
  exit 0
fi

# ── Parse arguments ──────────────────────────────────────────────────────────
PROFILE="${1:?Usage: $0 <profile> [subset] [range]  |  $0 --list}"
SUBSET_ARG="${2:-core_64}"
RANGE_ARG="${3:-}"

# ── Activate venv ────────────────────────────────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

DATASET_PATH="${DATASET_PATH:-$HOME/.cache/terminal-bench/repo/original-tasks}"
AGENT_IMPORT_PATH="${AGENT_IMPORT_PATH:-tbench_agent_v2:ReActTerminalAgent}"
N_CONCURRENT="${N_CONCURRENT:-1}"

export THESIS_MODEL_PROFILE="$PROFILE"

# ── Prevent litellm from fetching model cost map from GitHub at import time ──
# litellm calls get_model_cost_map() (HTTP GET to GitHub) on import unless this
# env var is set. Must be exported HERE (before Python starts) because litellm
# is imported before load_dotenv() is called inside Python.
export LITELLM_LOCAL_MODEL_COST_MAP=True

# ── Load centralized budgets from config.sh (group-aware) ────────────────────
source "$SCRIPT_DIR/config.sh"
configure_budgets

# Fallback defaults (only if group config didn't set them)
export THESIS_AGENT_PIPELINE="${THESIS_AGENT_PIPELINE:-single}"
export THESIS_AGENT_PROMPT_VERSION="${THESIS_AGENT_PROMPT_VERSION:-v1}"

# ── Resolve task list ────────────────────────────────────────────────────────
SUBSET_NAME=""

if [[ "$SUBSET_ARG" == single:* ]]; then
  # ── Single task mode ──
  TASK_ID="${SUBSET_ARG#single:}"
  SELECTED=("$TASK_ID")
  SUBSET_NAME="single:$TASK_ID"

else
  # ── JSON subset mode ──
  JSON_FILE="$SUBSETS_DIR/${SUBSET_ARG}.json"
  if [[ ! -f "$JSON_FILE" ]]; then
    echo "❌ Subset file not found: $JSON_FILE"
    echo ""
    echo "Available subsets:"
    for f in "$SUBSETS_DIR"/*.json; do
      [[ -f "$f" ]] && echo "  ${f##*/}" | sed 's/.json$//'
    done
    echo ""
    echo "Or use: $0 $PROFILE single:<task-id>"
    exit 1
  fi

  SUBSET_NAME="$SUBSET_ARG"

  # Read tasks from JSON (compatible with macOS bash 3.2)
  ALL_TASKS=()
  while IFS= read -r line; do
    ALL_TASKS+=("$line")
  done < <(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for t in d['tasks']:
    print(t)
" "$JSON_FILE")

  TOTAL=${#ALL_TASKS[@]}

  if [[ -n "$RANGE_ARG" ]]; then
    # ── Range within subset ──
    RANGE_START=$(echo "$RANGE_ARG" | cut -d'-' -f1)
    RANGE_END=$(echo "$RANGE_ARG" | cut -d'-' -f2)

    if ! [[ "$RANGE_START" =~ ^[0-9]+$ ]] || ! [[ "$RANGE_END" =~ ^[0-9]+$ ]]; then
      echo "❌ Invalid range '$RANGE_ARG'. Expected: START-END (e.g. 1-20)"
      exit 1
    fi
    if (( RANGE_START < 1 || RANGE_END > TOTAL || RANGE_START > RANGE_END )); then
      echo "❌ Range $RANGE_ARG out of bounds. Valid: 1-${TOTAL}"
      exit 1
    fi

    IDX_START=$(( RANGE_START - 1 ))
    SELECTED=("${ALL_TASKS[@]:${IDX_START}:$(( RANGE_END - IDX_START ))}")
    SUBSET_NAME="${SUBSET_ARG}[${RANGE_START}-${RANGE_END}]"
  else
    SELECTED=("${ALL_TASKS[@]}")
  fi
fi

# ── Header ───────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  THESIS BENCHMARK RUN                                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Group:        %-43s║\n" "${ACTIVE_STAGE:-unknown}"
printf "║  Profile:     %-43s║\n" "$PROFILE"
printf "║  Subset:      %-43s║\n" "$SUBSET_NAME"
printf "║  Tasks:       %-43s║\n" "${#SELECTED[@]} tasks"
printf "║  Pipeline:    %-43s║\n" "$THESIS_AGENT_PIPELINE"
printf "║  Prompt:      %-43s║\n" "$THESIS_AGENT_PROMPT_VERSION"
if [[ "$THESIS_AGENT_PIPELINE" == explorer+* ]]; then
  printf "║  Explorer:    %-43s║\n" "${THESIS_AGENT_EXPLORER_MAX_STEPS:-auto} steps / ${THESIS_AGENT_EXPLORER_MAX_WALLCLOCK:-auto}s"
  if [[ "$THESIS_AGENT_PIPELINE" == "explorer+planner+react" ]]; then
    printf "║  Planner:     %-43s║\n" "${THESIS_AGENT_PLANNER_MAX_STEPS:-auto} steps / ${THESIS_AGENT_PLANNER_MAX_WALLCLOCK:-auto}s"
  elif [[ "$THESIS_AGENT_PIPELINE" == "explorer+selfval+react" ]]; then
    printf "║  Retry min:   %-43s║\n" "${THESIS_AGENT_SELFVAL_RETRY_MIN_STEPS:-5} steps / ${THESIS_AGENT_SELFVAL_RETRY_MIN_WALLCLOCK:-60}s"
  fi
fi
# Show env var overrides (ablation knobs)
for _v in THESIS_LLM_TEMPERATURE THESIS_LLM_TOOL_CALLING THESIS_AGENT_MAX_STEPS THESIS_AGENT_PROMPT_VERSION THESIS_AGENT_TOOLS THESIS_AGENT_CONTEXT; do
  [[ -n "${!_v:-}" ]] && printf "║  %-14s%-42s║\n" "${_v#THESIS_}:" "${!_v}"
done
printf "║  Provider:    %-43s║\n" "${THESIS_LLM_PROVIDER:-vllm}"
printf "║  Backend:     %-43s║\n" "${THESIS_LLM_BACKEND:-ngrok}"
printf "║  API Base:    %-43s║\n" "${THESIS_LLM_API_BASE:-<from .env>}"
printf "║  Concurrent:  %-43s║\n" "$N_CONCURRENT"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Validate profile exists (Python check)
python3 -c "
from model_profiles import get_profile
p = get_profile('$PROFILE')
print(f'  Model:       {p.model_id}')
print(f'  GPU:         {p.gpu}')
print(f'  Quant:       {p.quantization or \"fp16\"}')
print(f'  Context:     {p.max_model_len}')
print(f'  Max Steps:   {p.max_steps}')
print(f'  Max Tokens:  {p.max_tokens}')
"
echo ""

# Skip countdown for single task
if [[ ${#SELECTED[@]} -gt 1 ]]; then
  echo "Starting in 3 seconds... (Ctrl-C to abort)"
  sleep 3
fi

# ── Run ──────────────────────────────────────────────────────────────────────
# --no-rebuild: skip docker compose build + buildx prune (use pre-built images).
# Set NO_REBUILD=1 when images are already pre-built via scripts/prebuild_solvable40.sh.
_NO_REBUILD_FLAG=()
if [[ "${NO_REBUILD:-1}" == "1" ]]; then
  _NO_REBUILD_FLAG=(--no-rebuild)
fi

# --no-cleanup: tb's default ("cleanup") removes ALL run docker images after the
# run — in per-task mode that destroys each task's image so the next task rebuilds
# from scratch (very slow under amd64 emulation on Apple Silicon). With pre-built
# images (scripts/prebuild_solvable40.sh) we keep them so tasks start instantly.
# Set NO_CLEANUP=0 to let tb reclaim images afterwards (useful when disk is tight).
_NO_CLEANUP_FLAG=()
if [[ "${NO_CLEANUP:-1}" == "1" ]]; then
  _NO_CLEANUP_FLAG=(--no-cleanup)
fi

# Determine per-task split mode
_SPLIT=0
if [[ "${SPLIT_TASKS:-0}" == "1" || "${DOCKER_CLEAN:-0}" == "1" || "${FULL_CLEAN:-0}" == "1" ]]; then
  _SPLIT=1
fi

if [[ "$_SPLIT" == "1" && ${#SELECTED[@]} -gt 1 ]]; then
  # ── Per-task mode: each task as a separate tb run ────────────────────────
  if [[ "${DOCKER_CLEAN:-0}" == "1" ]]; then
    echo "🐳 DOCKER_CLEAN mode: running tasks one-by-one with cleanup between each"
  else
    echo "🔀 SPLIT_TASKS mode: running each task as a separate tb run"
  fi
  echo "   Individual run folders will be merged at the end."
  echo "   If interrupted, merge manually with: python scripts/_merge_runs.py runs/<dir1> runs/<dir2> ... -o runs/<merged>"
  echo ""

  NEW_RUNS=()
  COMPLETED=0
  TOTAL=${#SELECTED[@]}
  FAILED_TASKS=()

  for task in "${SELECTED[@]}"; do
    COMPLETED=$((COMPLETED + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$COMPLETED/$TOTAL]  $task"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Snapshot existing run directories
    BEFORE_RUNS=$(ls -d runs/*/ 2>/dev/null | sort || true)

    # Run single task (allow failure without aborting the loop)
    if tb run \
      --dataset-path "$DATASET_PATH" \
      --agent-import-path "$AGENT_IMPORT_PATH" \
      --n-concurrent 1 \
      ${_NO_REBUILD_FLAG[@]+"${_NO_REBUILD_FLAG[@]}"} \
      ${_NO_CLEANUP_FLAG[@]+"${_NO_CLEANUP_FLAG[@]}"} \
      --task-id "$task"; then
      echo "  ✓ $task finished"
    else
      echo "  ✗ $task — tb run exited with error (result may still be saved)"
      FAILED_TASKS+=("$task")
    fi

    # Detect new run directory
    AFTER_RUNS=$(ls -d runs/*/ 2>/dev/null | sort || true)
    NEW_DIR=$(comm -13 <(echo "$BEFORE_RUNS") <(echo "$AFTER_RUNS") | head -1)
    if [[ -n "$NEW_DIR" ]]; then
      # Strip trailing slash for clean path
      NEW_DIR="${NEW_DIR%/}"
      NEW_RUNS+=("$NEW_DIR")
      echo "  → ${NEW_DIR}"
    else
      echo "  ⚠ No new run directory detected for $task"
    fi

    # Docker cleanup between tasks.
    #   DOCKER_CLEAN=1 → LIGHT prune (containers + dangling images), preserving the pip/apt build cache for
    #     speed (skips after the last task). Good when disk is roomy and images are light.
    #   FULL_CLEAN=1   → FULL prune after EVERY task (incl. the last): `image prune -af` + `builder prune -af`
    #     + volumes. Reclaims ALL space so Docker.raw never accumulates — REQUIRED for heavy tasks on a small
    #     disk (the boundary probe: kernel/cython/mailman images + GBs of build cache per task would otherwise
    #     fill the disk and hang the daemon). Costs a full rebuild each task (~+5min) but is disk-safe.
    if [[ "${FULL_CLEAN:-0}" == "1" ]]; then
      echo "  🐳 Docker FULL cleanup (image+builder+volumes)..."
      docker container prune -f  >/dev/null 2>&1 || true
      docker image   prune -af   >/dev/null 2>&1 || true
      docker builder prune -af   >/dev/null 2>&1 || true
      docker volume  prune -f    >/dev/null 2>&1 || true
    elif [[ "${DOCKER_CLEAN:-0}" == "1" && $COMPLETED -lt $TOTAL ]]; then
      echo "  🐳 Docker cleanup (light)..."
      docker container prune -f >/dev/null 2>&1 || true
      docker image prune -f >/dev/null 2>&1 || true
    fi
  done

  # ── Merge all individual runs ────────────────────────────────────────────
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  All $TOTAL tasks complete. ${#NEW_RUNS[@]} run directories created."
  if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo "  ⚠ ${#FAILED_TASKS[@]} task(s) had errors: ${FAILED_TASKS[*]}"
  fi
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [[ ${#NEW_RUNS[@]} -gt 1 ]]; then
    MERGED_NAME="merged-$(date +%Y-%m-%d__%H-%M-%S)"
    echo ""
    echo "Merging ${#NEW_RUNS[@]} runs → runs/$MERGED_NAME ..."
    python3 scripts/_merge_runs.py "${NEW_RUNS[@]}" -o "runs/$MERGED_NAME"
  elif [[ ${#NEW_RUNS[@]} -eq 1 ]]; then
    echo "  Only 1 run directory — no merge needed."
    MERGED_NAME="$(basename "${NEW_RUNS[0]}")"
  else
    echo "  ⚠ No run directories found to merge!"
  fi

else
  # ── Normal batch mode ────────────────────────────────────────────────────
  TASK_FLAGS=()
  for task in "${SELECTED[@]}"; do
    TASK_FLAGS+=(--task-id "$task")
  done

  tb run \
    --dataset-path "$DATASET_PATH" \
    --agent-import-path "$AGENT_IMPORT_PATH" \
    --n-concurrent "$N_CONCURRENT" \
    ${_NO_REBUILD_FLAG[@]+"${_NO_REBUILD_FLAG[@]}"} \
    ${_NO_CLEANUP_FLAG[@]+"${_NO_CLEANUP_FLAG[@]}"} \
    "${TASK_FLAGS[@]}"
fi

echo ""
echo "=== Run complete: group=${ACTIVE_STAGE:-?}  profile=$PROFILE  subset=$SUBSET_NAME  tasks=${#SELECTED[@]} ==="
