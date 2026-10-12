#!/usr/bin/env bash
# Pre-build Docker images for a set of tasks so the benchmark doesn't build at runtime.
#
# Usage:
#   bash scripts/prebuild_tasks.sh <subset>           # subset = scripts/task_subsets/<subset>.json
#   bash scripts/prebuild_tasks.sh core_64
#   bash scripts/prebuild_tasks.sh task-a task-b ...  # explicit task ids
#
# (Generalises the old prebuild_solvable40.sh, which had a hardcoded list.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKDIR="${TASKDIR:-$HOME/.cache/terminal-bench/repo/original-tasks}"
SUBSETS_DIR="$SCRIPT_DIR/task_subsets"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/prebuild_tasks.sh <subset|task-id...>"
  echo "Available subsets:"
  for f in "$SUBSETS_DIR"/*.json; do echo "  $(basename "${f%.json}")"; done
  exit 1
fi

# Resolve task list: a single arg matching a subset json → read its tasks; else treat args as task ids.
TASKS=()
if [[ $# -eq 1 && -f "$SUBSETS_DIR/$1.json" ]]; then
  SUBSET="$1"
  echo "Reading tasks from subset: $SUBSET"
  while IFS= read -r t; do TASKS+=("$t"); done < <(
    python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['tasks']))" "$SUBSETS_DIR/$SUBSET.json"
  )
else
  TASKS=("$@")
fi
echo "Tasks to prebuild: ${#TASKS[@]}"

BASE_IMAGES=(
  "ghcr.io/laude-institute/t-bench/python-3-13:20250620"
  "ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624"
  "python:3.13-slim-bookworm"
  "python:3.11-slim"
  "python:3.8-slim-bookworm"
  "ubuntu:22.04"
  "localstack/localstack:4.3"
)

echo ""
echo "=== Step 1: Pulling base images ==="
for img in "${BASE_IMAGES[@]}"; do
  if docker image inspect "$img" &>/dev/null; then
    echo "  [cached] $img"
  else
    echo "  [pulling] $img ..."
    docker pull "$img" || echo "  [warn] pull failed for $img (task may pin a different tag)"
  fi
done
echo "Base images ready."

echo ""
echo "=== Step 2: Building task images ==="
BUILT=0; SKIPPED=0; FAILED=0; FAILED_TASKS=()

for task in "${TASKS[@]}"; do
  task_dir="$TASKDIR/$task"
  image_name="tb__${task}__client"

  if ! [[ -d "$task_dir" ]]; then
    echo "  [MISSING] $task — task dir not found, skipping"
    ((FAILED++)) || true; FAILED_TASKS+=("$task (missing dir)"); continue
  fi
  if docker image inspect "$image_name" &>/dev/null; then
    echo "  [cached]  $task"; ((SKIPPED++)) || true; continue
  fi

  # Priority: root Dockerfile > client/Dockerfile (multi-service compose tasks)
  build_df=""; build_ctx=""
  if [[ -f "$task_dir/Dockerfile" ]]; then
    build_df="$task_dir/Dockerfile"; build_ctx="$task_dir"
  elif [[ -f "$task_dir/client/Dockerfile" ]]; then
    build_df="$task_dir/client/Dockerfile"; build_ctx="$task_dir/client"
  fi
  if [[ -z "$build_df" ]]; then
    echo "  [skip]    $task — no Dockerfile found"; ((SKIPPED++)) || true; continue
  fi

  echo "  [build]   $task"
  docker build -t "$image_name" -f "$build_df" "$build_ctx/" 2>&1 \
    | grep -E "^Step|^Successfully|^ERROR|error" || true
  if docker image inspect "$image_name" &>/dev/null; then
    ((BUILT++)) || true
  else
    echo "  [FAILED]  $task"; ((FAILED++)) || true; FAILED_TASKS+=("$task")
  fi
done

echo ""
echo "=== Done ==="
echo "  Built:   $BUILT"
echo "  Cached:  $SKIPPED"
echo "  Failed:  $FAILED"
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
  echo "  Failed tasks: ${FAILED_TASKS[*]}"
fi
echo ""
df -h / | tail -1
echo ""
docker system df
