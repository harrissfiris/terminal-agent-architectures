#!/usr/bin/env bash
# ============================================================
# Switch the active system (root symlinks) to a different
# experiment group / variant.
#
# Usage:
#   ./scripts/switch_stage.sh g1/baseline   # G1 vanilla baseline
#   ./scripts/switch_stage.sh g1/prompt     # G1 enhanced prompt
#   ./scripts/switch_stage.sh g1/infra      # G1 + infrastructure fixes
#   ./scripts/switch_stage.sh g2/explorer   # G2 explorer+executor
#   ./scripts/switch_stage.sh g2/planner    # G2 explorer+planner+executor
#   ./scripts/switch_stage.sh g2/analyst    # G2 analyst+executor
#   ./scripts/switch_stage.sh g3/verify     # G3 submit-intercept
#   ./scripts/switch_stage.sh               # show current
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LINKS=(agent llm terminal config.py model_profiles.py tbench_agent_v2.py)

# Show current if no argument
if [[ -z "${1:-}" ]]; then
  current=$(readlink agent 2>/dev/null || echo "NOT A SYMLINK")
  echo "Current: $current"
  exit 0
fi

TARGET="$1"
TARGET_DIR="systems/$TARGET"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "❌ '$TARGET' not found. Available:"
  for group in systems/g*/; do
    gname=$(basename "$group")
    for variant in "$group"*/; do
      [[ -d "$variant" ]] && echo "  $gname/$(basename "$variant")"
    done
  done
  exit 1
fi

# Verify target has required files
for link in "${LINKS[@]}"; do
  if [[ ! -e "$TARGET_DIR/$link" ]]; then
    echo "❌ Missing: $TARGET_DIR/$link"
    exit 1
  fi
done

# Remove old symlinks
for link in "${LINKS[@]}"; do
  rm -f "$link"
done

# Create new symlinks
for link in "${LINKS[@]}"; do
  ln -s "$TARGET_DIR/$link" "$link"
done

echo "✅ Switched to: $TARGET"
echo "   agent → $TARGET_DIR/agent"
echo "   config.py → $TARGET_DIR/config.py"
echo "   tbench_agent_v2.py → $TARGET_DIR/tbench_agent_v2.py"
