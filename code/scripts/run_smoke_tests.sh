#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"

# PersonaLens env file containing API configuration for evaluation
ENV_FILE="${ENV_FILE:-$CODE_DIR/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ENV file not found: $ENV_FILE"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

MEM_FRAME="${MEM_FRAME:-memos-api-online}"
TOP_K="${TOP_K:-10}"
MAX_TASKS="${MAX_TASKS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/../data/evaluation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CASE_FILTER="${1:-${CASE:-all}}"

# Usage:
#   bash scripts/run_smoke_tests.sh                # Run all (1,4,5)
#   bash scripts/run_smoke_tests.sh 1              # Run SMOKE-1 only
#   bash scripts/run_smoke_tests.sh 4              # Run SMOKE-4 only
#   bash scripts/run_smoke_tests.sh 5              # Run SMOKE-5 only
#   CASE=SMOKE-5 bash scripts/run_smoke_tests.sh   # Equivalent form
if [[ "$CASE_FILTER" == "--help" || "$CASE_FILTER" == "-h" ]]; then
  echo "Usage: bash scripts/run_smoke_tests.sh [all|1|4|5|SMOKE-1|SMOKE-4|SMOKE-5]"
  exit 0
fi

should_run() {
  local key="$1"
  case "$CASE_FILTER" in
    all) return 0 ;;
    "$key") return 0 ;;
    "SMOKE-$key") return 0 ;;
    *) return 1 ;;
  esac
}

run_eval() {
  local label="$1"
  shift
  echo "========== $label =========="
  "$PYTHON_BIN" src/evaluation.py "$@"
}

COMMON_ARGS=(
  --mode baseline
  --mem_frame "$MEM_FRAME"
  --stage add search answer eval
  --output_dir "$OUTPUT_DIR"
  --top_k "$TOP_K"
  --max_tasks "$MAX_TASKS"
  --run_overall_eval true
  --smoke_test
)

echo ">>> Smoke test starts..."

# Smoke suite includes 1, 4, and 5 (dataset_type=long).
# Smoke version uses interactive=false.

# (1) clean + single-domain + interactive=false (smoke)
if should_run "1"; then
  run_eval "SMOKE-1 standard: clean single-domain non-interactive" \
    "${COMMON_ARGS[@]}" \
    --multi_domain false \
    --interactive false \
    --no_noise true \
    --style false \
    --incremental false
fi

# (4) style + multi-domain + interactive=false (smoke)
if should_run "4"; then
  run_eval "SMOKE-4 standard: style multi-domain non-interactive" \
    "${COMMON_ARGS[@]}" \
    --multi_domain true \
    --interactive false \
    --no_noise false \
    --style true \
    --incremental false
fi

# (5) incremental + dataset_type=long + interactive=false (smoke)
# Note: long is the WildChat-filled, style-aligned long-context variant under a single-domain setting.
if should_run "5"; then
  run_eval "SMOKE-5 incremental: long (WildChat long-context) non-interactive" \
    "${COMMON_ARGS[@]}" \
    --multi_domain false \
    --interactive false \
    --no_noise false \
    --style true \
    --incremental true \
    --dataset_type long
fi

echo ">>> Smoke test finished."
