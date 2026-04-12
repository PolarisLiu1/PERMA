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
OUTPUT_DIR="${OUTPUT_DIR:-$CODE_DIR/../data/evaluation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CASE_FILTER="${1:-${CASE:-all}}"

# Usage:
#   bash scripts/run_standard_tests.sh                         # Run all
#   bash scripts/run_standard_tests.sh 1                       # Run CASE-1 only
#   bash scripts/run_standard_tests.sh 5-long                  # Run CASE-5 long only
#   CASE=5-style-multi bash scripts/run_standard_tests.sh      # Equivalent form
if [[ "$CASE_FILTER" == "--help" || "$CASE_FILTER" == "-h" ]]; then
  echo "Usage: bash scripts/run_standard_tests.sh [all|1|2|3|4|5-clean-single|5-clean-multi|5-noisy-multi|5-style-multi|5-long|5-long-multi]"
  exit 0
fi

should_run() {
  local key="$1"
  [[ "$CASE_FILTER" == "all" || "$CASE_FILTER" == "$key" ]]
}

run_eval() {
  local label="$1"
  shift
  echo "========== $label =========="
  "$PYTHON_BIN" src/evaluation.py "$@"
}

COMMON_ARGS=(
  --mode memory
  --mem_frame "$MEM_FRAME"
  --stage add search answer eval
  --output_dir "$OUTPUT_DIR"
  --top_k "$TOP_K"
  --run_overall_eval true
)

echo ">>> Standard evaluation starts..."

# (1) clean + single-domain + interactive=true
if should_run "1"; then
  run_eval "CASE-1 standard: clean single-domain interactive" \
    "${COMMON_ARGS[@]}" \
    --multi_domain false \
    --interactive true \
    --no_noise true \
    --style false \
    --incremental false
fi

# (2) clean + multi-domain + interactive=true
if should_run "2"; then
  run_eval "CASE-2 standard: clean multi-domain interactive" \
    "${COMMON_ARGS[@]}" \
    --multi_domain true \
    --interactive true \
    --no_noise true \
    --style false \
    --incremental false
fi

# (3) noisy + multi-domain + interactive=true
if should_run "3"; then
  run_eval "CASE-3 standard: noisy multi-domain interactive" \
    "${COMMON_ARGS[@]}" \
    --multi_domain true \
    --interactive true \
    --no_noise false \
    --style false \
    --incremental false
fi

# (4) style + multi-domain + interactive=true
# Note: style data is enabled via --style true.
if should_run "4"; then
  run_eval "CASE-4 standard: style multi-domain interactive" \
    "${COMMON_ARGS[@]}" \
    --multi_domain true \
    --interactive true \
    --no_noise false \
    --style true \
    --incremental false
fi

echo ">>> Incremental evaluation starts..."

# (5) Incremental mode notes:
# - dataset_type=standard: standard incremental variants (clean/noisy/style).
# - dataset_type=long / long_multi: WildChat-filled long-context variants (single-/multi-domain).

# incremental + standard(clean/single-domain)
if should_run "5-clean-single"; then
  run_eval "CASE-5 incremental: standard clean single-domain" \
    "${COMMON_ARGS[@]}" \
    --multi_domain false \
    --interactive true \
    --no_noise true \
    --style false \
    --incremental true \
    --dataset_type standard
fi

# incremental + standard(clean/multi-domain)
if should_run "5-clean-multi"; then
  run_eval "CASE-5 incremental: standard clean multi-domain" \
    "${COMMON_ARGS[@]}" \
    --multi_domain true \
    --interactive true \
    --no_noise true \
    --style false \
    --incremental true \
    --dataset_type standard
fi

# incremental + long (WildChat long context)
if should_run "5-long"; then
  run_eval "CASE-5 incremental: long (WildChat long-context)" \
    "${COMMON_ARGS[@]}" \
    --multi_domain false \
    --interactive true \
    --no_noise false \
    --style true \
    --incremental true \
    --dataset_type long
fi

# incremental + long_multi (WildChat long context, multi-domain)
if should_run "5-long-multi"; then
  run_eval "CASE-5 incremental: long_multi (WildChat long-context multi-domain)" \
    "${COMMON_ARGS[@]}" \
    --multi_domain true \
    --interactive true \
    --no_noise false \
    --style true \
    --incremental true \
    --dataset_type long_multi
fi

echo ">>> All standard tests finished."
