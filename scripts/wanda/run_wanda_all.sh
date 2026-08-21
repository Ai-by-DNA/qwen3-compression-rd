#!/usr/bin/env bash

set -euo pipefail

ENV_PATH="/workspace/envs/wanda-env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="/workspace/results/wanda"
MODEL_PATH="/workspace/models/Qwen3-8B-WANDA"

source "$ENV_PATH/bin/activate"

mkdir -p "$RESULT_DIR"

unset HF_HUB_ENABLE_HF_TRANSFER

echo "=========================================="
echo "WANDA Qwen3-8B FINAL BENCHMARK"
echo "=========================================="

# Check that the Wanda checkpoint exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Wanda model not found:"
    echo "$MODEL_PATH"
    exit 1
fi

echo
echo "STEP 1/2: Performance benchmark"
echo

python "$SCRIPT_DIR/measure_performance.py" \
  2>&1 | tee "$RESULT_DIR/performance.log"

echo
echo "STEP 2/2: Quality benchmarks"
echo

bash "$SCRIPT_DIR/run_quality.sh"

echo
echo "=========================================="
echo "WANDA FINAL BENCHMARK COMPLETED"
echo "Results: $RESULT_DIR"
echo "=========================================="