#!/usr/bin/env bash

set -euo pipefail

unset HF_HUB_ENABLE_HF_TRANSFER

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="/workspace/envs/turboquant"
RESULT_DIR="/workspace/results/turboquant"

source "$ENV_PATH/bin/activate"
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "TurboQuant Qwen3-8B COMPLETE PIPELINE"
echo "3-bit KV cache | BF16 | fused Triton"
echo "=========================================="

echo
echo "STEP 1/4: Compatibility smoke test"
python "$SCRIPT_DIR/smoke_test.py" \
    2>&1 | tee "$RESULT_DIR/smoke.log"

echo
echo "STEP 2/4: Quality benchmarks"
python "$SCRIPT_DIR/evaluate_quality.py" \
    2>&1 | tee "$RESULT_DIR/quality.log"

echo
echo "STEP 3/4: Performance benchmark"
python "$SCRIPT_DIR/measure_performance.py" \
    2>&1 | tee "$RESULT_DIR/performance.log"

echo
echo "STEP 4/4: Long-context KV-cache benchmark"
python "$SCRIPT_DIR/measure_long_context.py" \
    2>&1 | tee "$RESULT_DIR/long_context.log"

echo
echo "=========================================="
echo "TurboQuant full benchmark pipeline completed."
echo "Results: $RESULT_DIR"
echo "=========================================="
