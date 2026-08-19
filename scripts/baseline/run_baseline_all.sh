#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="/workspace/results/baseline"

mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "Qwen3-8B Full Baseline Evaluation"
echo "=========================================="

echo
echo "Step 1/2 - Running quality benchmarks..."
bash "$SCRIPT_DIR/run_quality.sh"

echo
echo "Step 2/2 - Running performance KPIs..."
python "$SCRIPT_DIR/measure_performance.py" \
  2>&1 | tee "$RESULT_DIR/performance.log"

echo
echo "=========================================="
echo "Full baseline evaluation completed."
echo "Results: $RESULT_DIR"
echo "=========================================="
