#!/usr/bin/env bash

set -euo pipefail

ENV_PATH="/workspace/envs/anybcq"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="/workspace/results/anybcq"

source "$ENV_PATH/bin/activate"

mkdir -p "$RESULT_ROOT/4bit"
mkdir -p "$RESULT_ROOT/3bit"
mkdir -p "$RESULT_ROOT/2bit"

echo "=========================================="
echo "AnyBCQ Performance Benchmarks"
echo "Order: 4-bit -> 3-bit -> 2-bit"
echo "=========================================="

python "$SCRIPT_DIR/measure_performance.py" --bitwidth 4 \
  2>&1 | tee "$RESULT_ROOT/4bit/performance.log"

python "$SCRIPT_DIR/measure_performance.py" --bitwidth 3 \
  2>&1 | tee "$RESULT_ROOT/3bit/performance.log"

python "$SCRIPT_DIR/measure_performance.py" --bitwidth 2 \
  2>&1 | tee "$RESULT_ROOT/2bit/performance.log"

echo
echo "=========================================="
echo "All AnyBCQ performance runs completed."
echo "=========================================="
