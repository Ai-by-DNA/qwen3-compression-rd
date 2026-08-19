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
echo "AnyBCQ GSM8K Evaluation"
echo "Order: 4-bit -> 3-bit -> 2-bit"
echo "=========================================="

python "$SCRIPT_DIR/evaluate_gsm8k.py" --bitwidth 4 \
  2>&1 | tee "$RESULT_ROOT/4bit/gsm8k.log"

python "$SCRIPT_DIR/evaluate_gsm8k.py" --bitwidth 3 \
  2>&1 | tee "$RESULT_ROOT/3bit/gsm8k.log"

python "$SCRIPT_DIR/evaluate_gsm8k.py" --bitwidth 2 \
  2>&1 | tee "$RESULT_ROOT/2bit/gsm8k.log"

echo
echo "=========================================="
echo "All AnyBCQ GSM8K runs completed."
echo "=========================================="
