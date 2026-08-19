#!/usr/bin/env bash

set -euo pipefail

ENV_PATH="/workspace/envs/anybcq"
OFFICIAL_REPO="/workspace/tools/anybcq-official"

MODEL_PATH="/workspace/models/anybcq-qwen3-8b"
RESULT_DIR="/workspace/results/anybcq"

GPU="${GPU:-0}"

echo "=========================================="
echo "AnyBCQ Official Quality Evaluation"
echo "Qwen3-8B | 2-bit / 3-bit / 4-bit"
echo "=========================================="

# Activate the isolated AnyBCQ environment.
source "$ENV_PATH/bin/activate"

# Make sure the required files exist.
if [ ! -d "$OFFICIAL_REPO" ]; then
    echo "ERROR: Official AnyBCQ repository not found."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: AnyBCQ Qwen3-8B checkpoint not found."
    echo "Run the PTQ step first."
    exit 1
fi

mkdir -p "$RESULT_DIR"

cd "$OFFICIAL_REPO"

# Official AnyBCQ evaluation.
# The AnyBCQ evaluator automatically evaluates
# every precision stored in the multi-precision model.
CUDA_VISIBLE_DEVICES="$GPU" python run_eval.py \
  --model_path "$MODEL_PATH" \
  --output_file "$RESULT_DIR/official_quality_results.json" \
  2>&1 | tee "$RESULT_DIR/official_quality_eval.log"

echo
echo "=========================================="
echo "AnyBCQ official evaluation completed."
echo "Results: $RESULT_DIR"
echo "=========================================="
