#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="/workspace/models/Qwen3-8B"
RESULT_DIR="/workspace/results/baseline"

mkdir -p "$RESULT_DIR"

# Avoid the Hugging Face transfer issue observed on RunPod.
unset HF_HUB_ENABLE_HF_TRANSFER

echo "=========================================="
echo "1/3 - WikiText-2 full baseline"
echo "=========================================="

lm-eval run \
  --model hf \
  --model_args pretrained="$MODEL_PATH",dtype=bfloat16,max_length=2048 \
  --tasks wikitext \
  --device cuda:0 \
  --batch_size 1 \
  --seed 42 \
  2>&1 | tee "$RESULT_DIR/wikitext_full.log"

echo
echo "=========================================="
echo "2/3 - MMLU full baseline"
echo "=========================================="

lm-eval run \
  --model hf \
  --model_args pretrained="$MODEL_PATH",dtype=bfloat16,max_length=2048 \
  --tasks mmlu \
  --num_fewshot 5 \
  --device cuda:0 \
  --batch_size 1 \
  --seed 42 \
  --output_path "$RESULT_DIR/mmlu_full" \
  2>&1 | tee "$RESULT_DIR/mmlu_full.log"

echo
echo "=========================================="
echo "3/3 - GSM8K full baseline"
echo "=========================================="

lm-eval run \
  --model hf \
  --model_args pretrained="$MODEL_PATH",dtype=bfloat16,enable_thinking=False \
  --tasks gsm8k \
  --device cuda:0 \
  --batch_size 1 \
  --num_fewshot 5 \
  --seed 42 \
  --apply_chat_template \
  --output_path "$RESULT_DIR/gsm8k_full" \
  2>&1 | tee "$RESULT_DIR/gsm8k_full.log"

echo
echo "=========================================="
echo "Baseline quality benchmarks completed."
echo "Results: $RESULT_DIR"
echo "=========================================="
