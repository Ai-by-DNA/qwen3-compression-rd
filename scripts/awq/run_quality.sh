#!/usr/bin/env bash

set -euo pipefail

ENV_PATH="/workspace/envs/llm-compressor-env"
MODEL_PATH="/workspace/models/Qwen3-8B-AWQ"
RESULT_DIR="/workspace/results/awq"

source "$ENV_PATH/bin/activate"

mkdir -p "$RESULT_DIR"

unset HF_HUB_ENABLE_HF_TRANSFER

echo "=========================================="
echo "AWQ Qwen3-8B QUALITY BENCHMARK"
echo "=========================================="

echo
echo "1/3 - WikiText-2 full AWQ"
echo "=========================================="

lm-eval run \
  --model hf \
  --model_args pretrained="$MODEL_PATH",max_length=2048 \
  --tasks wikitext \
  --device cuda:0 \
  --batch_size 1 \
  --seed 42 \
  2>&1 | tee "$RESULT_DIR/wikitext_full.log"

echo
echo "=========================================="
echo "2/3 - MMLU full AWQ"
echo "=========================================="

lm-eval run \
  --model hf \
  --model_args pretrained="$MODEL_PATH",max_length=2048 \
  --tasks mmlu \
  --num_fewshot 5 \
  --device cuda:0 \
  --batch_size 1 \
  --seed 42 \
  --output_path "$RESULT_DIR/mmlu_full" \
  2>&1 | tee "$RESULT_DIR/mmlu_full.log"

echo
echo "=========================================="
echo "3/3 - GSM8K full AWQ"
echo "=========================================="

lm-eval run \
  --model hf \
  --model_args pretrained="$MODEL_PATH",enable_thinking=False \
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
echo "AWQ quality benchmarks completed."
echo "Results: $RESULT_DIR"
echo "=========================================="