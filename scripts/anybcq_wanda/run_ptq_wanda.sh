#!/usr/bin/env bash

set -euo pipefail

# Reduce CUDA memory fragmentation on memory-constrained GPUs.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ENV_PATH="/workspace/envs/anybcq-wanda"
OFFICIAL_REPO="/workspace/tools/anybcq-wanda-official"

MODEL_PATH="/workspace/models/Qwen3-8B-WANDA"
SAVE_PATH="${SAVE_PATH:-/workspace/models/anybcq-wanda-qwen3-8b}"
CACHE_PATH="${CACHE_PATH:-/workspace/datasets/anybcq-cache}"

GPU="${GPU:-0}"

# Official full PTQ settings.
# They can be reduced later only for a compatibility/smoke test.
NUM_SAMPLES="${NUM_SAMPLES:-256}"
ITERS_W="${ITERS_W:-5000}"
RECON_DTYPE="${RECON_DTYPE:-bfloat16}"

echo "=========================================="
echo "AnyBCQ PTQ - Qwen3-8B"
echo "=========================================="

# ------------------------------------------------------------
# 1. Activate isolated AnyBCQ environment
# ------------------------------------------------------------

source "$ENV_PATH/bin/activate"

# ------------------------------------------------------------
# 2. Check required paths
# ------------------------------------------------------------

if [ ! -d "$OFFICIAL_REPO" ]; then
    echo "ERROR: Official AnyBCQ repository not found."
    echo "Run setup_anybcq.sh first."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Qwen3-8B model not found at $MODEL_PATH"
    exit 1
fi

mkdir -p "$SAVE_PATH"
mkdir -p "$CACHE_PATH"

# ------------------------------------------------------------
# 3. Run AnyBCQ post-training quantization
# ------------------------------------------------------------

cd "$OFFICIAL_REPO"

CUDA_VISIBLE_DEVICES="$GPU" python run_clm.py \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_name c4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 8 \
  --do_eval \
  --w_lr1 1e-4 \
  --w_lr2 1e-4 \
  --w_lr3 1e-4 \
  --n_bits_w 2 \
  --add_bits 2 \
  --group_size 128 \
  --num_samples "$NUM_SAMPLES" \
  --iters_w "$ITERS_W" \
  --input_prob 1.0 \
  --asymmetric False \
  --train_beta False \
  --recon_dtype "$RECON_DTYPE" \
  --output_dir "$SAVE_PATH" \
  --cache_dir "$CACHE_PATH"

echo
echo "=========================================="
echo "AnyBCQ PTQ completed."
echo "Checkpoint: $SAVE_PATH"
echo "=========================================="
