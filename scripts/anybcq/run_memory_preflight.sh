#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREFLIGHT_MODEL="/workspace/models/anybcq-qwen3-8b-preflight"
PREFLIGHT_CACHE="/workspace/datasets/anybcq-preflight-cache"
RESULT_DIR="/workspace/results/anybcq"

mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "AnyBCQ RTX 4090 Memory Preflight"
echo "256 samples | 5 iterations | BF16"
echo "=========================================="

SAVE_PATH="$PREFLIGHT_MODEL" \
CACHE_PATH="$PREFLIGHT_CACHE" \
NUM_SAMPLES=256 \
ITERS_W=5 \
RECON_DTYPE=bfloat16 \
bash "$SCRIPT_DIR/run_ptq.sh" \
2>&1 | tee "$RESULT_DIR/memory_preflight.log"

echo
echo "=========================================="
echo "MEMORY PREFLIGHT PASSED"
echo "=========================================="
