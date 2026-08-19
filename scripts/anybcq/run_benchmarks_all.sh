#!/usr/bin/env bash

set -euo pipefail

# RunPod enables this globally, but hf_transfer is not installed.
unset HF_HUB_ENABLE_HF_TRANSFER

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="/workspace/models/anybcq-qwen3-8b"
RESULT_ROOT="/workspace/results/anybcq"

echo "=========================================="
echo "AnyBCQ Qwen3-8B Full Benchmark Pipeline"
echo "=========================================="

# ------------------------------------------------------------
# Check that the quantized model exists
# ------------------------------------------------------------

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: AnyBCQ checkpoint not found at:"
    echo "$MODEL_PATH"
    echo
    echo "Run the full PTQ step first."
    exit 1
fi

mkdir -p "$RESULT_ROOT"

# ------------------------------------------------------------
# 1. Official AnyBCQ quality evaluation
#    WikiText-2 / MMLU
# ------------------------------------------------------------

echo
echo "=========================================="
echo "STEP 1: WikiText / MMLU"
echo "=========================================="

bash "$SCRIPT_DIR/run_official_eval.sh"


# ------------------------------------------------------------
# 2. GSM8K
#    Same protocol as baseline
#    4-bit -> 3-bit -> 2-bit
# ------------------------------------------------------------

echo
echo "=========================================="
echo "STEP 2: GSM8K"
echo "=========================================="

bash "$SCRIPT_DIR/run_gsm8k_all.sh"


# ------------------------------------------------------------
# 3. Performance KPIs
#    TTFT / latency / tokens/sec / VRAM / RAM
#    4-bit -> 3-bit -> 2-bit
# ------------------------------------------------------------

echo
echo "=========================================="
echo "STEP 3: Performance KPIs"
echo "=========================================="

bash "$SCRIPT_DIR/run_performance_all.sh"


echo
echo "=========================================="
echo "AnyBCQ full benchmark pipeline completed."
echo "Results: $RESULT_ROOT"
echo "=========================================="
