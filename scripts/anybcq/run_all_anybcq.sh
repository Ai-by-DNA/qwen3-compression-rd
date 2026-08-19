#!/usr/bin/env bash

set -euo pipefail
# RunPod may enable Hugging Face fast transfer globally.
# Disable it because hf_transfer is not required by this experiment.
unset HF_HUB_ENABLE_HF_TRANSFER

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULT_ROOT="/workspace/results/anybcq"
FULL_MODEL="/workspace/models/anybcq-qwen3-8b"

mkdir -p "$RESULT_ROOT"

# Save the complete pipeline output in one log file too.
exec > >(tee -a "$RESULT_ROOT/full_pipeline.log") 2>&1

echo "=========================================="
echo "AnyBCQ Qwen3-8B COMPLETE PIPELINE"
echo "=========================================="

# ------------------------------------------------------------
# STEP 1 - Setup isolated AnyBCQ environment
# ------------------------------------------------------------

echo
echo "STEP 1/4: AnyBCQ setup"
bash "$SCRIPT_DIR/setup_anybcq.sh"


# ------------------------------------------------------------
# STEP 2 - Small compatibility test
# ------------------------------------------------------------

echo
echo "STEP 2/4: Compatibility smoke test"
bash "$SCRIPT_DIR/run_smoke_ptq.sh"

echo
echo "Smoke test PASSED."
echo "Continuing automatically to the full experiment."


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

if [ -d "$FULL_MODEL" ] && [ "$(ls -A "$FULL_MODEL" 2>/dev/null)" ]; then
    echo
    echo "ERROR: Full AnyBCQ output directory is not empty:"
    echo "$FULL_MODEL"
    echo "Stopping to avoid overwriting an existing experiment."
    exit 1
fi


# ------------------------------------------------------------
# STEP 3 - Full AnyBCQ PTQ
# ------------------------------------------------------------

echo
echo "STEP 3/4: Full AnyBCQ PTQ"
echo "Calibration: 512 samples"
echo "Iterations: 5000"

bash "$SCRIPT_DIR/run_ptq.sh"


# ------------------------------------------------------------
# STEP 4 - All quality + performance benchmarks
# ------------------------------------------------------------

echo
echo "STEP 4/4: Full benchmark suite"

bash "$SCRIPT_DIR/run_benchmarks_all.sh"


echo
echo "=========================================="
echo "AnyBCQ COMPLETE PIPELINE FINISHED"
echo "Results: $RESULT_ROOT"
echo "=========================================="
