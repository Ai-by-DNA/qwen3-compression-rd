#!/usr/bin/env bash
set -euo pipefail

cd /workspace/repo-anybcq-wanda
source /workspace/envs/anybcq-wanda/bin/activate

RESULT_DIR="/workspace/results/anybcq-wanda"
mkdir -p "$RESULT_DIR"

START_TS=$(date +%s)
START_ISO=$(date -Iseconds)

echo "=========================================="
echo "AnyBCQ + Wanda PTQ"
echo "Started: $START_ISO"
echo "=========================================="

NUM_SAMPLES="${NUM_SAMPLES:-256}" \
ITERS_W="${ITERS_W:-5000}" \
bash scripts/anybcq_wanda/run_ptq_wanda.sh

END_TS=$(date +%s)
END_ISO=$(date -Iseconds)
ELAPSED=$((END_TS - START_TS))

python - <<PY
import json

seconds = $ELAPSED

result = {
    "stage": "anybcq_wanda_ptq",
    "start_time": "$START_ISO",
    "end_time": "$END_ISO",
    "runtime_seconds": seconds,
    "runtime_minutes": seconds / 60,
    "num_samples": int("${NUM_SAMPLES:-256}"),
    "iters_w": int("${ITERS_W:-5000}"),
    "input_model": "/workspace/models/Qwen3-8B-WANDA",
    "output_model": "/workspace/models/anybcq-wanda-qwen3-8b",
}

with open(
    "/workspace/results/anybcq-wanda/ptq_runtime.json",
    "w"
) as f:
    json.dump(result, f, indent=2)

print()
print("==========================================")
print("PTQ runtime")
print("==========================================")
print(f"{seconds} seconds")
print(f"{seconds/60:.2f} minutes")
PY
