#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "AnyBCQ Qwen3-8B Compatibility Smoke Test"
echo "=========================================="

# Small values ONLY for compatibility testing.
# These are NOT final experimental settings.
NUM_SAMPLES=4 \
ITERS_W=5 \
bash "$SCRIPT_DIR/run_ptq.sh"

echo
echo "=========================================="
echo "Compatibility smoke PTQ completed."
echo "Do NOT use these results as final metrics."
echo "=========================================="
