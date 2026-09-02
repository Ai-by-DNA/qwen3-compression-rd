#!/usr/bin/env bash

set -euo pipefail

ENV_PATH="/workspace/envs/anybcq"
OFFICIAL_REPO="/workspace/tools/anybcq-official"
OFFICIAL_REPO_URL="https://github.com/naver-aics/anybcq.git"

echo "=========================================="
echo "AnyBCQ environment setup"
echo "=========================================="

# ------------------------------------------------------------
# 1. Create isolated Python environment
# ------------------------------------------------------------

if [ ! -d "$ENV_PATH" ]; then
    python -m venv "$ENV_PATH"
fi

source "$ENV_PATH/bin/activate"

python -m pip install --upgrade pip

# ------------------------------------------------------------
# 2. Install PyTorch for the RunPod CUDA 12.8 environment
# ------------------------------------------------------------

pip install torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------
# 3. Clone the official AnyBCQ implementation
# ------------------------------------------------------------

mkdir -p /workspace/tools

if [ ! -d "$OFFICIAL_REPO/.git" ]; then
    git clone "$OFFICIAL_REPO_URL" "$OFFICIAL_REPO"
fi
git -C "$OFFICIAL_REPO" checkout c316a4d06fc7714aa6b500b4d607a08ec6fe0b24

# ------------------------------------------------------------
# 4. Install official AnyBCQ package and dependencies
# ------------------------------------------------------------

cd "$OFFICIAL_REPO"

pip install -e .

# ------------------------------------------------------------
# 5. Build/install the official AnyBCQ CUDA kernel
# ------------------------------------------------------------

cd "$OFFICIAL_REPO/anybcq/inference/custom_kernel"

# Ninja is used by the official PyTorch CUDA extension build.
python -m pip install ninja

# RTX 4090 is Ada (sm_89).
# Enable the Ada architecture that is commented out upstream.
sed -i \
  's/# "-gencode=arch=compute_89,code=sm_89", # Ada/"-gencode=arch=compute_89,code=sm_89", # Ada/' \
  setup.py

# Build inside the current AnyBCQ environment so setup.py can import torch.
MAX_JOBS=8 CC=g++ python -m pip install --no-build-isolation .

echo
echo "=========================================="
echo "AnyBCQ setup completed."
echo "Environment: $ENV_PATH"
echo "Official repo: $OFFICIAL_REPO"
echo "=========================================="
