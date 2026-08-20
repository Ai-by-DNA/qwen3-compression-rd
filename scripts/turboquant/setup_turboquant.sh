#!/usr/bin/env bash

set -euo pipefail

# RunPod may enable Hugging Face fast transfer globally.
unset HF_HUB_ENABLE_HF_TRANSFER

ENV_PATH="/workspace/envs/turboquant"
OFFICIAL_REPO="/workspace/tools/turboquant-kv"
REPO_URL="https://github.com/hackimov/turboquant-kv.git"

echo "=========================================="
echo "TurboQuant environment setup"
echo "=========================================="

# 1. Create isolated Python environment.
if [ ! -d "$ENV_PATH" ]; then
    python -m venv "$ENV_PATH"
fi

source "$ENV_PATH/bin/activate"

python -m pip install --upgrade pip

# 2. Use the same PyTorch/CUDA family as the baseline.
python -m pip install torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Clone the TurboQuant implementation outside our project repo.
mkdir -p /workspace/tools

if [ ! -d "$OFFICIAL_REPO/.git" ]; then
    git clone "$REPO_URL" "$OFFICIAL_REPO"
fi

# 4. Install TurboQuant with Hugging Face + Triton support.
cd "$OFFICIAL_REPO"

python -m pip install -e ".[hf,triton]"

# TurboQuant currently requires the Transformers 5.x cache API.
python -m pip install transformers==5.15.1

# Apply the compatibility fixes required by Transformers 5.15.1.
python "/workspace/repo-turboquant/scripts/turboquant/patch_transformers5_compat.py"

# 5. Extra packages used by our common evaluation protocol.
python -m pip install \
    lm-eval==0.4.12 \
    datasets==4.8.5 \
    accelerate==1.14.0 \
    psutil==7.2.2 \
    pyyaml

echo
echo "=========================================="
echo "TurboQuant setup completed."
echo "Environment: $ENV_PATH"
echo "Repository:  $OFFICIAL_REPO"
echo "=========================================="
