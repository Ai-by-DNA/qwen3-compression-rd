# Qwen3 Compression R&D

Common baseline setup for the Qwen3-8B compression experiments.

## Persistent RunPod paths

- Repository: /workspace/repo
- Models/checkpoints: /workspace/models
- Datasets: /workspace/datasets
- Environments: /workspace/envs
- Results: /workspace/results

Models and checkpoints must not be committed to GitHub.

## Baseline configuration

- Model: Qwen/Qwen3-8B
- Revision: b968826d9c46dd6066d109eabc6255188de91218
- Local model path: /workspace/models/Qwen3-8B
- dtype: bfloat16
- Thinking mode: OFF
- Seed: 42
- Batch size: 1

## Hardware

- GPU: NVIDIA GeForce RTX 4090
- VRAM: 24564 MiB
- NVIDIA Driver: 580.159.04

## Verified software

- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- PyTorch CUDA build: 12.8
- Transformers: 4.55.4
- Datasets: 4.8.5
- lm-evaluation-harness: 0.4.12

The complete Python environment snapshot is stored in requirements-lock.txt.

## Environment setup

Create the baseline virtual environment:

    python -m venv /workspace/envs/qwen3-baseline

Activate it:

    source /workspace/envs/qwen3-baseline/bin/activate

Install PyTorch:

    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

Install baseline dependencies:

    pip install numpy==2.5.2
    pip install transformers==4.55.4
    pip install "lm_eval[hf]==0.4.12" "datasets==4.8.5"

## Hugging Face download setting on RunPod

If HF_HUB_ENABLE_HF_TRANSFER=1 is set, disable it for the current shell before downloading:

    unset HF_HUB_ENABLE_HF_TRANSFER

## Download baseline model

Download the exact model revision outside the Git repository:

    hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir /workspace/models/Qwen3-8B

## Initial validation completed

The common baseline has been validated with:

- Local Qwen3-8B inference on the RTX 4090
- Thinking mode disabled
- WikiText-2 benchmark smoke test
- MMLU benchmark smoke test
- GSM8K benchmark smoke test

Smoke-test/subset scores are only for pipeline validation and must not be reported as final benchmark results.

## Environment rule

Do not modify the common baseline environment for method-specific dependencies.

If AWQ, AnyBCQ, TurboQuant, Wanda, or another method requires different or incompatible dependencies, create a separate virtual environment.

Final comparative benchmarks must run sequentially on the same RTX 4090.

## AWQ environment

The AWQ experiment used a separate Python 3.12.3 environment.

Create it with:

    python -m venv /workspace/envs/llm-compressor-env
    source /workspace/envs/llm-compressor-env/bin/activate
    pip install -r requirements-awq.txt

Key versions:
- llmcompressor: 0.13.0
- PyTorch: 2.13.0
- Transformers: 5.14.1
- Datasets: 5.0.1
