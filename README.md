# Qwen3-8B Compression Experiments

> **This is the central README for this project.** The `main` branch is the entry point for anyone arriving from the paper; method-specific branches may contain their own documentation.

Companion repository for the paper:

> *Evaluating Weight Quantization, Pruning, and KV-Cache Compression for Qwen3-8B on a Consumer GPU*

Repository: [https://github.com/Ai-by-DNA/qwen3-compression-rd](https://github.com/Ai-by-DNA/qwen3-compression-rd)

This repository contains the configuration files, scripts, summary CSVs, and **raw per-run logs** used in that study. It is not a drop-in serving stack. Checkpoints and datasets are **not** stored in git.

If you want to check a number or reproduce a run, start here. Do not treat `origin/main` as containing every method: the experiments live on method-specific branches (see [Repository layout](https://github.com/Ai-by-DNA/qwen3-compression-rd/blob/main/README.md#repository-layout)). Cloning `main` alone is not sufficient to reproduce AnyBCQ, AWQ, Wanda, TurboQuant, or the combined configuration.

## What was evaluated

| **Method**     | **Family**             | **Operating point**                              | **Branch**                      |
| -------------- | ---------------------- | ------------------------------------------------ | ------------------------------- |
| Baseline       | Uncompressed BF16      | Qwen3-8B, thinking off                           | `main`, `feature/baseline-kpis` |
| AnyBCQ         | Weight PTQ             | 4 / 3 / 2-bit, C4 calibration, 256 samples       | `feature/anybcq`                |
| AWQ            | Weight PTQ             | W4A16 asymmetric, WikiText-2-raw-v1, 128 samples | `feature/awq`                   |
| Wanda          | Unstructured pruning   | 50% sparsity, WikiText-2-raw-v1, 128 samples     | `feature/wanda`                 |
| TurboQuant     | KV-cache quantization  | 3-bit, `hackimov/turboquant-kv`, seed 42         | `feature/turboquant`            |
| AnyBCQ + Wanda | Combined (exploratory) | Wanda 50% then AnyBCQ 4-bit                      | `feature/anybcq-wanda`          |

Common experimental constants:

- Model: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`
- Precision: BF16 weights except AnyBCQ **inference**, which uses the official FP16 runtime
- Thinking mode: off (`enable_thinking=False`)
- Seed: 42
- Batch size: 1
- Performance protocol: 2 warm-up runs + 5 measured runs, 128 new tokens, `do_sample=False`

## Hardware

All reported numbers were measured on a single idle **NVIDIA GeForce RTX 4090** (24,564 MiB VRAM), NVIDIA driver 580.159.04, 187 GiB system RAM. Peak VRAM, latency, and OOM behaviour will not necessarily transfer to another GPU.

## Published numbers vs smoke tests

| **Use**                                                  | **Location**                                                                |
| -------------------------------------------------------- | --------------------------------------------------------------------------- |
| Numbers reported in the paper                            | `results/*_summary.csv` and `reports/*_kpi_report.md` on the method branch  |
| Raw per-run logs (five measured runs, quality JSON/logs) | `results/<method>/` on the method branch                                    |
| Pipeline smoke tests                                     | subset WikiText / MMLU / GSM8K runs. **Do not** cite these as paper results |

A clean reproduction of AnyBCQ post-training quantization and the official AnyBCQ WikiText-2 / MMLU evaluation has been verified against the paper (PPL and MMLU match after rounding). Full AWQ, Wanda, TurboQuant, and GSM8K re-runs were not part of that check.

## Repository layout

After checking out a method branch:

```
configs/    method YAML
scripts/    setup, PTQ/pruning, quality, performance
results/    summary CSVs and raw logs committed to git
reports/    KPI write-ups (where present)
patches/    compatibility patches (AnyBCQ+Wanda only)

```

Default checkout of `main` currently contains the **baseline** environment lockfile and this README. To work on a method:

```
git fetch origin
git checkout feature/anybcq            # example
```

### Per-method entry points

| **Method**   | **Setup / env**                                                   | **Run**                                                            | **Summary CSV**                    |
| ------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------- |
| Baseline     | `scripts/baseline/run_baseline_all.sh`                            | quality + performance                                              | `results/baseline_summary.csv`     |
| AnyBCQ       | `scripts/anybcq/setup_anybcq.sh`                                  | `scripts/anybcq/run_all_anybcq.sh`                                 | `results/anybcq_summary.csv`       |
| AWQ          | `requirements-awq.txt`, env `llm-compressor-env`                  | `scripts/awq/run_awq.py` then `scripts/awq/run_awq_all.sh`         | `results/awq_summary.csv`          |
| Wanda        | `requirements-wanda.txt`, env `wanda-env`                         | `scripts/wanda/run_wanda.py` then `scripts/wanda/run_wanda_all.sh` | `results/wanda_summary.csv`        |
| TurboQuant   | `scripts/turboquant/setup_turboquant.sh`                          | `scripts/turboquant/run_all_turboquant.sh`                         | `results/turboquant_summary.csv`   |
| AnyBCQ+Wanda | official AnyBCQ clone + `patches/anybcq_wanda/anybcq_wanda.patch` | `scripts/anybcq_wanda/run_ptq_wanda_timed.sh`                      | `results/anybcq_wanda_summary.csv` |

## Software environments

Do **not** install every method into one virtualenv. The paper's Table 2 lists the **baseline** stack. Method-specific environments used for the reported runs:

| **Stack**  | **Python** | **PyTorch** | **Transformers**                         | **Datasets** | **Other**                                               |
| ---------- | ---------- | ----------- | ---------------------------------------- | ------------ | ------------------------------------------------------- |
| Baseline   | 3.12.3     | 2.8.0+cu128 | 4.55.4                                   | 4.8.5        | lm-evaluation-harness 0.4.12                            |
| AnyBCQ     | 3.12.3     | 2.8.0+cu128 | 4.51.0                                   | 5.0.1        | official [AnyBCQ](https://github.com/naver-aics/anybcq) |
| AWQ        | 3.12.3     | 2.13.0      | 5.14.1                                   | 5.0.1        | llmcompressor 0.13.0                                    |
| Wanda      | 3.12.3     | 2.13.0      | 5.14.1                                   | 4.8.5        | llmcompressor 0.13.0                                    |
| TurboQuant | 3.12.3     | 2.8.0+cu128 | 5.x (setup script currently pins 5.15.1) | 4.8.5        | lm-eval 0.4.12, Triton kernels                          |

Snapshots: `requirements-lock.txt` (baseline, on `main`), `requirements-awq.txt` (on `feature/awq`) and `requirements-wanda.txt` (on `feature/wanda`).

### Upstream pins

The setup scripts check out the commits used for the paper numbers after cloning:

- AnyBCQ: `c316a4d06fc7714aa6b500b4d607a08ec6fe0b24` (`scripts/anybcq/setup_anybcq.sh` on `feature/anybcq` and `feature/anybcq-wanda`)
- TurboQuant KV: `12c38c796a7199c93d232ef197cf213809f91afe` (`scripts/turboquant/setup_turboquant.sh` on `feature/turboquant`)

If you clone those upstream repositories by hand, check out the same commits before installing.

## Paths used in the original runs

The scripts assume a RunPod-style layout. You can keep it or replace the prefixes consistently.

| **Role**             | **Path used in scripts**                                                  | **Recommendation for a local clone**                                                              |
| -------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Git repository       | `/workspace/repo` (TurboQuant patch script: `/workspace/repo-turboquant`) | this clone                                                                                        |
| Models / checkpoints | `/workspace/models`                                                       | outside git, e.g. `./models`                                                                      |
| Datasets / caches    | `/workspace/datasets`                                                     | outside git                                                                                       |
| Virtualenvs          | `/workspace/envs`                                                         | outside git                                                                                       |
| Working result dumps | `/workspace/results`                                                      | outside git; **summary CSVs and raw logs used in the paper are committed** on the method branches |

Do not commit model weights or Hugging Face caches.

On some hosts, disable Hugging Face transfer before downloads:

```
unset HF_HUB_ENABLE_HF_TRANSFER
```

## Minimal path: inspect published AnyBCQ numbers

No GPU required:

```
git checkout feature/anybcq
cat results/anybcq_summary.csv
cat reports/anybcq_kpi_report.md
```

Raw per-run files for that method are under `results/anybcq/`.

## Reproduce AnyBCQ quality (verified)

Requires an RTX 4090-class GPU, C4 access for PTQ, and several hours for full quantization.

```
git checkout feature/anybcq

# Isolated env + official AnyBCQ clone at c316a4d06fc7714aa6b500b4d607a08ec6fe0b24
bash scripts/anybcq/setup_anybcq.sh

hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir /workspace/models/Qwen3-8B

# Full PTQ: 256 C4 samples, 5000 iterations (about 3.5 hours on RTX 4090)
bash scripts/anybcq/run_ptq.sh

# Official evaluator: WikiText-2 PPL and MMLU for 2/3/4-bit
bash scripts/anybcq/run_official_eval.sh
```

Expected WikiText-2 PPL / MMLU (paper, 4-decimal / 2-decimal rounding): 10.8231 / 70.48%, 12.9522 / 66.56%, 40.9711 / 33.75%.

GSM8K for AnyBCQ is a **separate** lm-eval path (`scripts/anybcq/evaluate_gsm8k.py`) and uses `is_fp16=False` at load time. Performance figures use the official FP16 inference runtime. Do not assume quality and latency share one dtype.

## Baseline environment (Table 2 stack)

```
python -m venv /workspace/envs/qwen3-baseline
source /workspace/envs/qwen3-baseline/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.5.2
pip install transformers==4.55.4
pip install "lm_eval[hf]==0.4.12" "datasets==4.8.5"
# or: pip install -r requirements-lock.txt

hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir /workspace/models/Qwen3-8B
```

Then, on `feature/baseline-kpis` or a branch that includes `scripts/baseline/`:

```
bash scripts/baseline/run_quality.sh
python scripts/baseline/measure_performance.py
```

## AWQ and Wanda

Use **separate** environments from the committed snapshots on `feature/awq` and `feature/wanda`; do not reuse the baseline env.

```
git checkout feature/awq
# python -m venv ... && pip install -r requirements-awq.txt
python scripts/awq/run_awq.py
bash scripts/awq/run_awq_all.sh
```

For Wanda, check out `feature/wanda` and use `requirements-wanda.txt`, `scripts/wanda/run_wanda.py`, and `scripts/wanda/run_wanda_all.sh`.

AWQ and Wanda performance in the paper is **Hugging Face Transformers only**. There is no vLLM measurement in this repository.

## TurboQuant

```
git checkout feature/turboquant
bash scripts/turboquant/setup_turboquant.sh
# setup checks out 12c38c796a7199c93d232ef197cf213809f91afe
bash scripts/turboquant/run_all_turboquant.sh
```

For reproducible TurboQuant quality evaluation, use
`scripts/turboquant/fixed_seed_quality_preflight.py`.

This script uses a fixed quantizer seed (42) and evaluates
10 WikiText-2, 5 MMLU (abstract_algebra), and 5 GSM8K samples.

Quality numbers in the paper are a **small-sample screening** (10 WikiText-2, 5 MMLU abstract_algebra, 5 GSM8K), not a full benchmark. Long-context KV memory is measured by `scripts/turboquant/measure_long_context.py`.

## AnyBCQ + Wanda (combined)

This configuration is reproducible from `feature/anybcq-wanda`, but **not** from official AnyBCQ without the committed patch.

```
git checkout feature/anybcq-wanda
bash scripts/anybcq/setup_anybcq.sh

# Apply the exact local modifications used for the paper
# (run_clm.py, data_utils.py, anybcq/evaluate/eval.py,
#  anybcq/inference/AnyBCQForCausalLM.py)
git -C /path/to/anybcq apply "$(pwd)/patches/anybcq_wanda/anybcq_wanda.patch"

bash scripts/anybcq_wanda/run_ptq_wanda_timed.sh
```

See `patches/anybcq_wanda/README.md` on that branch for the upstream commit and apply command. Do not skip the patch.

## Measurement caveats (same as the paper)

- **AnyBCQ WikiText-2** uses the official AnyBCQ perplexity evaluator, not lm-evaluation-harness. Combined AnyBCQ+Wanda WikiText-2 / C4 PPL use the same official evaluator.
- **AnyBCQ performance** is FP16; baseline, AWQ, Wanda, and TurboQuant performance are BF16.
- **TurboQuant quality** is screening-only.
- **TurboQuant harness** uses a shorter prompt and 5 ms RAM sampling, and does not call `empty_cache()` before each run.

## Known limitations

These are remaining constraints of the supplementary package, not hidden experimental results:

1. **Default `main`** does not contain the method pipelines, raw logs, or patches. Check out the branch in the table at the top (this matches Appendix A.3 of the paper).
2. **Checkpoints and datasets** are not in git. Reproduction needs a local Qwen3-8B download at the pinned revision and, for AnyBCQ, C4 access.
3. **AWQ, Wanda, TurboQuant, and GSM8K** were not re-run from a clean clone as part of the AnyBCQ verification; the committed logs are the source of the paper numbers for those methods.

## License

Original code authored for this repository by Ai by DNA, across all branches, is licensed under the [Apache License, Version 2.0](https://github.com/Ai-by-DNA/qwen3-compression-rd/blob/main/LICENSE).

Third-party code, implementations, dependencies, and datasets remain subject to their respective licenses. See [THIRD_PARTY_LICENSES.md](https://github.com/Ai-by-DNA/qwen3-compression-rd/blob/main/THIRD_PARTY_LICENSES.md) and [NOTICE](https://github.com/Ai-by-DNA/qwen3-compression-rd/blob/main/NOTICE) on the `main` branch for repository-wide licensing information and applicable notices. Preserve original copyright and license notices where required.

## Citation

```
@article{qwen3compression,
  title   = {Evaluating Weight Quantization, Pruning, and KV-Cache Compression for Qwen3-8B on a Consumer GPU},
  author  = {Kotzamanis, George and Tasiopoulou, Anna and Iliadou, Zoi and Voskou, Efstratia},
  year    = {2026}
}
```
