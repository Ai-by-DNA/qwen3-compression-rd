# Qwen3-8B Compression Experiments
 
> **This is the single README for this project.** It lives on `main` only and is the entry point for anyone arriving from the paper. Feature branches do not carry a duplicate copy of this file.
 
Companion repository for the paper:
 
> *Evaluating Weight Quantization, Pruning, and KV-Cache Compression for Qwen3-8B on a Consumer GPU*
 
This repository contains the configuration files, scripts, and **summary** results used in that study. It is not a drop-in serving stack. Checkpoints and datasets are **not** stored in git.
 
If you want to check a number or reproduce a run, start here. Do not treat `origin/main` as containing every method: the experiments currently live on method-specific branches (see [Repository layout](#repository-layout)).
 
## What was evaluated
 
| Method | Family | Operating point | Branch |
|---|---|---|---|
| Baseline | Uncompressed BF16 | Qwen3-8B, thinking off | `main`, `feature/baseline-kpis` |
| AnyBCQ | Weight PTQ | 4 / 3 / 2-bit, C4 calibration, 256 samples | `feature/anybcq` |
| AWQ | Weight PTQ | W4A16 asymmetric, WikiText-2-raw-v1, 128 samples | `feature/AWQ-Wanda` |
| Wanda | Unstructured pruning | 50% sparsity, WikiText-2-raw-v1, 128 samples | `feature/AWQ-Wanda` |
| TurboQuant | KV-cache quantization | 3-bit, `hackimov/turboquant-kv`, seed 42 | `feature/turboquant` |
| AnyBCQ + Wanda | Combined (exploratory) | Wanda 50% then AnyBCQ 4-bit | `feature/anybcq-wanda` |
 
Related follow-ups (not in the main comparison tables): `feature/sparse-wanda`, `feature/sparse-kernel-wanda`.
 
Common experimental constants:
 
- Model: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`
- Precision: BF16 weights except AnyBCQ **inference**, which uses the official FP16 runtime
- Thinking mode: off (`enable_thinking=False`)
- Seed: 42
- Batch size: 1
- Performance Protocol: 2 warm-up runs + 5 measured runs, 128 new tokens, `do_sample=False`

## Hardware 

All reported numbers were measured on a single idle **NVIDIA GeForce RTX 4090** (24, 564 MiB VRAM), NVIDIA driver 580.159.04, 187 GiB system RAM. Peak VRAM, latency, and OOM behaviour will not necessarily transfer to another GPU. 

## Published numbers vs smoke tests

| Use | Location |
|---|---|
| Numbers reported in the paper | `results/*_summary.csv` and `reports/*_kpi_report.md` on the method branch |
| Pipeline smoke tests | subset WikiText / MMLU / GSM8K runs | **Do not** cite these as paper results |

A clean reproduction of AnyBCQ post-training quantization and the official AnyBCQ WikiTwxt-2 // MMLU evaluation has been verified against the paper (PPL and MMLU match after rounding). Full AWQ, Wanda, TurboQuant, and GSM8K re-runs were not part of that check.

## Repository layout

After checking out a method branch:

```
configs/    method YAML
scripts/    setupt, PTQ/pruning, quality, performance
results/    summary CSV commited to git 
reports/    KPI write-ups (where present)
```

Default checkout of `main` currently contains the **baseline** environment lockfile and this README. To work on a method:

```bash
git fetch origin
git checkout feature/anybcq            #example
```

### Pre-method entry points

| Method | Setup / env | Run | Summary CSV |
|---|---|---|---|
| Baseline | `scripts/baseline/run_baseline_all.sh` | quality + performance | `results/baseline_summary.csv` |
| AnyBCQ | `scripts/anybcq/setup_anybcq.sh` | `scripts/anybcq/run_all_anybcq.sh` | `results/anybcq_summary.csv` |
| AWQ | `requirements-awq.txt`, env `llm-compressor-env` | `scripts/awq/run_awq.py` then `scripts/awq/run_awq_all.sh` | `results/awq_summary.csv` |
| Wanda | `requirements-wanda.txt`, env `wanda-env` | `scripts/wanda/run_wanda.py` then `scripts/wanda/run_wanda_all.sh` | `results/wanda_summary.csv` |
| TurboQuant | `scripts/turboquant/setup_turboquant.sh` | `scripts/turboquant/run_all_turboquant.sh` | `results/turboquant_summary.csv` |
| AnyBCQ+Wanda | see [Known limitations](#known-limitations) | **⚠️ DO NOT RUN — not reproducible yet** (see below) | `results/anybcq_wanda_summary.csv` |

## Software environments 

Do **not** install every method into one virtualenv. The paper's Table 2 lists the **baseline** stack. Method-specific environments used for the reported runs:

| Stack | Python | PyTorch | Transformers | Datasets | Other |
|---|---|---|---|---|---|
| Baseline | 3.12.3 | 2.8.0+cu128 | 4.55.4 | 4.8.5 | lm-evaluation-harness 0.4.12 |
| AnyBCQ | 3.12.3 | 2.8.0+cu128 | 4.51.0 | 5.0.1 | official [AnyBCQ](https://github.com/naver-aics/anybcq) |
| AWQ | 3.12.3 | 2.13.0 | 5.14.1 | 5.0.1 | llmcompressor 0.13.0 |
| Wanda | 3.12.3 | 2.13.0 | 5.14.1 | 4.8.5 | llmcompressor 0.13.0 |
| TurboQuant | 3.12.3 | 2.8.0+cu128 | 5.x (setup script currently pins 5.15.1) | 4.8.5 | lm-eval 0.4.12, Triton kernels |

Snapshots: `requirements-lock.txt` (baseline, on `main`), `requirements-awq.txt` and `requirements-wanda.txt` (on `feature/AWQ-Wanda`).

### Upstream pins (not yet encoded in all setup scripts)

Setup scripts currently `git clone` without a commit pin. The runs that prosuced the paper numbers used:

- AnyBCQ: `c316a4d06fc7714aa6b500b4d607a08ec6fe0b24`
- TurboQuant KV: `12c38c796a7199c93d232ef197cf213809f91afe`

After cloning those repositories, check out the pinned commit before installing.

## Paths used in the original runs

The scripts assume a RunPod-style layout. You can keep it or replace the prefixes consistently.

| Role | Path used in scripts | Recommendation for a local clone |
|---|---|---|
| Git repository | `/workspace/repo` (TurboQuant patch script: `/workspace/repo-turboquant`) | this clone |
| Models / checkpoints | `/workspace/models` | outside git, e.g. `./models` |
| Datasets / caches | `/workspace/datasets` | outside git |
| Virtualenvs | `/workspace/envs` | outside git |
| Raw logs | `/workspace/results` | outside git; **summary** CSVs are committed |

Do not commit model weights or Hugging Face caches.

On some hosts, disable Hugging Face transfer before downloads:

```bash
unset HF_HUB_ENABLE_HF_TRANSFER
```

## Minimal path: inspect published AnyBCQ numbers

No GPU required:

 
```bash
git checkout feature/anybcq
cat results/anybcq_summary.csv
cat reports/anybcq_kpi_report.md
```

## Reproduce AnyBCQ quality (verified)

Requires an RTX 4090-class GPU, C4 access for PTQ, and several hours for full quantization.
 
```bash
git checkout feature/anybcq
 
# Isolated env + official AnyBCQ clone (then: git checkout c316a4d06fc7714aa6b500b4d607a08ec6fe0b24)
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
 
```bash
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
 
```bash
bash scripts/baseline/run_quality.sh
python scripts/baseline/measure_performance.py
```

## AWQ and Wanda 

On `feature/AWQ-Wanda`, create **separate** environments from the committed snapshots; do not reuse the baseline env.
 
```bash
git checkout feature/AWQ-Wanda
# python -m venv ... && pip install -r requirements-awq.txt
python scripts/awq/run_awq.py
bash scripts/awq/run_awq_all.sh
```
Wanda analogously: `requirements-wanda.txt`, `scripts/wanda/run_wanda.py`, `scripts/wanda/run_wanda_all.sh`.

AWQ and Wanda performance in the paper is **Hugging Face Transformers only**. There is no vLLM measurement in this repository.

## TurboQuant

```bash
git checkout feature/turboquant
bash scripts/turboquant/setup_turboquant.sh
# After clone: git -C /workspace/tools/turboquant-kv checkout 12c38c796a7199c93d232ef197cf213809f91afe
bash scripts/turboquant/run_all_turboquant.sh
```

Quality numbers in the paper are a **small-sample screening** (10 WikiText-2, 5 MMLU abstract_algebra, 5 GSM8K), not a full benchmark. Long-context KV memory is measured by `scripts/turboquant/measure_long_context.py`.

## Measurement caveats (same as the paper)

- **AnyBCQ WikiText-2** uses the official AnyBCQ perplexity evaluator, not lm-evaluation-harness. Combined AnyBCQ+Wanda WikiText-2 / C4 PPL use the same official evaluator.
- **AnyBCQ performance** is FP16; baseline, AWQ, Wanda, and TurboQuant performance are BF16.
- **TurboQuant quality** is screening-only.
- **TurboQuant harness** uses a shorter prompt and 5 ms RAM sampling, and does not call `empty_cache()` before each run.

## Known limitations

these are gaps in the supplementary package, not hidden experimental results:

1. **Raw per-run logs** (five-run JSON/log files) remain on the original machine. Git mainly has summary CSVs. Appendix claims in the manuscript should match this until the logs are added.
2. **AnyBCQ+Wanda used local modifications to official AnyBCQ** (`run_clm.py`, `data_utils.py`, `anybcq/evaluate/eval.py`, `anybcq/inference/AnyBCQForCausalLM.py`) that are **not** committed as a patch. **This configuration is not reproducible from this repository yet — do not attempt to run `scripts/anybcq_wanda/run_ptq_wanda_timed.sh` until this notice is removed.**
3. **Setup scripts** clone AnyBCQ and TurboQuant at `HEAD` unless you check out the pins above.
4. **Default `main`** does not contain the method pipelines. Check out the branch in the table at the top.


## License
 
*(To be added.)*
 
## Citation
 
```bibtex
@article{qwen3compression,
  title   = {Evaluating Weight Quantization, Pruning, and KV-Cache Compression for Qwen3-8B on a Consumer GPU},
  author  = {Kotzamanis, George and Tasiopoulou, Anna and Iliadou, Zoi and Voskou, Efstratia},
  year    = {2026},
  note    = {Manuscript}
}
```
 
Replace the `note` field with URL when the paper is published.