# Qwen3-8B Baseline KPI Report

## 1. Purpose

This report records the baseline quality, performance, and resource-utilization KPIs for the uncompressed Qwen3-8B model.

The baseline serves as the reference point for the subsequent compression experiments, including AWQ, AnyBCQ, TurboQuant, Wanda, and their selected combinations.

---

## 2. Experimental Setup

### Model

- Model: Qwen/Qwen3-8B
- Revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Precision: bfloat16
- Compression: None
- Thinking mode: Disabled
- Local model path: `/workspace/models/Qwen3-8B`

### Hardware

- GPU: NVIDIA GeForce RTX 4090
- GPU VRAM: 24,564 MiB
- NVIDIA Driver: 580.159.04
- System RAM: 187 GiB

### Software

- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- Transformers: 4.55.4
- datasets: 4.8.5
- lm-evaluation-harness: 0.4.12

---

## 3. Quality KPIs

| Benchmark | Metric | Baseline Result |
|---|---|---:|
| WikiText-2 | Word Perplexity (PPL) | 13.4824 |
| MMLU | Accuracy | 74.95% |
| GSM8K | Flexible Exact Match | 81.20% |
| GSM8K | Strict Exact Match | 75.59% |

### Benchmark Configuration

#### WikiText-2

- Full benchmark
- Batch size: 1
- Seed: 42
- Maximum evaluation sequence length: 2048 tokens

A maximum sequence length of 2048 tokens was used because the initial full-context evaluation exceeded the available RTX 4090 VRAM. The same evaluation configuration must therefore be preserved for future compressed-model comparisons.

#### MMLU

- Full MMLU benchmark
- 5-shot evaluation
- Batch size: 1
- Seed: 42
- Maximum sequence length: 2048 tokens

#### GSM8K

- Full GSM8K benchmark
- 1319 evaluation problems
- 5-shot evaluation
- Thinking mode: Disabled
- Chat template: Enabled
- Batch size: 1
- Seed: 42

---

## 4. Performance and Resource KPIs

Performance measurements were conducted using a fixed prompt and identical generation settings.

Two warm-up runs were executed before five measured runs. The reported values below are the averages across the five measured runs.

| KPI | Baseline Result |
|---|---:|
| Average TTFT | 25.16 ms |
| Average Decode Speed | 48.07 tokens/s |
| Average End-to-End Speed | 48.00 tokens/s |
| Average Peak VRAM Allocated | 15.29 GiB |
| Average Peak VRAM Reserved | 15.45 GiB |
| Average Peak RAM (Python process) | 1.58 GiB |
| Model Directory Size | 15.27 GiB |
| Weight Size | 15.26 GiB |

### Performance Protocol

- Batch size: 1
- Thinking mode: Disabled
- Seed: 42
- Warm-up runs: 2
- Measured runs: 5
- Maximum new tokens: 128
- Deterministic generation (`do_sample=False`)
- Same prompt used for every measured run
- GPU measurements performed with no other GPU workloads running

TTFT (Time To First Token) represents the time between submitting the input and generating the first output token.

Decode speed represents the token generation rate after the first generated token.

Peak VRAM reports the maximum GPU memory required during inference.

Peak RAM represents the maximum RAM used by the benchmark Python process rather than the total RAM consumption of the complete server.

---

## 5. Baseline Summary

The uncompressed Qwen3-8B baseline requires approximately 15.26 GiB for model weights and approximately 15.29 GiB of peak allocated GPU memory during the fixed performance test.

Under the current benchmark protocol, the model achieved:

- WikiText-2 PPL: 13.4824
- MMLU Accuracy: 74.95%
- GSM8K Flexible Exact Match: 81.20%
- GSM8K Strict Exact Match: 75.59%
- TTFT: 25.16 ms
- Decode Speed: 48.07 tokens/s

These measurements establish the reference point against which the resource savings and quality changes introduced by the compression methods will be compared.

Future compression experiments should preserve the same benchmark configuration and should be executed sequentially on the same RTX 4090 to ensure fair comparison.

---

## 6. Stored Results

Detailed benchmark outputs are stored under:

`/workspace/results/baseline/`

Summary KPI values are also available in:

`results/baseline_summary.csv`

Performance measurements are stored in:

`/workspace/results/baseline/performance.json`
