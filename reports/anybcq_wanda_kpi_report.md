# AnyBCQ + Wanda KPI Report

## Configuration

- Base model: Qwen3-8B
- Pruning: Wanda, 50% sparsity
- Quantization: AnyBCQ
- Calibration samples: 256
- PTQ iterations: 5000
- Group size: 128
- Evaluated precision: 4-bit
- GPU: NVIDIA GeForce RTX 4090
- PTQ runtime: 12663 s (211.05 min)

## Quality Results

| Method | WikiText-2 PPL | C4 PPL | MMLU Accuracy | GSM8K Flexible | GSM8K Strict |
|---|---:|---:|---:|---:|---:|
| AnyBCQ + Wanda 4-bit | 16.8990 | 29.2868 | 24.39% | 16.53% | 9.17% |

## Performance Results

| Method | TTFT | Latency | Decode Speed | End-to-End Speed | Peak VRAM | Peak RAM | Checkpoint Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnyBCQ + Wanda 4-bit | 57.41 ms | 2298.47 ms | 56.67 tok/s | 55.69 tok/s | 7.00 GiB | 1.61 GiB | 6.87 GiB |

## Comparison with Baseline

Baseline reference:

- MMLU: 74.95%
- GSM8K flexible: 81.20%
- GSM8K strict: 75.59%
- TTFT: 25.16 ms
- Latency: 2666.89 ms
- Decode speed: 48.07 tok/s
- Peak VRAM: 15.29 GiB
- Model size: 15.27 GiB

The AnyBCQ + Wanda 4-bit combination substantially reduces model size and GPU memory usage while increasing decode throughput and reducing total latency relative to the baseline.

However, the combination shows severe quality degradation on reasoning-oriented benchmarks. MMLU decreases to 24.39%, while GSM8K flexible and strict exact-match scores decrease to 16.53% and 9.17%, respectively.

In contrast, WikiText-2 and C4 perplexity remain substantially less degraded than the reasoning benchmarks. This suggests that general next-token prediction ability is retained to a greater extent than downstream reasoning performance.

The current results therefore indicate that the sequential Wanda pruning + AnyBCQ quantization pipeline provides strong compression and inference-efficiency benefits, but does not preserve sufficient reasoning quality in its current form.

## Experimental Note

The current pipeline applies AnyBCQ post-training quantization after Wanda pruning. The quantization procedure was not explicitly modified to preserve the Wanda pruning mask. Further analysis is required to determine whether the original sparsity pattern is retained after quantization and whether this interaction contributes to the observed reasoning degradation.
