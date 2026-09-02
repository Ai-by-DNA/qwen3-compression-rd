# AWQ KPI Report - Qwen3-8B

## Experimental Setup

- Base model: Qwen/Qwen3-8B
- Model revision: b968826d9c46dd6066d109eabc6255188de91218
- Hardware: NVIDIA GeForce RTX 4090
- Thinking mode: OFF
- Seed: 42
- Benchmark batch size: 1
- Performance protocol: 2 warm-up runs + 5 measured runs
- Max new tokens: 128
- Quantization: W4A16 asymmetric
- Implementation: llmcompressor 0.13.0
- AWQ duo scaling: both
- Quantization targets: Linear layers
- Excluded layer: lm_head
- Calibration dataset: WikiText-2 raw v1
- Calibration samples: 128
- Maximum calibration sequence length: 512

## Environment

The AWQ experiment used a separate Python 3.12.3 environment.

Key versions:

- llmcompressor: 0.13.0
- PyTorch: 2.13.0
- Transformers: 5.14.1
- Datasets: 5.0.1

The complete environment snapshot is stored in `requirements-awq.txt`.

## Quality Results

| Method | WikiText-2 PPL | MMLU Accuracy | GSM8K Flexible | GSM8K Strict |
|---|---:|---:|---:|---:|
| AWQ W4A16 | 14.0275 | 73.94% | 83.02% | 80.59% |

## Performance Results

Hugging Face Transformers runtime:

| Metric | AWQ W4A16 |
|---|---:|
| TTFT | 25.79 ms |
| Total latency | 2676.37 ms |
| Decode throughput | 47.91 tok/s |
| End-to-end throughput | 47.83 tok/s |
| Peak VRAM | 15.44 GiB |
| Peak RAM | 1.74 GiB |
| Checkpoint size | 5.69 GiB |

## Comparison with Baseline

Baseline reference:

- WikiText-2 PPL: 13.4824
- MMLU: 74.95%
- GSM8K flexible: 81.20%
- GSM8K strict: 75.59%
- TTFT: 25.16 ms
- Latency: 2666.89 ms
- Decode throughput: 48.07 tok/s
- Peak VRAM: 15.29 GiB
- Checkpoint size: 15.27 GiB

AWQ substantially reduces checkpoint size while retaining quality close to the baseline.

Under the tested Hugging Face runtime, the compressed checkpoint did not translate into a corresponding runtime VRAM or latency improvement.

## Raw Results

Final raw benchmark outputs are stored under:

`results/awq/`

These include WikiText-2, MMLU, GSM8K, and Hugging Face performance measurements.

## Conclusion

AWQ provides strong checkpoint compression with very limited quality degradation for Qwen3-8B.

In the tested standard Hugging Face execution path, however, the smaller checkpoint did not provide a corresponding runtime memory or throughput advantage.
