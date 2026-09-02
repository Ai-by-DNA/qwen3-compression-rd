# Wanda KPI Report - Qwen3-8B

## Experimental Setup

- Base model: Qwen/Qwen3-8B
- Model revision: b968826d9c46dd6066d109eabc6255188de91218
- Hardware: NVIDIA GeForce RTX 4090
- Thinking mode: OFF
- Seed: 42
- Benchmark batch size: 1
- Performance protocol: 2 warm-up runs + 5 measured runs
- Max new tokens: 128
- Pruning method: Wanda
- Sparsity: 50%
- Mask structure: 0:0 (unstructured)
- Pruning targets: Linear layers
- Excluded layer: lm_head
- Calibration dataset: WikiText-2 raw v1
- Calibration samples: 128
- Maximum calibration sequence length: 512
- Implementation: llmcompressor 0.13.0

## Environment

The Wanda experiment used a separate Python 3.12.3 environment.

Key versions:

- llmcompressor: 0.13.0
- PyTorch: 2.13.0
- Transformers: 5.14.1
- Datasets: 4.8.5

The complete environment snapshot is stored in `requirements-wanda.txt`.

## Quality Results

| Method | WikiText-2 PPL | MMLU Accuracy | GSM8K Flexible | GSM8K Strict |
|---|---:|---:|---:|---:|
| Wanda 50% | 19.5449 | 63.43% | 69.07% | 34.50% |

## Performance Results

| Metric | Wanda 50% |
|---|---:|
| TTFT | 25.18 ms |
| Total latency | 2658.45 ms |
| Decode throughput | 48.23 tok/s |
| End-to-end throughput | 48.15 tok/s |
| Peak VRAM | 15.29 GiB |
| Peak RAM | 1.78 GiB |
| Checkpoint size | 15.27 GiB |

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

Wanda at 50% unstructured sparsity reduces the number of non-zero weights but causes a noticeable quality degradation.

Under the tested standard dense Hugging Face execution path, unstructured sparsity does not reduce checkpoint size, VRAM usage, or inference latency because the sparse weights are still stored and executed using dense representations.

## Raw Results

Final raw benchmark outputs are stored under:

`results/wanda/`

These include WikiText-2, MMLU, GSM8K, and performance measurements.

## Conclusion

Wanda 50% unstructured pruning successfully introduces sparsity into Qwen3-8B, but the tested dense execution path does not provide practical memory or speed benefits.

Sparse storage and sparse-kernel experiments were investigated separately as follow-up work.
