# AnyBCQ KPI Report - Qwen3-8B

## Experimental Setup

- Base model: Qwen/Qwen3-8B
- Model revision: b968826d9c46dd6066d109eabc6255188de91218
- Hardware: NVIDIA GeForce RTX 4090
- Thinking mode: OFF
- Seed: 42
- Benchmark batch size: 1
- Performance protocol: 2 warm-up runs + 5 measured runs
- Max new tokens: 128
- AnyBCQ precisions: 4-bit, 3-bit, 2-bit
- Calibration dataset: C4
- Calibration samples: 256
- PTQ iterations: 5000
- Group size: 128
- Input probability: 1.0
- Reconstruction dtype: BF16
- PTQ train batch size: 1

The calibration size was reduced from 512 to 256 samples because the 512-sample configuration exceeded the 24 GB VRAM available on the RTX 4090.

## Quality Results

| Method | WikiText-2 PPL | MMLU Accuracy | GSM8K Flexible | GSM8K Strict |
|---|---:|---:|---:|---:|
| AnyBCQ 4-bit | 10.8231 | 70.48% | 69.37% | 40.49% |
| AnyBCQ 3-bit | 12.9522 | 66.56% | 45.64% | 1.14% |
| AnyBCQ 2-bit | 40.9711 | 33.75% | 2.27% | 0.00% |

## Performance Results

| Method | TTFT | Latency | Decode Speed | End-to-End Speed | Peak VRAM | Peak RAM | Checkpoint Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnyBCQ 4-bit | 57.38 ms | 2334.06 ms | 55.79 tok/s | 54.85 tok/s | 7.00 GiB | 1.62 GiB | 6.87 GiB |
| AnyBCQ 3-bit | 67.03 ms | 2371.39 ms | 55.11 tok/s | 53.98 tok/s | 7.00 GiB | 1.63 GiB | 6.87 GiB |
| AnyBCQ 2-bit | 58.28 ms | 2364.40 ms | 55.07 tok/s | 54.14 tok/s | 7.00 GiB | 1.61 GiB | 6.87 GiB |

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

AnyBCQ 4-bit provides the strongest overall trade-off. It substantially reduces VRAM and checkpoint size while improving decode speed and total latency, with a moderate MMLU accuracy drop. GSM8K shows a larger quality loss.

AnyBCQ 3-bit produces a larger reasoning-quality degradation, especially on GSM8K.

AnyBCQ 2-bit shows severe quality degradation on both MMLU and GSM8K despite retaining the memory and speed advantages.

## Note on WikiText-2

The AnyBCQ WikiText-2 perplexity was obtained through the official AnyBCQ evaluator, while the baseline WikiText result was obtained through the baseline lm-evaluation-harness setup. The values should therefore be interpreted cautiously until the exact perplexity calculation protocol is verified to be identical.
