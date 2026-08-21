# TurboQuant Qwen3-8B KPI Report

## Configuration

- Model: Qwen3-8B
- Model precision: BF16
- KV-cache compression: TurboQuant 3-bit
- Quantizer seed: 42
- Thinking mode: disabled
- Batch size: 1
- GPU: NVIDIA GeForce RTX 4090
- Implementation: `hackimov/turboquant-kv`
- Model weights remain uncompressed.

## Reproducibility

TurboQuant uses randomized quantizer initialization. The original experiment did not explicitly pass the experiment seed to `TurboQuantProd`.

The final configuration therefore creates the quantizer explicitly with `seed=42`. Repeated repository-native tests with this configuration produced reproducible outputs.

## Performance

Clean-GPU measurement using 2 warm-up runs and 5 measured runs:

| Metric | TurboQuant 3-bit |
|---|---:|
| TTFT | 41.80 ms |
| Total latency | 4867.79 ms |
| Decode throughput | 26.32 tok/s |
| End-to-end throughput | 26.30 tok/s |
| Peak VRAM | 15.40 GiB |
| Peak RAM | 1.77 GiB |
| Model size | 15.27 GiB |

The final performance run was executed with no competing GPU process.

## Quality screening

A fixed-seed quality preflight was performed before considering another full quality evaluation.

Results:

- MMLU abstract algebra: 20% on 5 samples
- GSM8K: 0% on 5 samples
- WikiText word perplexity: 29281.02 on a 10-sample screening run

The MMLU and GSM8K sample counts are too small to report as final benchmark accuracies. However, together with the extremely high WikiText perplexity, the screening indicates severe quality degradation for this specific 3-bit configuration.

A second full quality run was therefore not performed in order to avoid unnecessary GPU cost.

Earlier full quality results produced before explicitly fixing the TurboQuant quantizer seed are not treated as final reproducible benchmark results.

## Long-context memory

The long-context experiment compared the standard Hugging Face BF16 KV cache with TurboQuant 3-bit on an otherwise idle RTX 4090.

| Context | Plain resident KV | TurboQuant resident KV | Result |
|---|---:|---:|---|
| 2048 | 0.289 GiB | 1.411 GiB | Both OK |
| 4096 | 0.563 GiB | 2.821 GiB | Both OK |
| 8192 | 1.125 GiB | OOM | Plain only |

At 2048 tokens, the reported resident-memory reduction was -387.80%.

At 4096 tokens, the reported resident-memory reduction was -401.56%.

Negative reduction means that this implementation consumed more resident KV-cache memory than the plain BF16 cache.

At 8192 tokens, the standard Hugging Face cache completed successfully while TurboQuant reached the RTX 4090 memory limit.

## Implementation finding

Inspection of the compressed cache representation showed substantial metadata overhead. In particular, large `k_idx` and `v_idx` tensors are stored as `int64`.

For this Qwen3-8B experiment, the resulting live cache representation was therefore substantially larger than the plain BF16 KV cache despite the nominal 3-bit quantization.

## Conclusion

For Qwen3-8B on the tested RTX 4090, this specific third-party TurboQuant KV implementation did not provide the expected memory advantage.

The final fixed-seed experiment showed:

- reproducible quantizer initialization with seed 42;
- significant quality degradation in the fixed-seed screening;
- no model-weight size reduction, as expected for KV-only compression;
- larger live KV-cache memory than plain BF16 at 2048 and 4096 tokens;
- TurboQuant OOM at 8192 tokens while the plain cache completed successfully.

These findings apply to the tested implementation and configuration and should not be generalized to the TurboQuant method independently of its implementation.
