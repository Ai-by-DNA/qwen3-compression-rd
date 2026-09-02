# Wanda Sparse Kernel Experiment

## Setup

- Model family: Qwen3-8B
- Pruning: Wanda
- Sparsity: 50% unstructured
- GPU: NVIDIA GeForce RTX 4090
- PyTorch: 2.13.0+cu130
- CUDA: 13.0
- Sparse format for kernel test: PyTorch CSR
- Test matrix shape: 4096 x 4096
- Weight dtype: FP16

## Results

| Tokens | Dense latency (ms) | CSR sparse latency (ms) | Dense/Sparse ratio |
|---:|---:|---:|---:|
| 1 | 0.0359 | 0.0911 | 0.394x |
| 16 | 0.0188 | 0.4414 | 0.043x |
| 128 | 0.0319 | 0.6766 | 0.047x |

Storage for the test matrix:

- Dense: 32.00 MiB
- CSR: 80.02 MiB

## Conclusion

The generic PyTorch CSR sparse path was functional on the RTX 4090, but it was substantially slower than dense execution and also introduced significant indexing/storage overhead.

For this reason, the CSR sparse kernel path was not selected for the final compression stack.

The bitmap sparse representation is evaluated separately as a storage optimization.
