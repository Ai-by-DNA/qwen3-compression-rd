import time
import torch

DEVICE = "cuda"
DTYPE = torch.float16

# Representative Qwen3/Wanda Linear size
OUT_FEATURES = 4096
IN_FEATURES = 4096

# Simulates token batches during inference
BATCH_TOKENS = [1, 16, 128]

WARMUP = 20
ITERS = 100

print("=" * 70)
print("WANDA SPARSE KERNEL PREFLIGHT")
print("=" * 70)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

# ---------------------------------------------------------
# Create a 50%-sparse weight matrix
# ---------------------------------------------------------

torch.manual_seed(42)

W = torch.randn(
    OUT_FEATURES,
    IN_FEATURES,
    device=DEVICE,
    dtype=DTYPE
)

mask = torch.rand(
    OUT_FEATURES,
    IN_FEATURES,
    device=DEVICE
) < 0.5

W[mask] = 0

actual_sparsity = (W == 0).float().mean().item()

print(f"\nWeight shape: {tuple(W.shape)}")
print(f"Sparsity: {actual_sparsity * 100:.2f}%")

# PyTorch sparse CSR
W_sparse = W.to_sparse_csr()

dense_bytes = W.numel() * W.element_size()

sparse_bytes = (
    W_sparse.values().numel() * W_sparse.values().element_size()
    + W_sparse.col_indices().numel() * W_sparse.col_indices().element_size()
    + W_sparse.crow_indices().numel() * W_sparse.crow_indices().element_size()
)

print(f"Dense tensor: {dense_bytes / 1024**2:.2f} MiB")
print(f"CSR tensor:   {sparse_bytes / 1024**2:.2f} MiB")

# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

def bench(fn):
    for _ in range(WARMUP):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(ITERS):
        fn()

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / ITERS


print("\n" + "=" * 70)
print("KERNEL TEST")
print("=" * 70)

for tokens in BATCH_TOKENS:

    X = torch.randn(
        IN_FEATURES,
        tokens,
        device=DEVICE,
        dtype=DTYPE
    )

    # Dense
    Y_dense = W @ X

    # Sparse CSR
    try:
        Y_sparse = torch.sparse.mm(W_sparse, X)
    except Exception as e:
        print(f"\nTokens={tokens}")
        print("SPARSE KERNEL FAILED:")
        print(type(e).__name__, e)
        continue

    max_diff = (Y_dense - Y_sparse).abs().max().item()
    mean_diff = (Y_dense - Y_sparse).abs().mean().item()

    dense_ms = bench(lambda: W @ X)
    sparse_ms = bench(lambda: torch.sparse.mm(W_sparse, X))

    speedup = dense_ms / sparse_ms

    print(f"\nTokens: {tokens}")
    print(f"  max difference : {max_diff}")
    print(f"  mean difference: {mean_diff}")
    print(f"  dense latency  : {dense_ms:.4f} ms")
    print(f"  sparse latency : {sparse_ms:.4f} ms")
    print(f"  speedup        : {speedup:.3f}x")

print("\n" + "=" * 70)
print("PREFLIGHT COMPLETE")
print("=" * 70)
