import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = "/workspace/models/Qwen3-8B-WANDA"


def mib(nbytes):
    return nbytes / (1024 ** 2)


print(f"Loading model from {MODEL_PATH}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cpu",
)

total_dense_bytes = 0
total_values_bytes = 0
total_bitmap_bytes = 0

total_elements = 0
total_nonzero = 0


print()
print(
    f"{'Layer':65} "
    f"{'Sparsity':>10} "
    f"{'Dense MiB':>12} "
    f"{'Values MiB':>12} "
    f"{'Bitmap MiB':>12} "
    f"{'Packed MiB':>12} "
    f"{'Saving':>10}"
)
print("-" * 145)


for name, module in model.named_modules():

    if not isinstance(module, torch.nn.Linear):
        continue

    if "lm_head" in name:
        continue

    weight = module.weight.detach().cpu()

    numel = weight.numel()
    nnz = torch.count_nonzero(weight).item()
    zeros = numel - nnz

    sparsity = zeros / numel

    # Normal dense BF16 representation:
    # every weight occupies 2 bytes.
    dense_bytes = numel * weight.element_size()

    # Sparse values:
    # store only the non-zero BF16 weights.
    values_bytes = nnz * weight.element_size()

    # Bitmap:
    # one bit for every original weight.
    # 8 weights are represented by one byte.
    bitmap_bytes = (numel + 7) // 8

    packed_bytes = values_bytes + bitmap_bytes

    saving = 1 - (packed_bytes / dense_bytes)

    total_elements += numel
    total_nonzero += nnz

    total_dense_bytes += dense_bytes
    total_values_bytes += values_bytes
    total_bitmap_bytes += bitmap_bytes

    print(
        f"{name[:65]:65} "
        f"{sparsity:9.2%} "
        f"{mib(dense_bytes):12.2f} "
        f"{mib(values_bytes):12.2f} "
        f"{mib(bitmap_bytes):12.2f} "
        f"{mib(packed_bytes):12.2f} "
        f"{saving:9.2%}"
    )


total_packed_bytes = (
    total_values_bytes
    + total_bitmap_bytes
)

overall_sparsity = 1 - (
    total_nonzero / total_elements
)

overall_saving = 1 - (
    total_packed_bytes / total_dense_bytes
)


print()
print("=" * 80)
print("BITMAP + NON-ZERO VALUES STORAGE ANALYSIS")
print("=" * 80)

print(f"Elements:              {total_elements:,}")
print(f"Non-zero weights:      {total_nonzero:,}")
print(f"Overall sparsity:       {overall_sparsity:.4%}")

print()
print(f"Dense BF16:             {mib(total_dense_bytes):,.2f} MiB")
print(f"Non-zero BF16 values:   {mib(total_values_bytes):,.2f} MiB")
print(f"Packed bitmap:          {mib(total_bitmap_bytes):,.2f} MiB")
print(f"Total packed storage:   {mib(total_packed_bytes):,.2f} MiB")

print()
print(f"Storage saving:         {overall_saving:.2%}")
print(
    f"Compression ratio:      "
    f"{total_dense_bytes / total_packed_bytes:.3f}x"
)
