import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "/workspace/models/Qwen3-8B-WANDA"


def mib(nbytes):
    return nbytes / (1024 ** 2)


print(f"Loading model from {MODEL_PATH}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cpu",
)

total_dense_bytes = 0
total_coo_bytes = 0
total_csr_bytes = 0

total_elements = 0
total_nonzero = 0

print()
print(
    f"{'Layer':65} "
    f"{'Sparsity':>10} "
    f"{'Dense MiB':>12} "
    f"{'COO MiB':>12} "
    f"{'CSR MiB':>12}"
)
print("-" * 120)

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

    # Dense representation:
    # every BF16 value is stored.
    dense_bytes = numel * weight.element_size()

    # COO representation:
    # values + 2 coordinates for every non-zero weight.
    #
    # PyTorch sparse COO indices use int64:
    #   value: 2 bytes (BF16)
    #   row index: 8 bytes
    #   column index: 8 bytes
    coo_bytes = (
        nnz * weight.element_size()
        + nnz * 2 * torch.tensor([], dtype=torch.int64).element_size()
    )

    # CSR representation:
    # values + column indices + row pointer.
    #
    # Here we estimate int64 indices, matching standard PyTorch CSR.
    rows = weight.shape[0]

    csr_bytes = (
        nnz * weight.element_size()
        + nnz * torch.tensor([], dtype=torch.int64).element_size()
        + (rows + 1) * torch.tensor([], dtype=torch.int64).element_size()
    )

    total_elements += numel
    total_nonzero += nnz

    total_dense_bytes += dense_bytes
    total_coo_bytes += coo_bytes
    total_csr_bytes += csr_bytes

    print(
        f"{name[:65]:65} "
        f"{sparsity:9.2%} "
        f"{mib(dense_bytes):12.2f} "
        f"{mib(coo_bytes):12.2f} "
        f"{mib(csr_bytes):12.2f}"
    )


overall_sparsity = 1 - (total_nonzero / total_elements)

coo_saving = 1 - (total_coo_bytes / total_dense_bytes)
csr_saving = 1 - (total_csr_bytes / total_dense_bytes)

print()
print("=" * 80)
print("TOTAL LINEAR WEIGHT STORAGE ANALYSIS")
print("=" * 80)

print(f"Elements:          {total_elements:,}")
print(f"Non-zero weights:  {total_nonzero:,}")
print(f"Overall sparsity:   {overall_sparsity:.4%}")

print()
print(f"Dense BF16:         {mib(total_dense_bytes):,.2f} MiB")
print(f"COO BF16/int64:     {mib(total_coo_bytes):,.2f} MiB")
print(f"CSR BF16/int64:     {mib(total_csr_bytes):,.2f} MiB")

print()
print(f"COO storage saving: {coo_saving:.2%}")
print(f"CSR storage saving: {csr_saving:.2%}")
