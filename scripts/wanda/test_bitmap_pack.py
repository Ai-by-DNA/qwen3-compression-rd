import os
import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = "/workspace/models/Qwen3-8B-WANDA"
OUTPUT_PATH = "/workspace/results/wanda/bitmap_layer_test.pt"

TARGET_LAYER = "model.layers.0.mlp.down_proj"


def mib(nbytes):
    return nbytes / (1024 ** 2)


def pack_bits(mask):
    """
    Pack a boolean mask so that 8 boolean values use 1 byte.
    """

    mask = mask.flatten().to(torch.uint8)

    original_length = mask.numel()

    # Pad to a multiple of 8.
    padding = (-original_length) % 8

    if padding:
        mask = torch.cat(
            [
                mask,
                torch.zeros(padding, dtype=torch.uint8),
            ]
        )

    mask = mask.reshape(-1, 8)

    bit_weights = (
        2 ** torch.arange(8, dtype=torch.uint8)
    )

    packed = torch.sum(
        mask * bit_weights,
        dim=1,
    ).to(torch.uint8)

    return packed, original_length


def unpack_bits(packed, original_length):
    """
    Convert the packed bitmap back to a boolean mask.
    """

    shifts = torch.arange(8, dtype=torch.uint8)

    bits = (
        (packed.unsqueeze(1) >> shifts) & 1
    )

    bits = bits.reshape(-1)

    return bits[:original_length].bool()


print(f"Loading model from: {MODEL_PATH}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cpu",
)


modules = dict(model.named_modules())

if TARGET_LAYER not in modules:
    raise RuntimeError(
        f"Layer not found: {TARGET_LAYER}"
    )

module = modules[TARGET_LAYER]

if not isinstance(module, torch.nn.Linear):
    raise RuntimeError(
        f"{TARGET_LAYER} is not a Linear layer."
    )


weight = module.weight.detach().cpu()

shape = weight.shape
numel = weight.numel()

mask = weight != 0
values = weight[mask].clone()

nnz = values.numel()
sparsity = 1 - (nnz / numel)


print()
print("ORIGINAL")
print("=" * 60)

print(f"Layer:       {TARGET_LAYER}")
print(f"Shape:       {tuple(shape)}")
print(f"Elements:    {numel:,}")
print(f"Non-zero:    {nnz:,}")
print(f"Sparsity:    {sparsity:.4%}")
print(f"Dtype:       {weight.dtype}")


print()
print("PACKING BITMAP")
print("=" * 60)

packed_bitmap, original_length = pack_bits(mask)

dense_bytes = (
    weight.numel()
    * weight.element_size()
)

values_bytes = (
    values.numel()
    * values.element_size()
)

bitmap_bytes = (
    packed_bitmap.numel()
    * packed_bitmap.element_size()
)

packed_tensor_bytes = (
    values_bytes
    + bitmap_bytes
)

print(
    f"Dense tensor:       "
    f"{mib(dense_bytes):.2f} MiB"
)

print(
    f"Non-zero values:    "
    f"{mib(values_bytes):.2f} MiB"
)

print(
    f"Packed bitmap:      "
    f"{mib(bitmap_bytes):.2f} MiB"
)

print(
    f"Packed tensors:     "
    f"{mib(packed_tensor_bytes):.2f} MiB"
)

print(
    f"Theoretical saving: "
    f"{1 - packed_tensor_bytes / dense_bytes:.2%}"
)


print()
print("SAVING")
print("=" * 60)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

torch.save(
    {
        "values": values,
        "bitmap": packed_bitmap,
        "shape": tuple(shape),
        "numel": numel,
        "dtype": str(weight.dtype),
        "layer": TARGET_LAYER,
    },
    OUTPUT_PATH,
)

actual_file_bytes = os.path.getsize(OUTPUT_PATH)

print(f"Saved to: {OUTPUT_PATH}")

print(
    f"Actual file size:   "
    f"{mib(actual_file_bytes):.2f} MiB"
)

print(
    f"Actual disk saving: "
    f"{1 - actual_file_bytes / dense_bytes:.2%}"
)


print()
print("LOADING + RECONSTRUCTING")
print("=" * 60)

saved = torch.load(
    OUTPUT_PATH,
    map_location="cpu",
    weights_only=False,
)

loaded_values = saved["values"]
loaded_bitmap = saved["bitmap"]

reconstructed_mask = unpack_bits(
    loaded_bitmap,
    saved["numel"],
)

reconstructed = torch.zeros(
    saved["numel"],
    dtype=weight.dtype,
)

reconstructed[reconstructed_mask] = (
    loaded_values
)

reconstructed = reconstructed.reshape(
    saved["shape"]
)


exact_match = torch.equal(
    weight,
    reconstructed,
)

max_difference = (
    weight.float()
    - reconstructed.float()
).abs().max().item()


print(f"Exact match:        {exact_match}")
print(f"Max difference:     {max_difference}")


if not exact_match:
    raise RuntimeError(
        "Reconstructed tensor does not exactly match "
        "the original Wanda tensor."
    )


print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print(
    "Bitmap pack -> save -> load -> unpack "
    "reconstructed the Wanda weight exactly."
)
