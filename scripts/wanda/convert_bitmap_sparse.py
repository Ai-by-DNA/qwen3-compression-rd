import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = "/workspace/models/Qwen3-8B-WANDA"
OUTPUT_PATH = "/workspace/models/Qwen3-8B-WANDA-bitmap"

# Keep each output shard reasonably small.
MAX_SHARD_BYTES = 256 * 1024 ** 2  # 256 MiB

def mib(nbytes):
    return nbytes / (1024 ** 2)


def gib(nbytes):
    return nbytes / (1024 ** 3)


def pack_bits(mask):
    """Pack 8 boolean mask entries into one uint8 byte."""

    mask = mask.flatten().to(torch.uint8)
    original_length = mask.numel()

    padding = (-original_length) % 8

    if padding:
        mask = torch.cat(
            [
                mask,
                torch.zeros(padding, dtype=torch.uint8),
            ]
        )

    mask = mask.reshape(-1, 8)

    bit_weights = 2 ** torch.arange(
        8,
        dtype=torch.uint8,
    )

    packed = torch.sum(
        mask * bit_weights,
        dim=1,
    ).to(torch.uint8)

    return packed, original_length


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


print(f"Loading Wanda model from: {MODEL_PATH}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cpu",
)

os.makedirs(OUTPUT_PATH, exist_ok=True)


metadata = {
    "format": "wanda_unstructured_bitmap_v1",
    "source_model": MODEL_PATH,
    "value_dtype": "bfloat16",
    "bitmap_dtype": "uint8",
    "bit_order": "little_endian_within_byte",
    "layers": {},
}


current_shard = {}
current_shard_bytes = 0
shard_number = 1

total_dense_bytes = 0
total_sparse_bytes = 0
total_elements = 0
total_nonzero = 0
converted_layers = 0


def save_shard():
    global current_shard
    global current_shard_bytes
    global shard_number

    if not current_shard:
        return

    filename = f"sparse_weights_{shard_number:03d}.pt"
    path = os.path.join(OUTPUT_PATH, filename)

    print(
        f"Saving {filename} "
        f"({mib(current_shard_bytes):.2f} MiB tensors)..."
    )

    torch.save(
        current_shard,
        path,
        _use_new_zipfile_serialization=False,
    )

    current_shard = {}
    current_shard_bytes = 0
    shard_number += 1


print()
print("Converting Linear weights...")
print("=" * 80)


for name, module in model.named_modules():

    if not isinstance(module, torch.nn.Linear):
        continue

    # Wanda did not prune lm_head, so keep it out of
    # this sparse experiment.
    if "lm_head" in name:
        continue

    weight = module.weight.detach().cpu()

    numel = weight.numel()
    mask = weight != 0

    values = weight[mask].clone()
    nnz = values.numel()

    packed_bitmap, original_length = pack_bits(mask)

    dense_bytes = tensor_bytes(weight)
    values_bytes = tensor_bytes(values)
    bitmap_bytes = tensor_bytes(packed_bitmap)
    sparse_bytes = values_bytes + bitmap_bytes

    sparsity = 1 - (nnz / numel)

    # If adding this layer would make the current shard
    # exceed the target size, save the existing shard first.
    if (
        current_shard
        and current_shard_bytes + sparse_bytes > MAX_SHARD_BYTES
    ):
        save_shard()

    layer_key = f"layer_{converted_layers:03d}"

    current_shard[f"{layer_key}.values"] = values
    current_shard[f"{layer_key}.bitmap"] = packed_bitmap

    metadata["layers"][layer_key] = {
        "module_name": name,
        "shape": list(weight.shape),
        "numel": numel,
        "nnz": nnz,
        "sparsity": sparsity,
        "original_length": original_length,
        "dense_bytes": dense_bytes,
        "values_bytes": values_bytes,
        "bitmap_bytes": bitmap_bytes,
        "sparse_bytes": sparse_bytes,
        "shard": shard_number,
    }

    current_shard_bytes += sparse_bytes

    total_dense_bytes += dense_bytes
    total_sparse_bytes += sparse_bytes
    total_elements += numel
    total_nonzero += nnz
    converted_layers += 1

    print(
        f"[{converted_layers:03d}] "
        f"{name:55} "
        f"sparsity={sparsity:6.2%} "
        f"dense={mib(dense_bytes):8.2f} MiB "
        f"sparse={mib(sparse_bytes):8.2f} MiB"
    )


save_shard()


overall_sparsity = 1 - (
    total_nonzero / total_elements
)

storage_saving = 1 - (
    total_sparse_bytes / total_dense_bytes
)

compression_ratio = (
    total_dense_bytes / total_sparse_bytes
)


metadata["summary"] = {
    "converted_layers": converted_layers,
    "total_elements": total_elements,
    "total_nonzero": total_nonzero,
    "overall_sparsity": overall_sparsity,
    "dense_bytes": total_dense_bytes,
    "sparse_tensor_bytes": total_sparse_bytes,
    "storage_saving": storage_saving,
    "compression_ratio": compression_ratio,
}


metadata_path = os.path.join(
    OUTPUT_PATH,
    "metadata.json",
)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)


# Measure actual .pt files written to disk.
actual_sparse_file_bytes = sum(
    path.stat().st_size
    for path in Path(OUTPUT_PATH).glob("sparse_weights_*.pt")
)

actual_disk_saving = 1 - (
    actual_sparse_file_bytes / total_dense_bytes
)


print()
print("=" * 80)
print("FULL MODEL BITMAP CONVERSION SUMMARY")
print("=" * 80)

print(f"Converted Linear layers: {converted_layers}")
print(f"Total elements:          {total_elements:,}")
print(f"Total non-zero:          {total_nonzero:,}")
print(f"Overall sparsity:         {overall_sparsity:.4%}")

print()
print(f"Dense Linear weights:    {gib(total_dense_bytes):.3f} GiB")
print(f"Sparse tensor data:      {gib(total_sparse_bytes):.3f} GiB")
print(f"Actual sparse files:     {gib(actual_sparse_file_bytes):.3f} GiB")

print()
print(f"Theoretical saving:      {storage_saving:.2%}")
print(f"Actual disk saving:      {actual_disk_saving:.2%}")
print(f"Compression ratio:       {compression_ratio:.3f}x")

print()
print(f"Output: {OUTPUT_PATH}")
print(f"Metadata: {metadata_path}")
print()
print("Conversion completed successfully.")
