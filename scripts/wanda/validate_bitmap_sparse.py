import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = "/workspace/models/Qwen3-8B-WANDA"
SPARSE_PATH = Path("/workspace/models/Qwen3-8B-WANDA-bitmap")
METADATA_PATH = SPARSE_PATH / "metadata.json"


def unpack_bits(packed, original_length):
    shifts = torch.arange(8, dtype=torch.uint8)

    bits = (
        (packed.unsqueeze(1) >> shifts) & 1
    ).reshape(-1)

    return bits[:original_length].bool()


print(f"Loading original Wanda model from: {MODEL_PATH}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cpu",
)

model.eval()

modules = dict(model.named_modules())

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

layers = metadata["layers"]

print(f"Metadata layers: {len(layers)}")

checked = 0
failed = 0
max_difference = 0.0

# Cache only one shard at a time.
current_shard_number = None
current_shard = None


print()
print("Validating sparse model...")
print("=" * 70)


for layer_key, info in layers.items():

    shard_number = info["shard"]

    if shard_number != current_shard_number:

        shard_path = (
            SPARSE_PATH
            / f"sparse_weights_{shard_number:03d}.pt"
        )

        print(f"Loading {shard_path.name}...")

        current_shard = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=False,
        )

        current_shard_number = shard_number

    values = current_shard[f"{layer_key}.values"]
    bitmap = current_shard[f"{layer_key}.bitmap"]

    module_name = info["module_name"]
    shape = tuple(info["shape"])
    numel = info["numel"]

    mask = unpack_bits(
        bitmap,
        numel,
    )

    reconstructed = torch.zeros(
        numel,
        dtype=values.dtype,
    )

    reconstructed[mask] = values
    reconstructed = reconstructed.reshape(shape)

    original_module = modules[module_name]
    original = original_module.weight.detach().cpu()

    exact_match = torch.equal(
        reconstructed,
        original,
    )

    difference = (
        reconstructed.float()
        - original.float()
    ).abs().max().item()

    max_difference = max(
        max_difference,
        difference,
    )

    checked += 1

    if not exact_match:
        failed += 1

        print(
            f"FAIL: {module_name} "
            f"max_diff={difference}"
        )


print()
print("=" * 70)
print("FULL MODEL VALIDATION SUMMARY")
print("=" * 70)

print(f"Expected layers:      {len(layers)}")
print(f"Checked layers:       {checked}")
print(f"Failed layers:        {failed}")
print(f"Max difference:       {max_difference}")

if (
    failed == 0
    and checked == len(layers)
    and max_difference == 0.0
):
    print()
    print("SUCCESS")
    print(
        "All bitmap sparse weights reconstruct "
        "exactly to the original Wanda weights."
    )
else:
    print()
    print("VALIDATION FAILED")
