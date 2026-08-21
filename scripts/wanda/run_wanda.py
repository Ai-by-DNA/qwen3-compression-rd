import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
#from llmcompressor.modifiers.trandform.wanda import WandaModifier
from llmcompressor.modifiers.pruning import WandaPruningModifier

model_path = "/workspace/models/Qwen3-8B"
save_path = "/workspace/models/Qwen3-8B-WANDA"

print(f"Loading model from: {model_path}")

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

#pruning. 50% sparsing
# Counts sparsity only in Linear layers that wanda targets. Ignores lm_head layer. Mask structure is set to "0:0" for unstructured pruning.
recipe = [
    WandaPruningModifier(
        sparsity=0.5,
        mask_structure= "0:0",
        targets=["Linear"],
        ignore=["re:.*lm_head"],
    )
]

print("Wandaa running . . . Calibration and pruning . . .")

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    recipe=recipe,
    num_calibration_samples=128,
    max_seq_length=512,
)


print(f"Saving pruned model to: {save_path}")

os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

#test section
total_weights = 0
zero_weights = 0

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
        weights = module.weight

        total_weights += weights.numel()
        zero_weights += (weights == 0).sum().item()

actual_sparsity = zero_weights / total_weights

print(f"Linear weights: {total_weights:,}")
print(f"Zero Linear weights: {zero_weights:,}")
print(f"Linear sparsity: {actual_sparsity:.4%}")

print("Wanda pruning finished successfully.")