import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_PATH = "/workspace/models/Qwen3-8B"
SAVE_PATH = "/workspace/models/Qwen3-8B-AWQ"

print(f"Loading model from: {MODEL_PATH}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    trust_remote_code=True,
)

# AWQ + asymmetric INT4 weight-only quantization
recipe = [
    AWQModifier(duo_scaling="both"),
    QuantizationModifier(
        targets=["Linear"],
        scheme="W4A16_ASYM",
        ignore=["lm_head"],
    ),
]

print("Running AWQ calibration / quantization...")

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    recipe=recipe,
    num_calibration_samples=128,
    max_seq_length=512,
)

print(f"Saving quantized model to: {SAVE_PATH}")

os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("AWQ quantization finished successfully.")