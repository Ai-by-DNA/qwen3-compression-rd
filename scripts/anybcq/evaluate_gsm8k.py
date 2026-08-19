import argparse
import json
import os

import lm_eval
from lm_eval.models.huggingface import HFLM

from anybcq.evaluate import eval as anybcq_eval


MODEL_PATH = "/workspace/models/anybcq-qwen3-8b"
RESULT_ROOT = "/workspace/results/anybcq"

SEED = 42
NUM_FEWSHOT = 5
BATCH_SIZE = 1


# ============================================================
# lm-eval wrapper for Qwen3 with thinking OFF
# ============================================================

class Qwen3NoThinkingHFLM(HFLM):
    def apply_chat_template(self, chat_history):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


# ============================================================
# Select 2 / 3 / 4-bit from command line
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--bitwidth",
    type=int,
    required=True,
    choices=[2, 3, 4],
    help="AnyBCQ precision: 2, 3, or 4 bits.",
)

args = parser.parse_args()
BITWIDTH = args.bitwidth


# ============================================================
# Load quantized AnyBCQ model
# ============================================================

print("==========================================")
print(f"AnyBCQ Qwen3-8B GSM8K - {BITWIDTH}-bit")
print("==========================================")

_, tokenizer, model = anybcq_eval.auto_model_load(
    MODEL_PATH,
    is_fp16=False,
    device="cuda",
)

if BITWIDTH not in model.precisions:
    raise ValueError(
        f"{BITWIDTH}-bit is not available. "
        f"Supported precisions: {model.precisions}"
    )

model.set_precision(BITWIDTH)

print(f"Supported precisions: {model.precisions}")
print(f"Selected precision: {BITWIDTH}-bit")


# ============================================================
# Connect AnyBCQ model to lm-eval
# ============================================================

model_lm = Qwen3NoThinkingHFLM(
    pretrained=model,
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
)


# ============================================================
# GSM8K
# Same experimental protocol as baseline
# ============================================================

results = lm_eval.simple_evaluate(
    model=model_lm,
    tasks=["gsm8k"],
    num_fewshot=NUM_FEWSHOT,
    batch_size=BATCH_SIZE,

    apply_chat_template=True,
    fewshot_as_multiturn=False,

    random_seed=SEED,
    numpy_random_seed=SEED,
    torch_random_seed=SEED,
    fewshot_random_seed=SEED,
)


# ============================================================
# Save results separately for each bit-width
# ============================================================

result_dir = os.path.join(
    RESULT_ROOT,
    f"{BITWIDTH}bit",
)

os.makedirs(
    result_dir,
    exist_ok=True,
)

result_path = os.path.join(
    result_dir,
    "gsm8k.json",
)

with open(
    result_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=2,
        default=str,
    )


# ============================================================
# Print GSM8K accuracy
# ============================================================

gsm8k_results = results["results"]["gsm8k"]

print()
print("==========================================")
print(f"GSM8K RESULTS - AnyBCQ {BITWIDTH}-bit")
print("==========================================")

for metric, value in gsm8k_results.items():
    if "exact_match" in metric:
        print(f"{metric}: {value}")

print()
print(f"Results saved to: {result_path}")
