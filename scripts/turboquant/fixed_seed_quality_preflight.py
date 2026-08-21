import os
import sys
import torch

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

sys.path.insert(0, "/workspace/repo-turboquant")

from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from turboquant import TurboQuantModel, TurboQuantProd
from scripts.turboquant.evaluate_quality import TurboQuantHFLM

MODEL = "/workspace/models/Qwen3-8B"
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("Loading Qwen3-8B...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    local_files_only=True,
).cuda().eval()

head_dim = getattr(
    model.config,
    "head_dim",
    model.config.hidden_size // model.config.num_attention_heads,
)

# Explicit reproducible TurboQuant quantizer
quantizer = TurboQuantProd(
    bits=3.0,
    head_dim=head_dim,
    device="cuda",
    seed=SEED,
    dtype=torch.bfloat16,
)

wrapper = TurboQuantModel(
    model,
    quantizer=quantizer,
)

wrapper.enable_decoder_fused_attention(
    architecture="auto"
)

lm = TurboQuantHFLM(
    tq_wrapper=wrapper,
    pretrained=model,
    tokenizer=tokenizer,
    backend="causal",
    batch_size=1,
    max_length=2048,
    device="cuda",
    enable_thinking=False,
)


def run(task, fewshot, limit, chat=False):
    print(f"\n===== {task} | limit={limit} =====")

    result = evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=fewshot,
        batch_size=1,
        limit=limit,
        apply_chat_template=chat,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
        log_samples=False,
    )

    task_result = result["results"][task]

    for key, value in task_result.items():
        if isinstance(value, (int, float, str)):
            print(f"{key}: {value}")


# Small screening only
run(
    task="mmlu_abstract_algebra",
    fewshot=5,
    limit=5,
)

run(
    task="gsm8k",
    fewshot=5,
    limit=5,
    chat=True,
)

run(
    task="wikitext",
    fewshot=0,
    limit=10,
)

wrapper.disable_decoder_fused_attention()

print("\n================================")
print("Fixed-seed quality preflight finished.")
print("TurboQuant bits: 3")
print("TurboQuant seed: 42")
print("================================")
