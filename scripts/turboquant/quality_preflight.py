import torch

from lm_eval import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantModel

from evaluate_quality import TurboQuantHFLM


MODEL_PATH = "/workspace/models/Qwen3-8B"
SEED = 42
BITS = 3.0


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda").eval()

    head_dim = int(
        getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        )
    )

    wrapper = TurboQuantModel(
        model,
        bits=BITS,
        head_dim=head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    wrapper.enable_decoder_fused_attention(architecture="auto")

    lm = TurboQuantHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=1,
        max_length=2048,
        device="cuda",
        enable_thinking=False,
        tq_wrapper=wrapper,
    )

    print("=== MMLU preflight ===")
    evaluator.simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=5,
        batch_size=1,
        limit=1,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
    )

    print("=== GSM8K preflight ===")
    evaluator.simple_evaluate(
        model=lm,
        tasks=["gsm8k"],
        num_fewshot=5,
        batch_size=1,
        limit=1,
        apply_chat_template=True,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
    )

    wrapper.disable_decoder_fused_attention()

    print()
    print("==========================================")
    print("TurboQuant lm-eval quality preflight PASSED")
    print("==========================================")


if __name__ == "__main__":
    main()
