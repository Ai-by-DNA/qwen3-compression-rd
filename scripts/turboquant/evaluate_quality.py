import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from turboquant import TurboQuantModel


MODEL_PATH = "/workspace/models/Qwen3-8B"
RESULT_DIR = Path("/workspace/results/turboquant")

BITS = 3.0
MAX_LENGTH = 2048
SEED = 42


class TurboQuantHFLM(HFLM):
    def __init__(self, *args, tq_wrapper, **kwargs):
        self.tq_wrapper = tq_wrapper
        super().__init__(*args, **kwargs)

    def _new_cache(self):
        return self.tq_wrapper.make_dynamic_cache(
            triton_fused_layers=True,
            strict_reencode=False,
            hybrid_float_cache=False,
        )

    def _model_call(self, inps, attn_mask=None, labels=None):
        if attn_mask is not None or labels is not None:
            return super()._model_call(
                inps,
                attn_mask=attn_mask,
                labels=labels,
            )

        cache = self._new_cache()

        with torch.inference_mode():
            return self.model(
                input_ids=inps,
                past_key_values=cache,
                use_cache=True,
            ).logits

    def _model_generate(
        self,
        context,
        max_length,
        stop,
        **generation_kwargs,
    ):
        generation_kwargs.pop("past_key_values", None)

        return super()._model_generate(
            context,
            max_length,
            stop,
            past_key_values=self._new_cache(),
            **generation_kwargs,
        )


def run_task(lm, task, num_fewshot, apply_chat_template=False):
    return evaluator.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=num_fewshot,
        batch_size=1,
        apply_chat_template=apply_chat_template,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
        log_samples=False,
    )


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)

    print("Loading Qwen3-8B...")

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

    wrapper.enable_decoder_fused_attention(
        architecture="auto",
    )

    lm = TurboQuantHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=1,
        max_length=MAX_LENGTH,
        device="cuda",
        enable_thinking=False,
        tq_wrapper=wrapper,
    )

    print("\n===== WikiText-2 =====")
    wikitext = run_task(
        lm,
        task="wikitext",
        num_fewshot=0,
    )

    print("\n===== MMLU =====")
    mmlu = run_task(
        lm,
        task="mmlu",
        num_fewshot=5,
    )

    print("\n===== GSM8K =====")
    gsm8k = run_task(
        lm,
        task="gsm8k",
        num_fewshot=5,
        apply_chat_template=True,
    )

    results = {
        "configuration": {
            "model": "Qwen3-8B",
            "turboquant_bits": BITS,
            "kv_cache": "TurboQuant",
            "fused_attention": True,
            "dtype": "bfloat16",
            "thinking_mode": False,
            "batch_size": 1,
            "seed": SEED,
            "max_length": MAX_LENGTH,
        },
        "wikitext": wikitext["results"],
        "mmlu": mmlu["results"],
        "gsm8k": gsm8k["results"],
    }

    output = RESULT_DIR / "quality_results.json"

    with output.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    wrapper.disable_decoder_fused_attention()

    print()
    print("==========================================")
    print("TurboQuant quality evaluation completed.")
    print(f"Results: {output}")
    print("==========================================")


if __name__ == "__main__":
    main()
