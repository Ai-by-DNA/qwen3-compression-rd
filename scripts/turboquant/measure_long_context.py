import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from turboquant import TurboQuantModel

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

MODEL_PATH = "/workspace/models/Qwen3-8B"
RESULT_DIR = Path("/workspace/results/turboquant")

BITS = 3.0
SEED = 42
CONTEXT_LENGTHS = [2048, 4096, 8192, 16384]
MAX_NEW_TOKENS = 32


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def make_long_input(tokenizer, target_length):
    text = (
        "Artificial intelligence systems process information using mathematical "
        "representations and computational models. "
    )

    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    repeats = (target_length // len(ids)) + 1
    ids = ids.repeat(repeats)[:target_length]

    return ids.unsqueeze(0).to("cuda")


def run_case(model, input_ids, cache):
    cleanup()
    torch.cuda.reset_peak_memory_stats()

    baseline_allocated = torch.cuda.memory_allocated()

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
            pad_token_id=0,
        )

    torch.cuda.synchronize()
    end = time.perf_counter()

    peak = torch.cuda.max_memory_allocated()

    return {
        "generated_tokens": int(output.shape[1] - input_ids.shape[1]),
        "latency_seconds": end - start,
        "peak_vram_gib": peak / (1024 ** 3),
        "incremental_peak_vram_gib":
            (peak - baseline_allocated) / (1024 ** 3),
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

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

    results = []

    for context_length in CONTEXT_LENGTHS:
        print(f"\n===== Context: {context_length} tokens =====")

        input_ids = make_long_input(tokenizer, context_length)

        print("Plain HF KV cache...")

        plain_result = run_case(
            model,
            input_ids,
            DynamicCache(),
        )

        cleanup()

        print("TurboQuant 3-bit KV cache...")

        wrapper.enable_decoder_fused_attention(
            architecture="auto",
        )

        tq_cache = wrapper.make_dynamic_cache(
            triton_fused_layers=True,
            strict_reencode=False,
            hybrid_float_cache=False,
        )

        tq_result = run_case(
            model,
            input_ids,
            tq_cache,
        )

        wrapper.disable_decoder_fused_attention()

        reduction = (
            1
            - tq_result["incremental_peak_vram_gib"]
            / plain_result["incremental_peak_vram_gib"]
        ) * 100

        row = {
            "context_tokens": context_length,
            "plain": plain_result,
            "turboquant_3bit": tq_result,
            "incremental_vram_reduction_percent": reduction,
        }

        results.append(row)

        print(
            f"Plain incremental VRAM: "
            f"{plain_result['incremental_peak_vram_gib']:.2f} GiB"
        )
        print(
            f"TurboQuant incremental VRAM: "
            f"{tq_result['incremental_peak_vram_gib']:.2f} GiB"
        )
        print(f"VRAM reduction: {reduction:.2f}%")

        del input_ids
        cleanup()

    output = RESULT_DIR / "long_context.json"

    with output.open("w") as f:
        json.dump(
            {
                "model": "Qwen3-8B",
                "turboquant_bits": BITS,
                "dtype": "bfloat16",
                "max_new_tokens": MAX_NEW_TOKENS,
                "results": results,
            },
            f,
            indent=2,
        )

    print()
    print("==========================================")
    print("TurboQuant long-context benchmark completed.")
    print(f"Results: {output}")
    print("==========================================")


if __name__ == "__main__":
    main()
