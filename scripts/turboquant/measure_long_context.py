import gc
import json
import os
import time
from pathlib import Path

# Reduce CUDA allocator fragmentation.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from turboquant import TurboQuantModel, TurboQuantProd


MODEL_PATH = "/workspace/models/Qwen3-8B"
RESULT_DIR = Path("/workspace/results/turboquant")
OUTPUT = RESULT_DIR / "long_context.json"

BITS = 3.0
SEED = 42
CONTEXT_LENGTHS = [2048, 4096, 8192, 16384]


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def save_results(results):
    with OUTPUT.open("w") as f:
        json.dump(
            {
                "model": "Qwen3-8B",
                "turboquant_bits": BITS,
                "dtype": "bfloat16",
                "measurement": (
                    "prefill resident cache memory and prefill peak VRAM"
                ),
                "results": results,
            },
            f,
            indent=2,
        )


def make_input(tokenizer, target_length):
    text = (
        "Artificial intelligence systems process information using "
        "mathematical representations and computational models. "
    )

    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    repeats = target_length // len(ids) + 1
    ids = ids.repeat(repeats)[:target_length]

    return ids.unsqueeze(0).to("cuda")


def measure_prefill(model, input_ids, cache):
    cleanup()

    # Input is already on GPU, so this baseline excludes its memory.
    before = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Keep cache, remove model output/logits.
    del outputs
    gc.collect()
    torch.cuda.synchronize()

    resident = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    return {
        "status": "ok",
        "prefill_seconds": elapsed,
        "resident_incremental_gib":
            max(resident - before, 0) / (1024 ** 3),
        "peak_incremental_gib":
            max(peak - before, 0) / (1024 ** 3),
        "total_peak_vram_gib": peak / (1024 ** 3),
    }


def safe_measure(model, input_ids, cache):
    try:
        return measure_prefill(model, input_ids, cache)

    except torch.OutOfMemoryError:
        cleanup()
        return {"status": "oom"}

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            cleanup()
            return {"status": "oom"}
        raise


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

    quantizer = TurboQuantProd(
        bits=BITS,
        head_dim=head_dim,
        device="cuda",
        seed=SEED,
        dtype=torch.bfloat16,
    )

    wrapper = TurboQuantModel(
        model,
        quantizer=quantizer,
    )

    results = []

    for context_length in CONTEXT_LENGTHS:
        print(f"\n===== Context: {context_length} tokens =====")

        input_ids = make_input(tokenizer, context_length)

        print("Plain HF KV cache...")
        plain = safe_measure(
            model,
            input_ids,
            DynamicCache(),
        )

        cleanup()

        print("TurboQuant 3-bit KV cache...")
        wrapper.enable_decoder_fused_attention(
            architecture="auto",
        )

        try:
            tq_cache = wrapper.make_dynamic_cache(
                triton_fused_layers=True,
                strict_reencode=False,
                hybrid_float_cache=False,
            )

            turbo = safe_measure(
                model,
                input_ids,
                tq_cache,
            )

        finally:
            wrapper.disable_decoder_fused_attention()
            if 'tq_cache' in locals():
                del tq_cache
            cleanup()

        row = {
            "context_tokens": context_length,
            "plain": plain,
            "turboquant_3bit": turbo,
        }

        if (
            plain["status"] == "ok"
            and turbo["status"] == "ok"
            and plain["resident_incremental_gib"] > 0
        ):
            reduction = (
                1
                - turbo["resident_incremental_gib"]
                / plain["resident_incremental_gib"]
            ) * 100

            row["resident_kv_reduction_percent"] = reduction

            print(
                "Plain resident incremental: "
                f"{plain['resident_incremental_gib']:.3f} GiB"
            )
            print(
                "TurboQuant resident incremental: "
                f"{turbo['resident_incremental_gib']:.3f} GiB"
            )
            print(
                "Resident reduction: "
                f"{reduction:.2f}%"
            )

            print(
                "Plain peak incremental: "
                f"{plain['peak_incremental_gib']:.3f} GiB"
            )
            print(
                "TurboQuant peak incremental: "
                f"{turbo['peak_incremental_gib']:.3f} GiB"
            )

        else:
            print(
                f"Plain status: {plain['status']} | "
                f"TurboQuant status: {turbo['status']}"
            )

        results.append(row)

        # Save immediately so partial results are never lost.
        save_results(results)

        del input_ids
        cleanup()

        # Do not attempt even larger contexts after TurboQuant OOM.
        if turbo["status"] == "oom":
            print(
                "TurboQuant reached the GPU memory limit. "
                "Stopping larger-context tests."
            )
            break

    print()
    print("==========================================")
    print("TurboQuant long-context benchmark finished.")
    print(f"Results: {OUTPUT}")
    print("==========================================")


if __name__ == "__main__":
    main()
