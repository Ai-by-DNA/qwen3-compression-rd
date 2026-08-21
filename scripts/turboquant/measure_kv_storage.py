import gc
import os

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from turboquant import TurboQuantModel

MODEL_PATH = "/workspace/models/Qwen3-8B"
CONTEXTS = [2048, 4096]


def gib(n):
    return n / (1024 ** 3)


def tensor_bytes(t):
    return t.numel() * t.element_size()


def dict_tensor_bytes(d):
    if not isinstance(d, dict):
        return 0
    return sum(
        tensor_bytes(v)
        for v in d.values()
        if torch.is_tensor(v)
    )


def plain_cache_bytes(cache):
    total = 0

    for layer in cache.layers:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                total += tensor_bytes(t)

    return total


def turbo_cache_bytes(cache):
    payload = 0
    runtime_copy = 0

    for layer in cache.layers:
        payload += dict_tensor_bytes(
            getattr(layer, "_compressed", None)
        )
        runtime_copy += dict_tensor_bytes(
            getattr(layer, "_compressed_v", None)
        )

    return payload, runtime_copy


def make_input(tokenizer, length):
    text = (
        "Artificial intelligence systems process information using "
        "mathematical representations and computational models. "
    )

    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    ids = ids.repeat(length // len(ids) + 1)[:length]
    return ids.unsqueeze(0).cuda()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
).cuda().eval()

head_dim = getattr(
    model.config,
    "head_dim",
    model.config.hidden_size // model.config.num_attention_heads,
)

wrapper = TurboQuantModel(
    model,
    bits=3.0,
    head_dim=head_dim,
    device="cuda",
    dtype=torch.bfloat16,
)

for length in CONTEXTS:
    print(f"\n===== {length} tokens =====")

    input_ids = make_input(tokenizer, length)

    # Plain BF16 cache
    plain_cache = DynamicCache()

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            past_key_values=plain_cache,
            use_cache=True,
            logits_to_keep=1,
        )

    plain_bytes = plain_cache_bytes(plain_cache)

    del out, plain_cache
    cleanup()

    # TurboQuant cache
    wrapper.enable_decoder_fused_attention(architecture="auto")

    tq_cache = wrapper.make_dynamic_cache(
        triton_fused_layers=True,
        strict_reencode=False,
        hybrid_float_cache=False,
    )

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            past_key_values=tq_cache,
            use_cache=True,
            logits_to_keep=1,
        )

    payload_bytes, runtime_copy_bytes = turbo_cache_bytes(tq_cache)

    wrapper.disable_decoder_fused_attention()

    total_tq = payload_bytes + runtime_copy_bytes

    print(f"Plain BF16 KV storage:       {gib(plain_bytes):.4f} GiB")
    print(f"TurboQuant payload:          {gib(payload_bytes):.4f} GiB")
    print(f"TurboQuant runtime copy:     {gib(runtime_copy_bytes):.4f} GiB")
    print(f"TurboQuant cache total:      {gib(total_tq):.4f} GiB")

    if plain_bytes:
        payload_reduction = (1 - payload_bytes / plain_bytes) * 100
        total_reduction = (1 - total_tq / plain_bytes) * 100

        print(f"Payload reduction:           {payload_reduction:.2f}%")
        print(f"Live-cache reduction:        {total_reduction:.2f}%")

    del out, tq_cache, input_ids
    cleanup()
