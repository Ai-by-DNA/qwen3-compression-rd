import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import TurboQuantModel

MODEL_PATH = "/workspace/models/Qwen3-8B"
BITS = 3.0
MAX_NEW_TOKENS = 16
SEED = 42

torch.manual_seed(SEED)

print("Loading Qwen3-8B...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda").eval()

head_dim = int(
    getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
)

print(f"Head dim: {head_dim}")
print(f"TurboQuant bits: {BITS}")

wrapper = TurboQuantModel(
    model,
    bits=BITS,
    head_dim=head_dim,
    device="cuda",
    dtype=torch.bfloat16,
)

cache = wrapper.make_dynamic_cache(
    triton_fused_layers=True,
    strict_reencode=False,
    hybrid_float_cache=False,
)

wrapper.enable_decoder_fused_attention(
    architecture="auto",
)

messages = [
    {
        "role": "user",
        "content": "Explain photosynthesis briefly.",
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to("cuda")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        past_key_values=cache,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

wrapper.disable_decoder_fused_attention()

generated = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

print()
print("Generated text:")
print(generated)

print()
print("==========================================")
print("TurboQuant Qwen3-8B smoke test PASSED")
print("3-bit KV cache | BF16 model | fused Triton")
print("==========================================")
