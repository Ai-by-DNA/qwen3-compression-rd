import gc
import json
import os
import threading
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from turboquant import TurboQuantModel, TurboQuantProd

os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

MODEL_PATH = "/workspace/models/Qwen3-8B"
RESULT_DIR = Path("/workspace/results/turboquant")

BITS = 3.0
SEED = 42
MAX_NEW_TOKENS = 128
WARMUP_RUNS = 2
MEASURED_RUNS = 5

PROMPT = (
    "Explain photosynthesis in at least 250 words, including the roles of "
    "chlorophyll, light, water, carbon dioxide, and the Calvin cycle."
)


class TimingStreamer(BaseStreamer):
    def __init__(self):
        self.skip_prompt = True
        self.first_token_time = None

    def put(self, value):
        if self.skip_prompt:
            self.skip_prompt = False
            return
        if self.first_token_time is None:
            torch.cuda.synchronize()
            self.first_token_time = time.perf_counter()

    def end(self):
        pass


class RAMMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = 0
        self.running = False

    def _run(self):
        while self.running:
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(0.005)

    def start(self):
        self.peak = self.process.memory_info().rss
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        return self.peak


def directory_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def make_cache(wrapper):
    return wrapper.make_dynamic_cache(
        triton_fused_layers=True,
        strict_reencode=False,
        hybrid_float_cache=False,
    )


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

    wrapper.enable_decoder_fused_attention(architecture="auto")

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    input_tokens = int(inputs["input_ids"].shape[1])

    print("TurboQuant: 3-bit KV cache")
    print(f"Input tokens: {input_tokens}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")

    def generate_once(measured):
        gc.collect()
        torch.cuda.synchronize()

        if measured:
            torch.cuda.reset_peak_memory_stats()

        streamer = TimingStreamer()
        monitor = RAMMonitor()

        if measured:
            monitor.start()

        start = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                past_key_values=make_cache(wrapper),
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        end = time.perf_counter()

        peak_ram = monitor.stop() if measured else 0
        generated_tokens = int(output.shape[1] - inputs["input_ids"].shape[1])

        if not measured:
            return None

        ttft = streamer.first_token_time - start
        total = end - start
        decode_time = max(total - ttft, 1e-9)
        decode_tokens = max(generated_tokens - 1, 0)

        return {
            "ttft_ms": ttft * 1000,
            "total_latency_ms": total * 1000,
            "decode_tokens_per_second": decode_tokens / decode_time,
            "end_to_end_tokens_per_second": generated_tokens / total,
            "peak_vram_allocated_gib":
                torch.cuda.max_memory_allocated() / (1024 ** 3),
            "peak_vram_reserved_gib":
                torch.cuda.max_memory_reserved() / (1024 ** 3),
            "peak_ram_gib": peak_ram / (1024 ** 3),
        }

    print("\nRunning warm-ups...")
    for i in range(WARMUP_RUNS):
        generate_once(False)
        print(f"Warm-up {i + 1}/{WARMUP_RUNS}")

    print("\nRunning measured runs...")
    runs = []

    for i in range(MEASURED_RUNS):
        r = generate_once(True)
        runs.append(r)
        print(
            f"Run {i + 1}/5: "
            f"TTFT={r['ttft_ms']:.2f} ms | "
            f"Latency={r['total_latency_ms']:.2f} ms | "
            f"Decode={r['decode_tokens_per_second']:.2f} tok/s | "
            f"VRAM={r['peak_vram_allocated_gib']:.2f} GiB"
        )

    keys = runs[0].keys()
    averages = {
        key: sum(r[key] for r in runs) / len(runs)
        for key in keys
    }

    model_size = directory_size(MODEL_PATH) / (1024 ** 3)

    results = {
        "method": "TurboQuant",
        "bits": BITS,
        "model": "Qwen3-8B",
        "thinking_mode": False,
        "seed": SEED,
        "input_tokens": input_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "averages": averages,
        "model_directory_size_gib": model_size,
        "runs": runs,
    }

    output_path = RESULT_DIR / "performance.json"

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n===== TurboQuant PERFORMANCE SUMMARY =====")
    print(f"TTFT: {averages['ttft_ms']:.2f} ms")
    print(f"Latency: {averages['total_latency_ms']:.2f} ms")
    print(f"Decode: {averages['decode_tokens_per_second']:.2f} tok/s")
    print(f"End-to-end: {averages['end_to_end_tokens_per_second']:.2f} tok/s")
    print(f"Peak VRAM: {averages['peak_vram_allocated_gib']:.2f} GiB")
    print(f"Peak RAM: {averages['peak_ram_gib']:.2f} GiB")
    print(f"Model size: {model_size:.2f} GiB")
    print(f"Results: {output_path}")

    wrapper.disable_decoder_fused_attention()


if __name__ == "__main__":
    main()
