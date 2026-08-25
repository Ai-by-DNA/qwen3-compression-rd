import json
import os
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/workspace/models/Qwen3-8B-AWQ"
RESULT_PATH = "/workspace/results/awq/performance_hf.json"

SEED = 42
WARMUP_RUNS = 2
MEASURED_RUNS = 5
MAX_NEW_TOKENS = 128

PROMPT = (
    "Explain how photosynthesis works in plants, including the roles of "
    "chlorophyll, light, water, carbon dioxide, and the Calvin cycle. "
    "Give a clear and detailed answer in at least 250 words."
)


def bytes_to_gib(value):
    return value / (1024 ** 3)


def directory_size_bytes(path):
    total = 0

    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)

            if os.path.isfile(file_path):
                total += os.path.getsize(file_path)

    return total


def weight_size_bytes(path):
    total = 0

    for file_path in Path(path).rglob("*.safetensors"):
        total += file_path.stat().st_size

    return total


class RAMMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_rss = 0
        self.stop_event = threading.Event()
        self.thread = None

    def _monitor(self):
        while not self.stop_event.is_set():
            rss = self.process.memory_info().rss
            self.peak_rss = max(self.peak_rss, rss)
            time.sleep(0.01)

    def start(self):
        self.peak_rss = self.process.memory_info().rss
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join()

        self.peak_rss = max(
            self.peak_rss,
            self.process.memory_info().rss,
        )

        return self.peak_rss


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


# ---------------------------------------------------------
# Load AWQ with Hugging Face
# ---------------------------------------------------------

print("Loading AWQ model with Hugging Face Transformers...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": "cuda:0"},
    local_files_only=True,
    trust_remote_code=True,
)

model.eval()


# ---------------------------------------------------------
# Prepare prompt
# ---------------------------------------------------------

messages = [
    {
        "role": "user",
        "content": PROMPT,
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt",
    add_special_tokens=False,
)

inputs = {
    key: value.to("cuda:0")
    for key, value in inputs.items()
}

input_tokens = inputs["input_ids"].shape[-1]

print(f"Input tokens: {input_tokens}")


# ---------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------

@torch.inference_mode()
def generate_one_token():
    """
    Used for TTFT measurement.
    """

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        use_cache=True,
    )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    return end_time - start_time


@torch.inference_mode()
def generate_full():
    """
    Full generation for end-to-end latency.
    """

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    generated_tokens = (
        output.shape[-1]
        - inputs["input_ids"].shape[-1]
    )

    return end_time - start_time, int(generated_tokens)


# ---------------------------------------------------------
# Warm-up
# ---------------------------------------------------------

print()
print(f"Running {WARMUP_RUNS} warm-up runs...")

for i in range(WARMUP_RUNS):
    print(f"Warm-up {i + 1}/{WARMUP_RUNS}")

    generate_one_token()
    generate_full()


# ---------------------------------------------------------
# Measured runs
# ---------------------------------------------------------

print()
print(f"Running {MEASURED_RUNS} measured runs...")

runs = []

for i in range(MEASURED_RUNS):

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    ram_monitor = RAMMonitor()
    ram_monitor.start()

    # -------------------------
    # TTFT
    # -------------------------

    ttft_seconds = generate_one_token()

    # -------------------------
    # Full generation
    # -------------------------

    total_latency_seconds, generated_tokens = generate_full()

    peak_ram = ram_monitor.stop()

    # -------------------------
    # Speed calculations
    # -------------------------

    end_to_end_tokens_per_second = (
        generated_tokens / total_latency_seconds
        if total_latency_seconds > 0
        else 0.0
    )

    decode_tokens = max(generated_tokens - 1, 0)

    decode_time_seconds = max(
        total_latency_seconds - ttft_seconds,
        0.0,
    )

    decode_tokens_per_second = (
        decode_tokens / decode_time_seconds
        if decode_time_seconds > 0
        else 0.0
    )

    peak_vram_allocated = (
        torch.cuda.max_memory_allocated()
    )

    peak_vram_reserved = (
        torch.cuda.max_memory_reserved()
    )

    result = {
        "ttft_ms":
            ttft_seconds * 1000,

        "total_latency_ms":
            total_latency_seconds * 1000,

        "generated_tokens":
            generated_tokens,

        "decode_tokens_per_second":
            decode_tokens_per_second,

        "end_to_end_tokens_per_second":
            end_to_end_tokens_per_second,

        "peak_vram_allocated_gib":
            bytes_to_gib(peak_vram_allocated),

        "peak_vram_reserved_gib":
            bytes_to_gib(peak_vram_reserved),

        "peak_ram_gib":
            bytes_to_gib(peak_ram),
    }

    runs.append(result)

    print(
        f"Run {i + 1}/{MEASURED_RUNS}: "
        f"TTFT={result['ttft_ms']:.2f} ms | "
        f"Latency={result['total_latency_ms']:.2f} ms | "
        f"Decode={result['decode_tokens_per_second']:.2f} tok/s | "
        f"E2E={result['end_to_end_tokens_per_second']:.2f} tok/s | "
        f"VRAM={result['peak_vram_allocated_gib']:.2f} GiB | "
        f"RAM={result['peak_ram_gib']:.2f} GiB"
    )


# ---------------------------------------------------------
# Averages
# ---------------------------------------------------------

def average(metric):
    return statistics.mean(
        run[metric] for run in runs
    )


summary = {
    "method": "AWQ",
    "model": "Qwen/Qwen3-8B",
    "model_path": MODEL_PATH,
    "quantization": "W4A16_ASYM",

    "runtime": "Hugging Face Transformers",

    "thinking_mode": False,

    "seed": SEED,
    "batch_size": 1,

    "warmup_runs": WARMUP_RUNS,
    "measured_runs": MEASURED_RUNS,

    "input_tokens": int(input_tokens),

    "max_new_tokens": MAX_NEW_TOKENS,

    "model_directory_size_gib":
        bytes_to_gib(
            directory_size_bytes(MODEL_PATH)
        ),

    "weight_size_gib":
        bytes_to_gib(
            weight_size_bytes(MODEL_PATH)
        ),

    "averages": {

        "ttft_ms":
            average("ttft_ms"),

        "total_latency_ms":
            average("total_latency_ms"),

        "decode_tokens_per_second":
            average("decode_tokens_per_second"),

        "end_to_end_tokens_per_second":
            average("end_to_end_tokens_per_second"),

        "peak_vram_allocated_gib":
            average("peak_vram_allocated_gib"),

        "peak_vram_reserved_gib":
            average("peak_vram_reserved_gib"),

        "peak_ram_gib":
            average("peak_ram_gib"),
    },

    "runs": runs,
}


# ---------------------------------------------------------
# Save JSON
# ---------------------------------------------------------

os.makedirs(
    os.path.dirname(RESULT_PATH),
    exist_ok=True,
)

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print()
print("==============================================")
print("Qwen3-8B AWQ HF PERFORMANCE SUMMARY")
print("==============================================")

print(
    f"TTFT: "
    f"{summary['averages']['ttft_ms']:.2f} ms"
)

print(
    f"Average latency: "
    f"{summary['averages']['total_latency_ms']:.2f} ms"
)

print(
    f"Decode speed: "
    f"{summary['averages']['decode_tokens_per_second']:.2f} tokens/s"
)

print(
    f"End-to-end speed: "
    f"{summary['averages']['end_to_end_tokens_per_second']:.2f} tokens/s"
)

print(
    f"Peak VRAM allocated: "
    f"{summary['averages']['peak_vram_allocated_gib']:.2f} GiB"
)

print(
    f"Peak VRAM reserved: "
    f"{summary['averages']['peak_vram_reserved_gib']:.2f} GiB"
)

print(
    f"Peak RAM: "
    f"{summary['averages']['peak_ram_gib']:.2f} GiB"
)

print(
    f"Model directory size: "
    f"{summary['model_directory_size_gib']:.2f} GiB"
)

print(
    f"Weight size: "
    f"{summary['weight_size_gib']:.2f} GiB"
)

print()
print(f"Results saved to: {RESULT_PATH}")