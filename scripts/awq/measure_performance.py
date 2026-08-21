import json
import os
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_PATH = "/workspace/models/Qwen3-8B-AWQ"
RESULT_PATH = "/workspace/results/awq/performance.json"

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


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

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

input_tokens = len(
    tokenizer(
        formatted_prompt,
        add_special_tokens=False,
    )["input_ids"]
)

print("Loading AWQ model with vLLM...")

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    seed=SEED,
    max_model_len=2048,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_NEW_TOKENS,
)


def run_once(measure=True):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if measure:
        torch.cuda.reset_peak_memory_stats()

    ram_monitor = RAMMonitor()

    if measure:
        ram_monitor.start()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    outputs = llm.generate(
        [formatted_prompt],
        sampling_params,
        use_tqdm=False,
    )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    if measure:
        peak_ram = ram_monitor.stop()
    else:
        peak_ram = None

    output = outputs[0]
    generated_tokens = len(output.outputs[0].token_ids)

    if not measure:
        return None

    total_latency = end_time - start_time

    end_to_end_tokens_per_second = (
        generated_tokens / total_latency
        if total_latency > 0
        else 0.0
    )

    peak_vram_allocated = torch.cuda.max_memory_allocated()
    peak_vram_reserved = torch.cuda.max_memory_reserved()

    return {
        "total_latency_ms": total_latency * 1000,
        "generated_tokens": int(generated_tokens),
        "end_to_end_tokens_per_second":
            end_to_end_tokens_per_second,
        "peak_vram_allocated_gib":
            bytes_to_gib(peak_vram_allocated),
        "peak_vram_reserved_gib":
            bytes_to_gib(peak_vram_reserved),
        "peak_ram_gib":
            bytes_to_gib(peak_ram),
    }


print()
print(f"Running {WARMUP_RUNS} warm-up runs...")

for i in range(WARMUP_RUNS):
    print(f"Warm-up {i + 1}/{WARMUP_RUNS}")
    run_once(measure=False)


print()
print(f"Running {MEASURED_RUNS} measured runs...")

runs = []

for i in range(MEASURED_RUNS):
    result = run_once(measure=True)
    runs.append(result)

    print(
        f"Run {i + 1}/{MEASURED_RUNS}: "
        f"Latency={result['total_latency_ms']:.2f} ms | "
        f"End-to-End={result['end_to_end_tokens_per_second']:.2f} tok/s | "
        f"VRAM={result['peak_vram_allocated_gib']:.2f} GiB | "
        f"RAM={result['peak_ram_gib']:.2f} GiB"
    )


def average(metric):
    return statistics.mean(
        run[metric] for run in runs
    )


summary = {
    "method": "AWQ",
    "model": "Qwen/Qwen3-8B",
    "model_path": MODEL_PATH,
    "quantization": "W4A16_ASYM",
    "runtime": "vLLM",
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
        "total_latency_ms":
            average("total_latency_ms"),

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


print()
print("==============================================")
print("Qwen3-8B AWQ PERFORMANCE SUMMARY")
print("==============================================")

print(
    f"Average latency: "
    f"{summary['averages']['total_latency_ms']:.2f} ms"
)

print(
    f"Average end-to-end speed: "
    f"{summary['averages']['end_to_end_tokens_per_second']:.2f} tokens/s"
)

print(
    f"Average peak VRAM allocated: "
    f"{summary['averages']['peak_vram_allocated_gib']:.2f} GiB"
)

print(
    f"Average peak VRAM reserved: "
    f"{summary['averages']['peak_vram_reserved_gib']:.2f} GiB"
)

print(
    f"Average peak RAM: "
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