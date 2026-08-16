"""
Simple generation benchmark: send the model one prompt that requires a
somewhat long response, and measure peak VRAM usage and decode tokens/s.

Meant as a before/after baseline for changes like quantization, not a
correctness check (see ../test_*.py for those).
"""

import argparse
import json
import os
import time

import torch

from nooblang.inference.models import Qwen2_5
from nooblang.inference.server import sample_tokens
from nooblang.inference.tokenizer import Tokenizer
from nooblang.inference.utils import (
    add_device_arg,
    resolve_device,
    add_quantization_arg,
    resolve_quantization,
)

DEFAULT_PROMPT = (
    "Write a detailed short story about a lighthouse keeper who discovers a "
    "message in a bottle. Include dialogue and give it a clear beginning, "
    "middle, and end."
)

# GPU-only: peak VRAM isn't measured on CPU, and CPU tokens/s live on a
# different scale, so regression tracking only applies to --device gpu.
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "generation_benchmark.json")
TOKENS_PER_SEC_TOLERANCE = 0.85  # fail if tokens/s drops below 85% of the recorded baseline
PEAK_VRAM_TOLERANCE = 1.10  # fail if peak VRAM grows past 110% of the recorded baseline


def load_benchmarks():
    if not os.path.exists(BENCHMARK_PATH):
        return {}
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def save_benchmarks(benchmarks):
    with open(BENCHMARK_PATH, "w") as f:
        json.dump(benchmarks, f, indent=2, sort_keys=True)


def check_regression(quantization, result, device):
    """Compares result against the recorded baseline for this quantization
    mode, recording a new baseline on first run. Returns (ok, note)."""
    if device != "cuda":
        return True, "regression check skipped (requires --device gpu)"

    benchmarks = load_benchmarks()
    tokens_per_sec = result["tokens_per_sec"]
    peak_vram_mb = result["peak_vram_bytes"] / (1024 ** 2)

    if quantization not in benchmarks:
        benchmarks[quantization] = dict(tokens_per_sec=tokens_per_sec, peak_vram_mb=peak_vram_mb)
        save_benchmarks(benchmarks)
        return True, f"no benchmark yet, recorded tokens/s={tokens_per_sec:.2f} peak_vram={peak_vram_mb:.1f}MB as baseline (review this)"

    baseline = benchmarks[quantization]
    tokens_per_sec_floor = baseline["tokens_per_sec"] * TOKENS_PER_SEC_TOLERANCE
    peak_vram_ceiling = baseline["peak_vram_mb"] * PEAK_VRAM_TOLERANCE

    speed_ok = tokens_per_sec >= tokens_per_sec_floor
    memory_ok = peak_vram_mb <= peak_vram_ceiling
    ok = speed_ok and memory_ok

    note = (
        f"tokens/s={tokens_per_sec:.2f} (floor={tokens_per_sec_floor:.2f}, baseline={baseline['tokens_per_sec']:.2f}) "
        f"peak_vram={peak_vram_mb:.1f}MB (ceiling={peak_vram_ceiling:.1f}MB, baseline={baseline['peak_vram_mb']:.1f}MB)"
    )
    return ok, note


def run_benchmark(model, tokenizer, prompt, device, max_tokens=400, temperature=0.7, top_p=0.8):
    seq = tokenizer.tokenize(messages=[{"role": "user", "content": prompt}])

    if device == "cuda":
        torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    distr = model.prefill(seq)
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_time = time.perf_counter() - prefill_start

    buffer = []
    num_tokens = 0
    decode_start = time.perf_counter()
    for _ in range(max_tokens):
        next_token = sample_tokens(distr, temperature, top_p)
        buffer.append(next_token)
        num_tokens += 1
        try:
            done, _ = tokenizer.detokenize(buffer)
            buffer = []
            if done:
                break
        except UnicodeDecodeError:
            pass
        distr = model.decode(next_token)
    if device == "cuda":
        torch.cuda.synchronize()
    decode_time = time.perf_counter() - decode_start

    peak_vram_bytes = torch.cuda.max_memory_allocated() if device == "cuda" else None

    return dict(
        num_tokens=num_tokens,
        prefill_time=prefill_time,
        decode_time=decode_time,
        tokens_per_sec=num_tokens / decode_time if decode_time > 0 else float("nan"),
        peak_vram_bytes=peak_vram_bytes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    add_device_arg(parser)
    add_quantization_arg(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    quantization = resolve_quantization(args.quantization)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tokenizer = Tokenizer(args.model_path)
    model = Qwen2_5(args.model_path, device=device, quantization=quantization)

    result = run_benchmark(
        model,
        tokenizer,
        args.prompt,
        device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"prompt: {args.prompt!r}")
    print(f"quantization: {args.quantization}")
    print(f"tokens generated: {result['num_tokens']}")
    print(f"prefill time: {result['prefill_time']:.3f}s")
    print(f"decode time: {result['decode_time']:.3f}s")
    print(f"tokens/s (decode): {result['tokens_per_sec']:.2f}")
    if result["peak_vram_bytes"] is not None:
        print(f"peak VRAM: {result['peak_vram_bytes'] / (1024 ** 2):.1f} MB")
    else:
        print("peak VRAM: n/a (running on cpu)")

    ok, note = check_regression(args.quantization, result, device)
    status = "OK  " if ok else "FAIL"
    print(f"[regression] {status} {note}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
