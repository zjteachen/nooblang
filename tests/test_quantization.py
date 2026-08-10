import argparse
import json
import os

import torch

from nooblang.inference.quantization import QuantizedTensor, RTNQuantizer

"""
flow:
- Create random N-dimensional weight tensor (can test over 1-2 datatypes)
- Run quantization
- Check size of new tensor is small
- Dequantize
- Measure error relative to original tensor
- First run: if the threshold is reasonable, set it as the benchmark
- otherwise check error vs threshold
Other assertions:
- Quantized tensor remains on the same device
"""

BITS = 4
PACK_SIZE = 64
BENCHMARK_TOLERANCE = 1.1  # allow error to drift up to 10% above the recorded benchmark
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "quantization_benchmark.json")

CASES = [
    dict(name="2d_float32", shape=(256, 512), dtype=torch.float32),
    dict(name="3d_bfloat16", shape=(8, 64, 256), dtype=torch.bfloat16),
]


def tensor_bytes(t):
    return t.numel() * t.element_size()


def quantized_footprint_bytes(quantized_tensor, dequant_data):
    """Actual measured memory footprint of the quantized representation:
    the packed payload tensor plus the starts/leaps side info needed to
    dequantize, at whatever dtypes RTNQuantizer.quantize() really returns."""
    starts_t, leaps_t = dequant_data
    return tensor_bytes(quantized_tensor) + tensor_bytes(starts_t) + tensor_bytes(leaps_t)


def relative_error(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    scale = expected.float().abs().max().item() + 1e-6
    return diff.mean().item() / scale


def load_benchmarks():
    if not os.path.exists(BENCHMARK_PATH):
        return {}
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def save_benchmarks(benchmarks):
    with open(BENCHMARK_PATH, "w") as f:
        json.dump(benchmarks, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    benchmarks = load_benchmarks()
    benchmarks_changed = False

    failures = 0
    total_original_bytes = 0
    total_quantized_bytes = 0
    for i, case in enumerate(CASES):
        torch.manual_seed(0)
        name = case["name"]
        t = torch.randn(case["shape"], dtype=case["dtype"], device=args.device)

        quantizer = RTNQuantizer(bits=BITS, pack_size=PACK_SIZE)
        q = QuantizedTensor(t, quantizer)

        quantized_tensor = q.get_quantized_tensor()

        original_bytes = tensor_bytes(t)
        compressed_bytes = quantized_footprint_bytes(quantized_tensor, q.dequant_data)
        size_ok = compressed_bytes < original_bytes
        savings_pct = (1 - compressed_bytes / original_bytes) * 100
        total_original_bytes += original_bytes
        total_quantized_bytes += compressed_bytes

        device_ok = quantized_tensor.device == t.device

        dequantized = q.dequantize()
        error = relative_error(dequantized, t)

        if name not in benchmarks:
            benchmarks[name] = error
            benchmarks_changed = True
            benchmark_ok = True
            note = f"no benchmark yet, recorded error={error:.6f} as baseline (review this)"
        else:
            threshold = benchmarks[name] * BENCHMARK_TOLERANCE
            benchmark_ok = error <= threshold
            note = f"error={error:.6f} threshold={threshold:.6f}"

        ok = size_ok and device_ok and benchmark_ok
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"[{i}] {status} {name:16s} "
            f"orig_bytes={original_bytes} quantized_bytes={compressed_bytes:.0f} "
            f"savings={savings_pct:.1f}% {note}"
        )
        if not size_ok:
            print(f"     quantized size ({compressed_bytes:.0f}B) not smaller than original ({original_bytes}B)")
        if not device_ok:
            print(f"     device mismatch: quantized on {quantized_tensor.device}, expected {t.device}")

    if benchmarks_changed:
        save_benchmarks(benchmarks)

    overall_savings_pct = (1 - total_quantized_bytes / total_original_bytes) * 100
    print(
        f"\nmemory usage: {total_original_bytes}B -> {total_quantized_bytes:.0f}B "
        f"({overall_savings_pct:.1f}% smaller, {BITS}-bit RTN, pack_size={PACK_SIZE})"
    )
    print(f"{len(CASES) - failures}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
