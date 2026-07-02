"""
Compares Qwen2Layer.normalize (torch.rms_norm) against HF's Qwen2RMSNorm,
loaded with the same gain weight, on both random and adversarial (large
magnitude, near-zero) hidden states.
"""

import argparse
import torch

from nooblang.inference.load_model import ModelLoader, load_model_config
from nooblang.inference.layers import Qwen2Layer
from .common import build_hf_rmsnorm, compare_tensors

COS_SIM_THRESHOLD = 0.999
MAX_ABS_DIFF_THRESHOLD = 0.1

CASES = [
    ("random", lambda shape: torch.randn(shape, dtype=torch.bfloat16)),
    ("near_zero", lambda shape: torch.randn(shape, dtype=torch.bfloat16) * 1e-4),
    ("large_magnitude", lambda shape: torch.randn(shape, dtype=torch.bfloat16) * 100),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()

    config = load_model_config(args.model_path)
    rms_norm_eps = config["rms_norm_eps"]
    hidden_size = config["hidden_size"]

    loader = ModelLoader(args.model_path)
    tensors = loader.load_layer(0)
    weight = tensors["input_layernorm.weight"]

    layer = Qwen2Layer(
        config["num_attention_heads"],
        config["num_key_value_heads"],
        tensors,
        config["rope_theta"],
        rms_norm_eps,
    )
    ref_norm = build_hf_rmsnorm(weight, rms_norm_eps)

    seq_len = 16
    failures = 0
    for i, (name, gen) in enumerate(CASES):
        torch.manual_seed(0)
        input_seq = gen((seq_len, hidden_size))

        actual = layer.normalize(input_seq, weight)
        with torch.no_grad():
            expected = ref_norm(input_seq.unsqueeze(0)).squeeze(0)

        result = compare_tensors(actual, expected)
        ok = (
            result["cos_sim"] >= COS_SIM_THRESHOLD
            and result["max_abs_diff"] <= MAX_ABS_DIFF_THRESHOLD
        )
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"[{i}] {status} {name:16s} cos_sim={result['cos_sim']:.6f} "
            f"max_abs_diff={result['max_abs_diff']:.6f}"
        )

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
