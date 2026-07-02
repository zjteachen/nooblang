"""
Compares Qwen2Layer.mlp (hand-rolled SwiGLU) against HF's Qwen2MLP, loaded
with the same gate/up/down projection weights, across several layers.
"""

import argparse
import torch

from nooblang.inference.load_model import ModelLoader, load_model_config
from nooblang.inference.layers import Qwen2Layer
from .common import load_hf_config, build_hf_mlp, compare_tensors

COS_SIM_THRESHOLD = 0.999
MAX_ABS_DIFF_THRESHOLD = 0.5

LAYER_INDICES = [0, 1, 13, 27]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()

    config = load_model_config(args.model_path)
    hf_config = load_hf_config(args.model_path)
    loader = ModelLoader(args.model_path)

    seq_len = 16
    hidden_size = config["hidden_size"]

    failures = 0
    for i, layer_idx in enumerate(LAYER_INDICES):
        tensors = loader.load_layer(layer_idx)
        layer = Qwen2Layer(
            config["num_attention_heads"],
            config["num_key_value_heads"],
            tensors,
            config["rope_theta"],
            config["rms_norm_eps"],
        )
        ref_mlp = build_hf_mlp(hf_config, tensors)

        torch.manual_seed(layer_idx)
        input_seq = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16)

        actual = layer.mlp(input_seq)
        with torch.no_grad():
            expected = ref_mlp(input_seq.unsqueeze(0)).squeeze(0)

        result = compare_tensors(actual, expected)
        ok = (
            result["cos_sim"] >= COS_SIM_THRESHOLD
            and result["max_abs_diff"] <= MAX_ABS_DIFF_THRESHOLD
        )
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"[{i}] {status} layer={layer_idx:3d} cos_sim={result['cos_sim']:.6f} "
            f"max_abs_diff={result['max_abs_diff']:.6f}"
        )

    print(f"\n{len(LAYER_INDICES) - failures}/{len(LAYER_INDICES)} passed")


if __name__ == "__main__":
    main()
