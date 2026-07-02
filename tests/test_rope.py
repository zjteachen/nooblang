"""
Compares Qwen2Layer.apply_rope (used during prefill) against HF's
Qwen2RotaryEmbedding + apply_rotary_pos_emb, across several sequence lengths
(including length 1, to catch off-by-one position indexing).

Also compares apply_rope_single (used during decode, to rotate a single
Q/K vector at an arbitrary absolute cache position) against the same HF
reference evaluated at that one position - decode-step RoPE must agree with
what prefill-style RoPE would produce for a token at that position.
"""

import argparse
import torch

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from nooblang.inference.load_model import ModelLoader, load_model_config
from nooblang.inference.layers import Qwen2Layer
from .common import (
    load_hf_config,
    rotary_cos_sin,
    rotary_cos_sin_at_positions,
    compare_tensors,
)

COS_SIM_THRESHOLD = 0.999
MAX_ABS_DIFF_THRESHOLD = 0.1

SEQ_LENS = [1, 2, 16, 200]
DECODE_POSITIONS = [0, 1, 50, 127, 199]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()

    config = load_model_config(args.model_path)
    hf_config = load_hf_config(args.model_path)
    n_heads = config["num_attention_heads"]
    rope_base = config["rope_theta"]

    loader = ModelLoader(args.model_path)
    tensors = loader.load_layer(0)
    layer = Qwen2Layer(
        n_heads, config["num_key_value_heads"], tensors, rope_base, config["rms_norm_eps"]
    )
    k_dim = layer.k_dim

    cases = []

    for seq_len in SEQ_LENS:
        torch.manual_seed(0)
        # [n_heads, seq_len, head_dim], matching Qwen2Layer's internal head-major layout
        q = torch.randn(n_heads, seq_len, k_dim, dtype=torch.bfloat16)

        actual = layer.apply_rope(q)

        cos, sin = rotary_cos_sin(hf_config, seq_len)
        # HF expects [batch, heads, seq, head_dim]; unsqueeze_dim=1 broadcasts cos/sin
        # of shape [batch, seq, head_dim] over the heads dim.
        expected, _ = apply_rotary_pos_emb(q.unsqueeze(0), q.unsqueeze(0), cos, sin)
        expected = expected.squeeze(0)

        cases.append((f"apply_rope       seq_len={seq_len:4d}", actual, expected))

    for position in DECODE_POSITIONS:
        torch.manual_seed(position)
        # [n_heads, 1, head_dim], matching the shape apply_rope_single receives
        # at a real decode call site (single new token, per-head Q or K).
        q = torch.randn(n_heads, 1, k_dim, dtype=torch.bfloat16)

        actual = layer.apply_rope_single(q, position)

        cos, sin = rotary_cos_sin_at_positions(hf_config, [position])
        expected, _ = apply_rotary_pos_emb(q.unsqueeze(0), q.unsqueeze(0), cos, sin)
        expected = expected.squeeze(0)

        cases.append((f"apply_rope_single position={position:4d}", actual, expected))

    failures = 0
    for i, (name, actual, expected) in enumerate(cases):
        result = compare_tensors(actual, expected)
        ok = (
            result["cos_sim"] >= COS_SIM_THRESHOLD
            and result["max_abs_diff"] <= MAX_ABS_DIFF_THRESHOLD
        )
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"[{i}] {status} {name} cos_sim={result['cos_sim']:.6f} "
            f"max_abs_diff={result['max_abs_diff']:.6f}"
        )

    print(f"\n{len(cases) - failures}/{len(cases)} passed")


if __name__ == "__main__":
    main()
