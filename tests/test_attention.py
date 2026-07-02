"""
Compares Qwen2Layer.attention (hand-rolled GQA + RoPE, with KV cache) against
HF's Qwen2Attention (eager), loaded with the same q/k/v/o projection weights.

Two modes are exercised, matching the two call sites in the codebase:
  - prefill (new_kvcache=True): full-sequence attention with a causal mask,
    populates the cache from scratch.
  - decode (new_kvcache=False): single new token attends over the cache
    populated by a prior prefill call. Reference is a full HF recompute over
    the concatenated [prefill_tokens, new_token] sequence, taking only the
    last position's output - i.e. decode-with-cache must reproduce exactly
    what recompute-from-scratch would produce for that position.
"""

import argparse
import torch

from nooblang.inference.load_model import ModelLoader, load_model_config
from nooblang.inference.layers import Qwen2Layer
from .common import (
    load_hf_config,
    build_hf_attention,
    rotary_cos_sin,
    additive_causal_mask,
    compare_tensors,
)

COS_SIM_THRESHOLD = 0.99
MAX_REL_DIFF_THRESHOLD = 0.25

SEQ_LENS = [1, 2, 16, 200]
LAYER_IDX = 5


def run_prefill_cases(config, hf_config, tensors, hidden_size):
    layer = Qwen2Layer(
        config["num_attention_heads"],
        config["num_key_value_heads"],
        tensors,
        config["rope_theta"],
        config["rms_norm_eps"],
    )
    ref_attn = build_hf_attention(hf_config, LAYER_IDX, tensors)

    results = []
    for seq_len in SEQ_LENS:
        torch.manual_seed(seq_len)
        input_seq = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16)

        causal_mask = torch.full(
            (seq_len, seq_len), float("-inf"), dtype=torch.bfloat16
        ).triu(diagonal=1)
        actual = layer.attention(input_seq, causal_mask, new_kvcache=True)

        cos, sin = rotary_cos_sin(hf_config, seq_len)
        mask = additive_causal_mask(seq_len)
        with torch.no_grad():
            expected, _ = ref_attn(
                hidden_states=input_seq.unsqueeze(0),
                position_embeddings=(cos, sin),
                attention_mask=mask,
            )
        expected = expected.squeeze(0)

        results.append((f"prefill seq_len={seq_len:4d}", actual, expected))
    return results


def run_decode_cases(config, hf_config, tensors, hidden_size):
    ref_attn = build_hf_attention(hf_config, LAYER_IDX, tensors)

    results = []
    for prompt_len in SEQ_LENS:
        # fresh layer per case: KV cache is stateful, must not leak across cases
        layer = Qwen2Layer(
            config["num_attention_heads"],
            config["num_key_value_heads"],
            tensors,
            config["rope_theta"],
            config["rms_norm_eps"],
        )

        torch.manual_seed(1000 + prompt_len)
        prompt = torch.randn(prompt_len, hidden_size, dtype=torch.bfloat16)
        new_token = torch.randn(1, hidden_size, dtype=torch.bfloat16)

        prompt_mask = torch.full(
            (prompt_len, prompt_len), float("-inf"), dtype=torch.bfloat16
        ).triu(diagonal=1)
        layer.attention(prompt, prompt_mask, new_kvcache=True)  # populate cache
        actual = layer.attention(new_token, None, new_kvcache=False)

        full_seq = torch.cat([prompt, new_token], dim=0)
        total_len = prompt_len + 1
        cos, sin = rotary_cos_sin(hf_config, total_len)
        mask = additive_causal_mask(total_len)
        with torch.no_grad():
            expected_full, _ = ref_attn(
                hidden_states=full_seq.unsqueeze(0),
                position_embeddings=(cos, sin),
                attention_mask=mask,
            )
        expected = expected_full.squeeze(0)[-1:]

        results.append((f"decode  prompt_len={prompt_len:4d}", actual, expected))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()

    config = load_model_config(args.model_path)
    hf_config = load_hf_config(args.model_path)
    loader = ModelLoader(args.model_path)
    tensors = loader.load_layer(LAYER_IDX)
    hidden_size = config["hidden_size"]

    cases = run_prefill_cases(
        config, hf_config, tensors, hidden_size
    ) + run_decode_cases(config, hf_config, tensors, hidden_size)

    failures = 0
    for i, (name, actual, expected) in enumerate(cases):
        result = compare_tensors(actual, expected)
        ok = (
            result["cos_sim"] >= COS_SIM_THRESHOLD
            and result["max_rel_diff"] <= MAX_REL_DIFF_THRESHOLD
        )
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"[{i}] {status} {name} cos_sim={result['cos_sim']:.6f} "
            f"max_rel_diff={result['max_rel_diff']:.4f}"
        )

    print(f"\n{len(cases) - failures}/{len(cases)} passed")


if __name__ == "__main__":
    main()
