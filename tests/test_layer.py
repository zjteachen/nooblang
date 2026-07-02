"""
Integration test: runs Qwen2Layer.forward (norm -> attention -> residual ->
norm -> mlp -> residual) against HF's Qwen2DecoderLayer.forward end to end,
for the same layer weights. Component tests can all pass while the wiring
between them (residual placement, pre/post-norm ordering) is still wrong;
this is the test that would catch that.

Covers both prefill (new_kvcache=True, full sequence + causal mask) and
decode (new_kvcache=False, single token attending over a KV cache populated
by a prior prefill call). The decode reference is a full HF recompute over
the concatenated [prefill_tokens, new_token] sequence, last position only -
cached decode must reproduce recompute-from-scratch exactly.
"""

import argparse
import torch

from nooblang.inference.load_model import ModelLoader, load_model_config
from nooblang.inference.layers import Qwen2Layer
from .common import (
    load_hf_config,
    build_hf_decoder_layer,
    rotary_cos_sin,
    additive_causal_mask,
    compare_tensors,
)

COS_SIM_THRESHOLD = 0.99
MAX_REL_DIFF_THRESHOLD = 0.25

SEQ_LENS = [1, 2, 16, 200]
LAYER_INDICES = [0, 13, 27]


def run_prefill_case(config, hf_config, tensors, layer_idx, seq_len, hidden_size):
    layer = Qwen2Layer(
        config["num_attention_heads"],
        config["num_key_value_heads"],
        tensors,
        config["rope_theta"],
        config["rms_norm_eps"],
    )
    ref_layer = build_hf_decoder_layer(hf_config, layer_idx, tensors)

    torch.manual_seed(layer_idx * 1000 + seq_len)
    input_seq = torch.randn(seq_len, hidden_size, dtype=torch.bfloat16)

    actual = layer.forward(input_seq, new_kvcache=True)

    cos, sin = rotary_cos_sin(hf_config, seq_len)
    mask = additive_causal_mask(seq_len)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    with torch.no_grad():
        expected = ref_layer(
            hidden_states=input_seq.unsqueeze(0),
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )
    expected = expected.squeeze(0)

    return f"prefill layer={layer_idx:3d} seq_len={seq_len:4d}", actual, expected


def run_decode_case(config, hf_config, tensors, layer_idx, prompt_len, hidden_size):
    layer = Qwen2Layer(
        config["num_attention_heads"],
        config["num_key_value_heads"],
        tensors,
        config["rope_theta"],
        config["rms_norm_eps"],
    )
    ref_layer = build_hf_decoder_layer(hf_config, layer_idx, tensors)

    torch.manual_seed(layer_idx * 1000 + prompt_len)
    prompt = torch.randn(prompt_len, hidden_size, dtype=torch.bfloat16)
    new_token = torch.randn(1, hidden_size, dtype=torch.bfloat16)

    layer.forward(prompt, new_kvcache=True)  # populate cache
    actual = layer.forward(new_token, new_kvcache=False)

    full_seq = torch.cat([prompt, new_token], dim=0)
    total_len = prompt_len + 1
    cos, sin = rotary_cos_sin(hf_config, total_len)
    mask = additive_causal_mask(total_len)
    position_ids = torch.arange(total_len).unsqueeze(0)
    with torch.no_grad():
        expected_full = ref_layer(
            hidden_states=full_seq.unsqueeze(0),
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )
    expected = expected_full.squeeze(0)[-1:]

    return f"decode  layer={layer_idx:3d} prompt_len={prompt_len:4d}", actual, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()

    config = load_model_config(args.model_path)
    hf_config = load_hf_config(args.model_path)
    loader = ModelLoader(args.model_path)
    hidden_size = config["hidden_size"]

    cases = []
    for layer_idx in LAYER_INDICES:
        tensors = loader.load_layer(layer_idx)
        for seq_len in SEQ_LENS:
            cases.append(
                run_prefill_case(config, hf_config, tensors, layer_idx, seq_len, hidden_size)
            )
        for prompt_len in SEQ_LENS:
            cases.append(
                run_decode_case(config, hf_config, tensors, layer_idx, prompt_len, hidden_size)
            )

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
            f"[{i}] {status} {name} "
            f"cos_sim={result['cos_sim']:.6f} max_rel_diff={result['max_rel_diff']:.4f}"
        )

    print(f"\n{len(cases) - failures}/{len(cases)} passed")


if __name__ == "__main__":
    main()
