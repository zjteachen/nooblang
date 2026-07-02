"""
Shared helpers for building HF reference modules from our own loaded tensors.
"""

import torch

from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)


def load_hf_config(model_path):
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    return config


def _sub_state_dict(tensors, prefix):
    return {
        k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)
    }


def build_hf_mlp(config, tensors, dtype=torch.bfloat16):
    mlp = Qwen2MLP(config).to(dtype)
    mlp.load_state_dict(_sub_state_dict(tensors, "mlp."), strict=True)
    mlp.eval()
    return mlp


def build_hf_rmsnorm(weight, eps, dtype=torch.bfloat16):
    norm = Qwen2RMSNorm(weight.shape[-1], eps=eps).to(dtype)
    norm.load_state_dict({"weight": weight}, strict=True)
    norm.eval()
    return norm


def build_hf_attention(config, layer_idx, tensors, dtype=torch.bfloat16):
    attn = Qwen2Attention(config, layer_idx).to(dtype)
    attn.load_state_dict(_sub_state_dict(tensors, "self_attn."), strict=True)
    attn.eval()
    return attn


def build_hf_decoder_layer(config, layer_idx, tensors, dtype=torch.bfloat16):
    layer = Qwen2DecoderLayer(config, layer_idx).to(dtype)
    layer.load_state_dict(tensors, strict=True)
    layer.eval()
    return layer


def rotary_cos_sin_at_positions(config, positions, dtype=torch.bfloat16, device="cpu"):
    """Returns (cos, sin) of shape [1, len(positions), head_dim] for arbitrary
    (possibly non-contiguous / non-zero-start) absolute positions, as consumed
    by `apply_rotary_pos_emb`/`Qwen2Attention`. Used to validate decode-step
    RoPE, which rotates a single token at an absolute cache position rather
    than at position 0."""
    rotary = Qwen2RotaryEmbedding(config)
    dummy = torch.empty(1, dtype=dtype, device=device)
    position_ids = torch.tensor([positions], device=device)
    return rotary(dummy, position_ids)


def rotary_cos_sin(config, seq_len, dtype=torch.bfloat16, device="cpu"):
    """Returns (cos, sin) of shape [1, seq_len, head_dim], as consumed by
    `apply_rotary_pos_emb`/`Qwen2Attention`."""
    return rotary_cos_sin_at_positions(
        config, list(range(seq_len)), dtype=dtype, device=device
    )


def reset_kvcache(model):
    """Clears every layer's KV cache, as if starting a new request. Lets a
    single loaded model be reused across multiple prefill/decode scenarios
    in a test without reloading weights."""
    from nooblang.inference.layers import KVCache

    for layer in model.layers:
        layer.kvcache = KVCache()
        layer.kvcache_len = 0


def additive_causal_mask(seq_len, dtype=torch.bfloat16):
    """[1, 1, seq_len, seq_len] additive causal mask, broadcastable over
    HF's [batch, heads, q_len, kv_len] attention weights."""
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype).triu(diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float()
    expected = expected.float()
    diff = (actual - expected).abs()
    cos_sim = torch.nn.functional.cosine_similarity(
        actual.reshape(-1), expected.reshape(-1), dim=0
    ).item()
    scale = expected.abs().max().item() + 1e-6
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "max_rel_diff": diff.max().item() / scale,
        "cos_sim": cos_sim,
    }
