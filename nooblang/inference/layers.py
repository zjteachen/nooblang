import argparse
import torch
import math

from .load_model import ModelLoader, load_model_config
from torch.nn import functional as F
from typing import Dict
from jaxtyping import Float


class Qwen2Layer:
    def __init__(
        self,
        n_heads: int,
        n_kvheads: int,
        tensors: Dict[str, torch.Tensor],
        rope_base: float,
        rms_norm_eps: float,
    ) -> None:
        self.tensors: Dict[str, torch.Tensor] = tensors
        self.n_heads = n_heads
        self.n_kvheads = n_kvheads
        self.rope_base = rope_base
        self.k_dim = 128
        self.rms_norm_eps = rms_norm_eps

        # TODO: check structure of layer against model config.

    def attention(self, input_seq, causal_mask):
        #
        B_K = self.tensors["self_attn.k_proj.bias"]
        W_K = self.tensors["self_attn.k_proj.weight"]
        B_Q = self.tensors["self_attn.q_proj.bias"]
        W_Q = self.tensors["self_attn.q_proj.weight"]
        B_V = self.tensors["self_attn.v_proj.bias"]
        W_V = self.tensors["self_attn.v_proj.weight"]
        W_O = self.tensors["self_attn.o_proj.weight"]

        Q = (
            (input_seq @ W_Q.T + B_Q)
            .view(-1, self.n_heads, self.k_dim)
            .transpose(0, 1)
        )
        K = (
            (input_seq @ W_K.T + B_K)
            .view(-1, self.n_kvheads, self.k_dim)
            .transpose(0, 1)
        )

        V = (
            (input_seq @ W_V.T + B_V)
            .view(-1, self.n_kvheads, self.k_dim)
            .transpose(0, 1)
        )

        # duplicate kv heads to match Q.
        assert (
            self.n_heads % self.n_kvheads == 0
        ), "# of heads not divisible by # of kvheads"
        reps = self.n_heads // self.n_kvheads
        K_rep = torch.repeat_interleave(K, reps, dim=0)
        V_rep = torch.repeat_interleave(V, reps, dim=0)

        # apply RoPE
        Q = self.apply_rope(Q)
        K_rep = self.apply_rope(K_rep)

        scores = (Q @ K_rep.transpose(-1, -2)) / math.sqrt(self.k_dim)
        scores += causal_mask
        scores = F.softmax(scores, dim=-1, dtype=torch.bfloat16)

        scores = (scores @ V_rep).transpose(0, 1).reshape(-1, self.k_dim * self.n_heads)
        scores = scores @ W_O.T
        return scores

    def apply_rope(self, tensor):
        """
        Apply RoPE over a tensor of assumed dimensions [**, sequence_length, head_dimension].
        """
        seq_len = tensor.shape[-2]
        d_k = tensor.shape[-1]

        assert d_k == self.k_dim

        theta_index = torch.arange(d_k // 2, device=tensor.device)
        token_index = torch.arange(seq_len, device=tensor.device)
        theta = torch.outer(
            token_index, (self.rope_base ** -((2 * theta_index) / d_k)).repeat(2)
        )

        x1, x2 = (
            tensor[..., : d_k // 2],
            tensor[..., d_k // 2 :],
        )

        rotated_half = torch.cat([-x2, x1], dim=-1)

        cos_t = torch.cos(theta).to(torch.bfloat16)
        sin_t = torch.sin(theta).to(torch.bfloat16)

        return cos_t * tensor + sin_t * rotated_half

    def mlp(self, input_seq):
        """
        Feed data through multi-layer perceptron.
        """
        mlp_down = self.tensors["mlp.down_proj.weight"]
        mlp_gate = self.tensors["mlp.gate_proj.weight"]
        mlp_up = self.tensors["mlp.up_proj.weight"]

        input_up = input_seq @ mlp_up.T
        input_gate = input_seq @ mlp_gate.T

        return (input_up * F.silu(input_gate)) @ mlp_down.T

    def normalize(self, input_seq, gain):
        return torch.rms_norm(
            input_seq, (input_seq.shape[-1],), gain, self.rms_norm_eps
        )

    def forward(self, input_seq):
        input_length = input_seq.shape[0]
        input_norm = self.tensors["input_layernorm.weight"]
        post_attention_norm = self.tensors["post_attention_layernorm.weight"]
        h = self.normalize(input_seq, input_norm)
        causal_mask = torch.full(
            (input_length, input_length), float("-inf"), dtype=torch.bfloat16
        ).triu(diagonal=1)
        h = self.attention(h, causal_mask)
        out = input_seq + h

        h = self.normalize(out, post_attention_norm)
        h = self.mlp(h)
        out += h
        return out

    # layer.attention(sample_input, causal_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
    parser.add_argument("-m", "--model-path", help="Path to model folder.")

    args = parser.parse_args()

    model_path = args.model_path
    config = load_model_config(model_path)

    # sample data
    input_length = 200
    ## generate random input vector of 200 tokens
    d_v = config.get("hidden_size")
    assert d_v is not None
    sample_input = torch.randn(input_length, d_v, dtype=torch.bfloat16)

    n_heads = config["num_attention_heads"]
    n_kvheads = config["num_key_value_heads"]
    rope_base = config["rope_theta"]
    rms_norm_eps = config["rms_norm_eps"]

    loader = ModelLoader(model_path)

    layer = Qwen2Layer(
        n_heads, n_kvheads, loader.load_layer(1), rope_base, rms_norm_eps
    )
    causal_mask = torch.full((input_length, input_length), float("-inf")).triu(
        diagonal=1
    )
    print("Input shape:", sample_input.shape)
    # layer.attention(sample_input, causal_mask)
    # out = layer.mlp(sample_input)
    layer_out = layer.forward(sample_input)
    print("Output shape:", layer_out.shape)
