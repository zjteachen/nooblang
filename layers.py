import argparse
import torch
from load_model import ModelLoader, load_model_config
from torch.nn import functional as F
from typing import Dict
from jaxtyping import Float


class Qwen2Layer:
    def __init__(
        self, n_heads: int, n_kvheads: int, tensors: Dict[str, torch.Tensor]
    ) -> None:
        self.tensors: Dict[str, torch.Tensor] = tensors
        self.n_heads = n_heads
        self.n_kvheads = n_kvheads
        # check structure of layer against model config.

    def attention(self, input_seq):
        #
        B_K = self.tensors.get("self_attn.k_proj.bias")
        W_K = self.tensors.get("self_attn.k_proj.weight")
        B_Q = self.tensors.get("self_attn.q_proj.bias")
        W_Q = self.tensors.get("self_attn.q_proj.weight")
        B_V = self.tensors.get("self_attn.v_proj.bias")
        W_V = self.tensors.get("self_attn.v_proj.weight")
        W_O = self.tensors.get("self_attn.o_proj.weight")

        Q = F.linear(input_seq, W_Q, B_Q)
        K = F.linear(input_seq, W_K, B_K)
        V = F.linear(input_seq, W_V, B_V)

        for m in [Q, K, V]:
            print(m.shape)
        # split grouped heads into multiple heads.

    def mlp(self):
        pass


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

    loader = ModelLoader(model_path)

    layer = Qwen2Layer(n_heads, n_kvheads, loader.load_layer(1))
    layer.attention(sample_input)
