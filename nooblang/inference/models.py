import argparse
import torch

from .load_model import ModelLoader, load_model_config
from .layers import Qwen2Layer
from .utils import add_device_arg, resolve_device, add_quantization_arg, resolve_quantization
from abc import ABC


class Model(ABC):
    pass


class Qwen2_5(Model):
    def __init__(self, model_path: str, device: str = "cpu", quantization=None) -> None:
        self.device = device
        self.quantization = quantization
        loader = ModelLoader(model_path, device=device, quantization=quantization)
        config = load_model_config(model_path)

        n_heads = config["num_attention_heads"]
        n_kvheads = config["num_key_value_heads"]
        rope_base = config["rope_theta"]
        self.rms_norm_eps = config["rms_norm_eps"]
        num_layers = config["num_hidden_layers"]

        nonlayer_tensors = loader.load_nonlayer_tensors()
        self.embedding_matrix = nonlayer_tensors["model.embed_tokens.weight"]
        self.full_norm_weights = nonlayer_tensors["model.norm.weight"]

        self.layers = []
        for i in range(num_layers):
            tensors = loader.load_layer(i)
            self.layers.append(
                Qwen2Layer(
                    n_heads, n_kvheads, tensors, rope_base, self.rms_norm_eps, self.quantization
                )
            )

    def prefill(self, tokens):
        """
        Prefill mode of the model. Populates KV cache.
        Returns: logits
        """
        embeddings = self.embedding_matrix[tokens]
        for layer in self.layers:
            embeddings = layer.forward(embeddings, new_kvcache=True, )

        embeddings = torch.rms_norm(
            embeddings,
            (embeddings.shape[-1],),
            self.full_norm_weights,
            self.rms_norm_eps,
        )

        logits = (embeddings @ self.embedding_matrix.T)[-1]

        return logits
    
    def decode(self, new_token):
        """
        Model decode mode. Uses kv cache.
        """
        # embedding_matrix[new_token] drops the seq dim when new_token is a
        # scalar; layer.forward requires input_seq.shape[0] == 1 for kvcache.
        token_embedding = self.embedding_matrix[new_token].view(1, -1)
        for layer in self.layers:
            token_embedding = layer.forward(token_embedding)
        token_embedding = torch.rms_norm(
            token_embedding,
            (token_embedding.shape[-1],),
            self.full_norm_weights,
            self.rms_norm_eps,
        )

        logits = (token_embedding @ self.embedding_matrix.T)[-1]
        return logits
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
    parser.add_argument("-m", "--model-path", help="Path to model folder.")
    add_device_arg(parser)
    add_quantization_arg(parser)

    args = parser.parse_args()
    model_path = args.model_path
    device = resolve_device(args.device)
    quantization = resolve_quantization(args.quantization)

    model = Qwen2_5(model_path, device=device, quantization=quantization)
    config = load_model_config(model_path)
    vocab_size = config["vocab_size"]

    sample_length = 200
    sample_tokens = torch.randint(0, vocab_size - 1, (sample_length,), device=device)
    model.prefill(sample_tokens)
