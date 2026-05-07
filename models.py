import argparse
import torch

from load_model import ModelLoader, load_model_config
from layers import Qwen2Layer
from torch.nn import functional as F
from abc import ABC


class Model(ABC):
    pass


class Qwen2_5(Model):
    def __init__(self, model_path: str) -> None:
        loader = ModelLoader(model_path)
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
                Qwen2Layer(n_heads, n_kvheads, tensors, rope_base, self.rms_norm_eps)
            )

    def predict(self, tokens, temperature=1.0):
        embeddings = self.embedding_matrix[tokens]
        print("embeddings:", embeddings.shape)
        for layer in self.layers:
            embeddings = layer.forward(embeddings)

        embeddings = torch.rms_norm(
            embeddings,
            (embeddings.shape[-1],),
            self.full_norm_weights,
            self.rms_norm_eps,
        )

        logits = F.linear(embeddings, self.embedding_matrix)

        last_logit = logits[-1]
        distr = F.softmax(last_logit / temperature, dim=0)
        return distr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
    parser.add_argument("-m", "--model-path", help="Path to model folder.")

    args = parser.parse_args()
    model_path = args.model_path

    model = Qwen2_5(model_path)
    config = load_model_config(model_path)
    vocab_size = config["vocab_size"]

    sample_length = 200
    sample_tokens = torch.randint(0, vocab_size - 1, (sample_length,))
    model.predict(sample_tokens)
