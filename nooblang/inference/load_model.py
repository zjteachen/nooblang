"""
Loads model and defines model APIs for interactions.
"""

import os
import argparse
import re
import json

from safetensors import safe_open

from .utils import add_device_arg, resolve_device, add_quantization_arg, resolve_quantization


"""
Extracts tensors with per-layer granularity.
"""
class ModelLoader:
    def __init__(self, model_path, device: str = "cpu", quantization=None):
        fname = os.path.join(model_path, "model.safetensors")
        self.state_dict = safe_open(fname, framework="pt", device=device)
        self.quantization = quantization

    def load_nonlayer_tensors(self):
        layer_keys = {}

        for key in self.state_dict.keys():
            mt = re.match(r"model\.layers\.(\d+)", key)
            if not mt:
                layer_keys[key] = self.state_dict.get_tensor(key)

        return layer_keys

    def load_layer(self, layer_index: int):
        """Loads the tensors for a specific hidden layer."""
        layer_keys = {}

        for key in self.state_dict.keys():
            mt = re.match(r"model\.layers\.(\d+)", key)
            if mt and int(search := mt.group(1)) == layer_index:
                layer_keys[key.split(search)[1][1:]] = self.state_dict.get_tensor(key)

        return layer_keys


def load_model_config(model_path: str) -> dict:
    fname = os.path.join(model_path, "config.json")
    with open(fname, "r") as f:
        data = json.loads(f.read())
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
    parser.add_argument("-m", "--model-path", help="Path to model folder.")
    add_device_arg(parser)
    add_quantization_arg(parser)

    args = parser.parse_args()

    model_path = args.model_path
    device = resolve_device(args.device)
    quantization = resolve_quantization(args.quantization)

    loader = ModelLoader(model_path, device=device, quantization=quantization)
    loader.load_layer(4)
    print("others:")
    loader.load_nonlayer_tensors()
