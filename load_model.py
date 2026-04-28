"""
Loads model and defines model APIs for interactions.
"""

import os
import argparse
import re

from safetensors import safe_open


class ModelLoader:
    def __init__(self, model_path):
        fname = os.path.join(model_path, "model.safetensors")
        self.state_dict = safe_open(fname, framework="pt", device="cpu")

    def load_nonlayer_tensors(self):
        layer_keys = {}

        for key in self.state_dict.keys():
            mt = re.match(r"model\.layers\.(\d+)", key)
            if not mt:
                layer_keys[key] = self.state_dict.get_tensor(key)

        for k in layer_keys:
            print(k, layer_keys[k].shape)

        return layer_keys

    def load_layer(self, layer_index: int):
        """Loads the tensors for a specific hidden layer."""
        layer_keys = {}

        for key in self.state_dict.keys():
            mt = re.match(r"model\.layers\.(\d+)", key)
            if mt and int(mt.group(1)) == layer_index:
                layer_keys[key] = self.state_dict.get_tensor(key)

        for k in layer_keys:
            print(k, layer_keys[k].shape)

        return layer_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
    parser.add_argument("-m", "--model-path", help="Path to model folder.")

    args = parser.parse_args()

    model_path = args.model_path

    loader = ModelLoader(model_path)
    loader.load_layer(4)
    print("others:")
    loader.load_nonlayer_tensors()
