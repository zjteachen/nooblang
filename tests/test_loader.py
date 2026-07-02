"""
Loader integrity: verifies ModelLoader's regex+string-split tensor slicing
against an independently-written prefix-slicing ground truth, for every
layer (not just layer 0/1) plus the non-layer tensors.
"""

import argparse
import os
import re
import torch

from safetensors import safe_open
from nooblang.inference.load_model import ModelLoader, load_model_config


def independent_layer_tensors(handle, layer_idx):
    prefix = f"model.layers.{layer_idx}."
    return {
        key[len(prefix) :]: handle.get_tensor(key)
        for key in handle.keys()
        if key.startswith(prefix)
    }


def independent_nonlayer_tensors(handle):
    return {
        key: handle.get_tensor(key)
        for key in handle.keys()
        if not re.match(r"model\.layers\.\d+\.", key)
    }


def compare(name, actual, expected, failures):
    if set(actual.keys()) != set(expected.keys()):
        failures[0] += 1
        print(f"[{name}] FAIL key set mismatch")
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        if missing:
            print(f"     missing: {sorted(missing)}")
        if extra:
            print(f"     extra:   {sorted(extra)}")
        return

    mismatched = [k for k in expected if not torch.equal(actual[k], expected[k])]
    if mismatched:
        failures[0] += 1
        print(f"[{name}] FAIL value mismatch on keys: {mismatched}")
    else:
        print(f"[{name}] OK  ({len(expected)} tensors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()

    fname = os.path.join(args.model_path, "model.safetensors")
    config = load_model_config(args.model_path)
    num_layers = config["num_hidden_layers"]

    loader = ModelLoader(args.model_path)
    handle = safe_open(fname, framework="pt", device="cpu")

    failures = [0]

    compare(
        "nonlayer",
        loader.load_nonlayer_tensors(),
        independent_nonlayer_tensors(handle),
        failures,
    )

    for i in range(num_layers):
        compare(
            f"layer {i}",
            loader.load_layer(i),
            independent_layer_tensors(handle, i),
            failures,
        )

    total = num_layers + 1
    print(f"\n{total - failures[0]}/{total} passed")


if __name__ == "__main__":
    main()
