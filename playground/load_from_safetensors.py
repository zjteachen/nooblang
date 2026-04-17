"""
Legacy file: used to manually inspect HF safetensors format for educational purposes.
"""

import argparse
import re
import os
import json

# import torch

from collections import defaultdict


def extract_metadata(f):
    """
    Extract json metadata from safetensors.
    """
    # from format https://github.com/huggingface/safetensors
    N = int.from_bytes(f.read(8), byteorder="little")
    f.seek(8)
    json_bytes = f.read(N)
    json_string = json_bytes.decode("utf-8")
    json_data = json.loads(json_string)
    return json_data


def extract_tensor_bytes(tensor_data: dict):
    """
    Example of extracted structure in Qwen:

    "self_attn.v_proj": {
        "dtype": "BF16",
        "shape": [
            256,
            1536
        ],
        "data_offsets": [
            559556608,
            560343040
        ]
    }
    """
    dtype = tensor_data.get("dtype")
    shape = tensor_data.get("shape")
    data_offsets = tensor_data.get("data_offsets")
    assert (
        dtype == "BF16" and isinstance(shape, list) and isinstance(data_offsets, list)
    ), f"Attempted to extract tensor bytes with invalid structure {tensor_data}"
    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
    parser.add_argument("-m", "--model-path", help="Path to model folder.")

    args = parser.parse_args()

    model_path = args.model_path

    # load model safetensors
    f = open(os.path.join(model_path, "model.safetensors"), "rb")
    json_data = extract_metadata(f)

    pattern = r"model\.layers\.(\d+)\.(.+)\.(weight|bias)$"
    tensors = defaultdict(lambda: {"weight": [], "bias": []})
    non_matches = []

    for key in json_data.keys():
        m = re.match(pattern, key)
        if not m:
            print(f"Metadata key did not match extraction pattern: {key}")
            non_matches.append(key)
        else:
            idx, tensortype, kind = m.groups()
            if kind not in ["weight", "bias"]:
                print(f"Unexpected kind found in key{key}")
            else:
                tensors[idx][kind].append({tensortype: json_data.get(key)})
    print("\nEnd of process\n\n")
    print(tensors["0"])
    sampletensor = tensors["0"]["weight"][0]
    extract_tensor_bytes(sampletensor.get("input_layernorm"))
