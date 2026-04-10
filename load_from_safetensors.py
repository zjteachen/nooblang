"""
Load model weights from HF safetensors format.
"""

import argparse
import re
import os
import json

from collections import defaultdict

parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
parser.add_argument("-m", "--model-path", help="Path to model folder.")

args = parser.parse_args()

model_path = args.model_path


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


if __name__ == "__main__":

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
                tensors[idx][kind].append(tensortype)
    print(tensors["0"])
