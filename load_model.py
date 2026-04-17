"""
Loads model and defines model APIs for interactions.
"""

import os
import argparse

from safetensors import safe_open

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
    parser.add_argument("-m", "--model-path", help="Path to model folder.")

    args = parser.parse_args()

    model_path = args.model_path

    # load model safetensors
    fname = os.path.join(model_path, "model.safetensors")

    state_dict = safe_open(fname, framework="pt", device="cpu")
    breakpoint()
