"""
Take a natural language response and break it up into the sequence of token values
that can be fed into the neural network.
"""

import argparse
import os
import json
import regex
from typing import List, Sequence
from utils import bytes_to_unicode_map


class Pretokenizer:
    """
    From inspection, tokenizer_config contains pretokenizers;
    these accomplish several preprocessing steps that must be executed on an input
    before algorithms such as BPE can be run.
    For example, the Split tokenizer splits the sequence on its model-config controlled delimiters; such as space, comma, etc.
    These special characters are preserved in the split result, but the reason they are split
    is to prevent token-merging across boundaries.
    The Sequence pretokenizer creates a sort of recursive structure of tokenizers,
    so we encode that here, although I don't yet know the scope of this recursive structure
    (I suspect that they do not grow too complicated.)

    """

    def __init__(self, config: dict):
        self.config = config
        self.type = config.get("type")
        self.children = []
        if self.type == "Sequence":
            for d in config.get("pretokenizers") or []:
                self.children.append(Pretokenizer(d))

    def pretokenize(self, inp: List[str]):
        if self.type == "Split":
            # the following lines make assumptions about the Split pretokenizer's structure
            # that may not hold.
            pattern = self.config.get("pattern")
            assert pattern.get("Regex") is not None, "Found a non-regex Split pattern"
            rg = pattern.get("Regex")
            out = []
            for i in inp:
                out += regex.findall(rg, i)
            return out

        elif self.type == "Sequence":
            for pretokenizer in self.children:
                inp = pretokenizer.pretokenize(inp)
            return inp

        elif self.type == "ByteLevel":
            out = []
            for chunk in inp:
                out.append(
                    "".join(bytes_to_unicode_map[a] for a in bytes(chunk, "utf-8"))
                )
            return out

        raise NotImplementedError("Encountered unhandled pretokenizer")


def BPE_merge(text: str):
    pass


def tokenize(prompt, model_path):
    """
    This function needs to eventually fully tokenize the input.
    Pseudocode/necessary steps:
    - Apply the relevant chat template to surround the message.
    - Store and split on special tokens (tokens that should not be part of the merging process).
    - Pretokenize the split sections (which should return the chunks of byte-level encoded strings)
    - BPE merge on each individual chunk.
    - Recombine, inserting the lookups from the previously removed special tokens.
    """
    with open(os.path.join(model_path, "tokenizer.json"), "r") as f:
        tokenizer_config = json.loads(f.read())

    pretokenizer = Pretokenizer(tokenizer_config.get("pre_tokenizer"))
    return pretokenizer.pretokenize([prompt])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
    parser.add_argument("-m", "--model-path", help="Path to model folder.")

    args = parser.parse_args()

    model_path = args.model_path

    sample = """Hi, how are you?"""

    result = tokenize(sample, model_path)
    print(result)

    # tokenized = tokenize(sample)
    # tokenizer_config = open(os.path.join(model_path, "tokenizer.json"), "r").read()
    # tokenizer_config = json.loads(tokenizer_config)

    # for key in tokenizer_config.keys():
    #    s = str(tokenizer_config.get(key))
    #    if len(s) > 1000:
    #        s = s[:500] + "(truncated)"
    # print(f"{key}:", s)
    # print()

    breakpoint()
