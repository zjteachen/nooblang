"""
Take a natural language response and break it up into the sequence of token values
that can be fed into the neural network.
"""

import argparse
import os
import json
import regex
import heapq
import jinja2

from typing import List, Sequence, Optional, Tuple
from itertools import count
from .utils import bytes_to_unicode_map, reverse_bytes_to_unicode_map
from dataclasses import dataclass


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


def BPE_merge(text: str, merges: dict, vocab: dict) -> List[int]:
    if text == "":
        return []

    @dataclass
    class Node:
        id: int
        next: "None | Node"
        prev: "None | Node"
        left: str
        right: str
        deleted: bool = False

    head = Node(0, None, None, "#placeholder$", text[0])
    last = head

    pq = []

    merge_to_id = dict(
        (tuple(merge.split(" ", 1)), i) for i, merge in enumerate(merges)
    )

    counter = count()
    for i in range(1, len(text)):
        pair = (text[i - 1], text[i])
        newnode = Node(i, None, last, *pair)

        last.next = newnode
        last = newnode

        priority = merge_to_id.get(pair)
        if priority is not None:
            heapq.heappush(pq, (priority, next(counter), last))

    def update_heapq(node):
        if not node.deleted:
            merge_id = merge_to_id.get((node.left, node.right))
            if merge_id is not None:
                heapq.heappush(pq, (merge_id, next(counter), node))

    while pq:
        merge, _, node = heapq.heappop(pq)
        # staleness check: if the merge id does not match the node's stored merge
        if not node.deleted and merges[merge] == node.left + " " + node.right:
            # unchecked assertion: we will never try to remove the only node
            assert node is not head
            node.prev.next = node.next
            node.prev.right = node.left + node.right
            if node.prev is not head:
                update_heapq(node.prev)

            if node.next is not None:
                node.next.prev = node.prev
                node.next.left = node.left + node.right
                update_heapq(node.next)

            node.deleted = True

    seq = []
    while head:
        seq.append(vocab[head.right])  # qwen has no UNK
        head = head.next

    return seq


class Tokenizer:
    """
    Full translation flow from plaintext -> tokens:
    - Split around special tokens that are mapped one-to-one with token ids.
    - Get unicode bytes of chunks and remap to readable representations.
    - Merge bytes with BPE to get the tokens.
    - Map the token bytes to their IDs.
    """

    def __init__(self, model_path):
        with open(os.path.join(model_path, "tokenizer.json"), "r") as f:
            self.tokenizer_config = json.loads(f.read())

        with open(os.path.join(model_path, "tokenizer_config.json"), "r") as f:
            self.extra_config = json.loads(f.read())

        self.eos_token = self.extra_config["eos_token"]
        self.pretokenizer = Pretokenizer(self.tokenizer_config.get("pre_tokenizer"))

        with open("template.txt", "r") as f:
            self.template = jinja2.Template(f.read())

        self.reverse_special_map = dict(
            (a["id"], a["content"]) for a in self.tokenizer_config["added_tokens"]
        )
        vocab = self.tokenizer_config["model"]["vocab"]

        self.reverse_token_map = dict((vocab[k], k) for k in vocab)

    def tokenize(self, prompt=None, messages=None):
        """
        This function needs to eventually fully tokenize the input.
        Pseudocode/necessary steps:
        - Apply the relevant chat template to surround the message.
        - Store and split on special tokens (tokens that should not be part of the merging process).
        - Pretokenize the split sections (which should return the chunks of byte-level encoded strings)
        - BPE merge on each individual chunk.
        - Recombine, inserting the lookups from the previously removed special tokens.
        """

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # apply chat template

        text = self.template.render(
            messages=messages,
            add_generation_prompt=True,
        )

        # extract special tokens
        special_tokens = dict(
            (a["content"], a["id"]) for a in self.tokenizer_config["added_tokens"]
        )
        pattern = f"({'|'.join(map(regex.escape, special_tokens.keys()))})"
        pieces = regex.split(pattern, text)

        ret = []
        for piece in pieces:
            if special_tokens.get(piece) is not None:
                ret.append(special_tokens[piece])
            else:
                for block in self.pretokenizer.pretokenize([piece]):
                    ret += BPE_merge(
                        block,
                        self.tokenizer_config["model"]["merges"],
                        self.tokenizer_config["model"]["vocab"],
                    )
        return ret

    def detokenize(self, tokens) -> Tuple[bool, str]:
        # construct reverse mao

        result = []
        buffer = bytearray()
        for token in tokens:
            if (chunk := self.reverse_special_map.get(token)) is not None:
                if buffer:
                    result.append(buffer.decode("utf-8", errors="replace"))
                    buffer.clear()
                result.append(chunk)
            else:
                token_bytes = self.reverse_token_map[token]
                for byte in token_bytes:
                    buffer.append(reverse_bytes_to_unicode_map[byte])

        if buffer:
            result.append(buffer.decode("utf-8"))

        finished = result[-1] == self.eos_token

        return finished, "".join(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads model from hf safetensors.")
    parser.add_argument("-m", "--model-path", help="Path to model folder.")

    args = parser.parse_args()

    model_path = args.model_path

    sample = """Hi, how are you?"""
    tokenizer = Tokenizer(model_path)

    result = tokenizer.tokenize(sample)
    print(result)

    # tokenized = tokenize(sample)

    print(tokenizer.detokenize(result))
