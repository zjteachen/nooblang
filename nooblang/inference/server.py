import argparse
import os
import json
import torch

import torch.nn.functional as F

from .tokenizer import Tokenizer
from .models import Qwen2_5
from .utils import add_device_arg, resolve_device


def sample_tokens(logits, temperature=0.7, top_p=0.8):
    """implementation of top-p sampling algorithm."""
    new_distr = F.softmax(logits / temperature, dim=-1)
    sorted_logits, sorted_indices = torch.sort(new_distr, dim=-1, descending=True)
    cumulative_distr = torch.cumsum(sorted_logits, dim=-1) - sorted_logits > top_p

    remove_indices = cumulative_distr.scatter(
        dim=-1, index=sorted_indices, src=cumulative_distr
    )
    probs = F.softmax(
        logits.masked_fill(remove_indices, float("-inf")) / temperature, dim=-1
    )
    return torch.multinomial(probs, num_samples=1).item()


def generate(model, tokenizer, messages, max_tokens=400, temperature=0.7, top_p=0.8):
    """Streams the assistant's reply to stdout and returns the full text."""
    seq = tokenizer.tokenize(messages=messages)

    buffer = []
    response = []
    distr = model.prefill(seq)
    for _ in range(max_tokens):
        next = sample_tokens(distr, temperature, top_p)
        buffer.append(next)
        try:
            done, result = tokenizer.detokenize(buffer)
            buffer = []
            if done:
                break
            response.append(result)
            print(result, end="", flush=True)
        except UnicodeDecodeError:
            pass

        distr = model.decode(next)

    print()
    return "".join(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    add_device_arg(parser)
    args = parser.parse_args()
    model_path = args.model_path
    device = resolve_device(args.device)

    tokenizer = Tokenizer(model_path)
    model = Qwen2_5(model_path, device=device)

    print("Chat with the model. Type 'exit' or press Ctrl-D to quit.")

    messages = []
    while True:
        try:
            user_input = input(">>> ")
        except EOFError:
            print()
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})
        reply = generate(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})
