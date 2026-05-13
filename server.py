import argparse
import os
import json
import torch

import torch.nn.functional as F

from tokenizer import Tokenizer
from models import Qwen2_5




def sample_tokens(logits, temperature=0.7, top_p=0.8):
    """implementation of top-p sampling algorithm."""
    new_distr = F.softmax(logits / temperature)
    sorted_logits, sorted_indices = torch.sort(new_distr, dim=-1, descending=True)
    cumulative_distr = torch.cumsum(sorted_logits, dim=-1) - sorted_logits > top_p

    remove_indices = cumulative_distr.scatter(dim=-1, index=sorted_indices, src=cumulative_distr)
    probs = F.softmax(logits.masked_fill(remove_indices, float('-inf')) / temperature)
    return torch.multinomial(probs, num_samples=1).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()
    model_path = args.model_path

    prompt = "Hi, how are you?"
    tokenizer = Tokenizer(model_path)
    seq = tokenizer.tokenize(prompt)
    model = Qwen2_5(model_path)

    new_seq = [*seq]
    buffer = []
    max_tokens=400
    for i in range(max_tokens):
        distr = model.predict(new_seq)
        temperature = 0.7
        top_p = 0.8
        next = sample_tokens(distr, temperature, top_p)
        new_seq.append(next)
        buffer.append(next)
        try:
            done, result = tokenizer.detokenize(buffer)
            buffer = []
            if done:
                break
            print(result, end="", flush=True)
        except: pass
