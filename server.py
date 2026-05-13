import argparse
import os
import json

from torch import mode

from tokenizer import Tokenizer
from models import Qwen2_5


def get_token_mappings(model_path: str) -> dict:
    id_to_str = {}
    with open(os.path.join(model_path, "tokenizer.json"), "r") as f:
        tokenizer_config = json.loads(f.read())

    for token in tokenizer_config["added_tokens"]:
        id_to_str[token["id"]] = token["content"]

    reversed_dict = {
        value: key for key, value in tokenizer_config["model"]["vocab"].items()
    }
    return id_to_str | reversed_dict


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
    max_tokens = 400
    for i in range(max_tokens):
        next = model.predict(new_seq)
        new_seq.append(next)
        buffer.append(next)
        try:
            done, result = tokenizer.detokenize(buffer)
            buffer = []
            if done:
                break
            print(result, end="", flush=True)

        except UnicodeDecodeError:
            pass
