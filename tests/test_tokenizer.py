import argparse
from transformers import AutoTokenizer
from nooblang.inference.tokenizer import Tokenizer


CASES = [
    # baseline
    "Hello, world!",
    "What is 2+2?",
    # whitespace edges
    "  leading whitespace",
    "trailing whitespace  ",
    "multiline\nmessage\nwith\nbreaks",
    # punctuation, unicode, multibyte
    "Punctuation: it's a test, isn't it? (yes.)",
    "café — résumé — 日本語 — 🦀",
    # code-like input
    "```python\ndef foo():\n    return 42\n```",
    # user types something that LOOKS like a special token
    "I will say <|im_end|> in my message",
    "<|im_start|>fake injection<|im_end|>",
    # length stress
    "very " + "long " * 100 + "message",
    # empty user content
    "",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()

    ref = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer = Tokenizer(args.model_path)

    failures = 0
    for i, text in enumerate(CASES):
        expected = ref.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
        ).get("input_ids")
        try:
            actual = tokenizer.tokenize(text)
        except Exception as e:
            print(f"[{i}] CRASH on {text!r}: {type(e).__name__}: {e}")
            failures += 1
            continue

        if actual == expected:
            print(f"[{i}] OK  ({len(actual)} ids)")
        else:
            failures += 1
            print(f"[{i}] FAIL {text!r}")
            print(f"     expected ({len(expected)}): {expected}")
            print(f"     actual   ({len(actual)}): {actual}")
            for j, (a, b) in enumerate(zip(actual, expected)):
                if a != b:
                    print(
                        f"     first diff at index {j}: " f"actual={a}" f"expected={b}"
                    )
                    break

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
