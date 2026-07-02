"""
End-to-end numerical test for the full prefill/decode split in models.py.

For each prompt:
  1. prefill vs HF: model.prefill(tokens) compared against
     AutoModelForCausalLM's last-token logits for `tokens`.
  2. decode vs HF: model.decode() called for several synthetic continuation
     tokens (simulating a real generation loop), each step compared against
     the corresponding position's logits from a single HF forward pass over
     the full [tokens, continuation] sequence (logits at position p predict
     token p+1, so decode step j after prefill(tokens) lines up with
     ref_logits[len(tokens) - 1 + j]).
  3. cache self-consistency (no HF needed): the final decode step's logits
     must closely match a from-scratch model.prefill(tokens + continuation)
     on the *same* model instance (cache reset in between) - this is the
     core correctness property of a KV cache: incremental decode must equal
     full recomputation. "Closely" rather than "exactly" because batched
     (prefill) vs sequential (decode) matmuls hit bf16 rounding in a
     different order - a few tokens usually match bit-for-bit, but that's
     not guaranteed.
"""

import argparse
import torch

from transformers import AutoModelForCausalLM
from nooblang.inference.tokenizer import Tokenizer
from nooblang.inference.models import Qwen2_5
from nooblang.inference.load_model import load_model_config
from .common import reset_kvcache

PROMPTS = [
    "Hello, world!",
    "What is 2+2?",
    "The quick brown fox jumps over the lazy dog.",
    "def foo():\n    return 42",
    "Once upon a time,",
    "café — résumé — 日本語 — 🦀",
    "very " + "long " * 100 + "message",
]

CONTINUATION_STEPS = 3
TOP_K = 5
COSINE_SIM_THRESHOLD = 0.95
TOPK_OVERLAP_THRESHOLD = 0.6
SELF_CONSISTENCY_MAX_REL_DIFF = 0.02


def compare_logits(actual: torch.Tensor, expected: torch.Tensor, k: int = TOP_K):
    actual = actual.float()
    expected = expected.float()

    cos_sim = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
    max_abs_diff = (actual - expected).abs().max().item()

    actual_top1 = actual.argmax().item()
    expected_top1 = expected.argmax().item()

    actual_topk = set(torch.topk(actual, k).indices.tolist())
    expected_topk = set(torch.topk(expected, k).indices.tolist())
    topk_overlap = len(actual_topk & expected_topk) / k

    return {
        "cos_sim": cos_sim,
        "max_abs_diff": max_abs_diff,
        "top1_match": actual_top1 == expected_top1,
        "actual_top1": actual_top1,
        "expected_top1": expected_top1,
        "topk_overlap": topk_overlap,
    }


def report(i, name, result, failures):
    ok = (
        result["top1_match"]
        and result["cos_sim"] >= COSINE_SIM_THRESHOLD
        and result["topk_overlap"] >= TOPK_OVERLAP_THRESHOLD
    )
    if ok:
        print(
            f"[{i}] OK   {name:34s} cos_sim={result['cos_sim']:.4f} "
            f"top{TOP_K}_overlap={result['topk_overlap']:.2f}"
        )
    else:
        failures[0] += 1
        print(f"[{i}] FAIL {name}")
        print(
            f"     cos_sim={result['cos_sim']:.4f} "
            f"max_abs_diff={result['max_abs_diff']:.4f} "
            f"top1: actual={result['actual_top1']} expected={result['expected_top1']} "
            f"top{TOP_K}_overlap={result['topk_overlap']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer(args.model_path)
    model = Qwen2_5(args.model_path)
    vocab_size = load_model_config(args.model_path)["vocab_size"]

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    ref_model.eval()

    failures = [0]
    case_count = 0
    for prompt in PROMPTS:
        tokens = tokenizer.tokenize(prompt)
        # deterministic pseudo-random continuation, doesn't need to be
        # semantically valid text - only needs to be valid token ids.
        continuation = [(1000 + 37 * j) % vocab_size for j in range(CONTINUATION_STEPS)]
        full_seq = tokens + continuation

        with torch.no_grad():
            ref_out = ref_model(input_ids=torch.tensor([full_seq]))
        ref_logits = ref_out.logits[0]

        # 1. prefill vs HF
        prefill_logits = model.prefill(tokens)
        result = compare_logits(prefill_logits, ref_logits[len(tokens) - 1])
        report(case_count, f"{prompt!r} prefill", result, failures)
        case_count += 1

        # 2. decode vs HF, multi-step (cache growing across steps)
        decode_logits = None
        for j, tok in enumerate(continuation):
            decode_logits = model.decode(tok)
            expected = ref_logits[len(tokens) + j]
            result = compare_logits(decode_logits, expected)
            report(case_count, f"{prompt!r} decode[{j}]", result, failures)
            case_count += 1

        # 3. cache self-consistency: cached decode must equal a from-scratch
        # recompute of the same full sequence, on the same model instance.
        reset_kvcache(model)
        recompute_logits = model.prefill(full_seq)
        diff = (decode_logits.float() - recompute_logits.float()).abs().max().item()
        scale = recompute_logits.float().abs().max().item() + 1e-6
        rel_diff = diff / scale
        ok = rel_diff <= SELF_CONSISTENCY_MAX_REL_DIFF
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures[0] += 1
        print(
            f"[{case_count}] {status} {prompt!r} decode-vs-recompute "
            f"max_rel_diff={rel_diff:.6f}"
        )
        case_count += 1

        reset_kvcache(model)  # leave clean for the next prompt

    print(f"\n{case_count - failures[0]}/{case_count} passed")


if __name__ == "__main__":
    main()
