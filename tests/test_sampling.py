"""
Verifies server.sample_tokens' top-p (nucleus) filtering against an
independently-written reference nucleus-mask calculation (same published
top-p formula, different code path). No model/checkpoint needed.

For several fixed synthetic logit distributions and a fixed RNG seed:
  - every token ever drawn must lie inside the reference nucleus set
    (hard invariant - must always hold, or the algorithm is filtering wrong).
  - every token in the reference nucleus set is eventually drawn (the
    synthetic distributions are engineered so this holds with overwhelming
    probability across N_SAMPLES draws - not flaky).
"""

import argparse
import torch

from nooblang.inference.server import sample_tokens

N_SAMPLES = 3000


def reference_nucleus_set(logits, temperature, top_p):
    """Standard nucleus-sampling formula (Holtzman et al.), written
    independently of server.sample_tokens' scatter-based implementation."""
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    keep_sorted = cum_before <= top_p
    kept_positions = sorted_idx[keep_sorted]
    return set(kept_positions.tolist())


CASES = [
    # 10 well-separated logits, rest hugely negative -> small, unambiguous nucleus
    dict(
        name="separated_top_p_0.9",
        logits=torch.tensor(
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]
            + [-100.0] * 20
        ),
        temperature=1.0,
        top_p=0.9,
    ),
    dict(
        name="separated_top_p_0.5",
        logits=torch.tensor(
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]
            + [-100.0] * 20
        ),
        temperature=1.0,
        top_p=0.5,
    ),
    dict(
        # no near-zero-probability padding here: at top_p=1.0 the nucleus is
        # the *entire* vocab, so every included token must have a real chance
        # of being drawn within N_SAMPLES for this to be a meaningful check.
        name="uniform_top_p_1.0",
        logits=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
        temperature=1.0,
        top_p=1.0,
    ),
    dict(
        name="low_temperature_sharpens",
        logits=torch.tensor(
            [3.0, 2.9, 2.8, 0.0, -1.0] + [-100.0] * 20
        ),
        temperature=0.3,
        top_p=0.9,
    ),
]


def main():
    failures = 0
    for i, case in enumerate(CASES):
        torch.manual_seed(42)
        nucleus = reference_nucleus_set(case["logits"], case["temperature"], case["top_p"])

        drawn = set()
        for _ in range(N_SAMPLES):
            token = sample_tokens(case["logits"], case["temperature"], case["top_p"])
            drawn.add(token)

        leaked = drawn - nucleus
        missed = nucleus - drawn

        ok = not leaked and not missed
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"[{i}] {status} {case['name']:24s} "
            f"nucleus_size={len(nucleus):2d} drawn_size={len(drawn):2d}"
        )
        if leaked:
            print(f"     drew tokens OUTSIDE nucleus (filter bug): {sorted(leaked)}")
        if missed:
            print(f"     nucleus tokens never drawn in {N_SAMPLES} samples: {sorted(missed)}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
