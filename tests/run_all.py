"""
Runs every test module in this suite and prints a consolidated summary.
See TESTING.md for what each tier covers.
"""

import argparse
import re
import subprocess
import sys

MODEL_REQUIRED = [
    "test_loader",
    "test_rmsnorm",
    "test_rope",
    "test_mlp",
    "test_attention",
    "test_layer",
    "test_tokenizer",
    "test_logits",
]
NO_MODEL_REQUIRED = [
    "test_sampling",
]

SUMMARY_RE = re.compile(r"(\d+)/(\d+) passed")


def run(module, model_path):
    cmd = [sys.executable, "-m", f"tests.{module}"]
    if model_path is not None:
        cmd += ["-m", model_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    match = None
    for line in output.splitlines()[::-1]:
        match = SUMMARY_RE.search(line)
        if match:
            break
    return proc.returncode, output, match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    args = parser.parse_args()

    results = []
    for module in MODEL_REQUIRED:
        results.append((module, *run(module, args.model_path)))
    for module in NO_MODEL_REQUIRED:
        results.append((module, *run(module, None)))

    print(f"{'module':20s} {'result':12s} detail")
    print("-" * 60)
    total_failed_modules = 0
    for module, returncode, output, match in results:
        if returncode != 0 and match is None:
            print(f"{module:20s} {'CRASH':12s}")
            print(output.strip()[-1500:])
            total_failed_modules += 1
            continue

        passed, total = match.groups()
        ok = passed == total
        label = "OK" if ok else "FAIL"
        if not ok:
            total_failed_modules += 1
        print(f"{module:20s} {label:12s} {passed}/{total} passed")

    print("-" * 60)
    if total_failed_modules == 0:
        print(f"All {len(results)} test modules fully passed.")
    else:
        print(f"{total_failed_modules}/{len(results)} test modules had failures.")


if __name__ == "__main__":
    main()
