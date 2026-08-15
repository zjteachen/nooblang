# Nooblang

A simple, barely functional inference server.

### Purpose

This project is an inference server with 2 aims:

- Development provides educational value on the subject of AI inference to the developer (me).
- This server acts as a test bed for experimentation with different AI inference optimization techniques.

Longshot goals:

- Add FP8 KV cache quantization and speculative decoding.
- Add CUDA/triton backends for fused kernel support (needed to make weight quantization actually faster, not just smaller — see [Benchmarks](#benchmarks)).

### Achievements

- Analysis and understanding of Huggingface safetensors format — [`nooblang/inference/load_model.py`](nooblang/inference/load_model.py)
- Full implementation of GQA attention — [`nooblang/inference/layers.py`](nooblang/inference/layers.py)
- Implementation of RoPE — [`nooblang/inference/layers.py`](nooblang/inference/layers.py)
- Full BPE tokenization process tested on at least Qwen-2.5 — [`nooblang/inference/tokenizer.py`](nooblang/inference/tokenizer.py)
- Basic inference generation support — [`nooblang/inference/server.py`](nooblang/inference/server.py)
- Prefill/decode stages with KV cache — [`nooblang/inference/layers.py`](nooblang/inference/layers.py)
- Basic chat interface
- 4-bit (INT4) round-to-nearest weight quantization, selectable at load time — [`nooblang/inference/quantization.py`](nooblang/inference/quantization.py), wired into [`nooblang/inference/layers.py`](nooblang/inference/layers.py)
- Generation benchmark for tokens/s and peak VRAM, to measure the effect of changes like quantization — [`tests/benchmarks/bench_generate.py`](tests/benchmarks/bench_generate.py)

### Compatibility
Currently tested on:
- Qwen-2.5-1.5B-Instruct

### Project setup
Clone the project, then install python dependencies with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```
Use `hf download` to grab the needed models, e.g.:
```bash
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen-2.5-1.5B-Instruct
```

### How to run
Basic invocation:
```bash
python -m nooblang.inference.server [args]
```
Arguments:
- `-m, --model-path [path-to-safetensors]`: (required) path to the directory containing the model in safetensors format.
- `-d, --device {gpu, cpu}`: device on which to run the server. Default `gpu`.
- `-q, --quantization {none, INT4}`: weight quantization scheme to apply as the model loads. Default `none`.

### Testing
Correctness tests against HuggingFace `transformers` reference modules; see [`TESTING.md`](TESTING.md) for tiers.

```bash
uv run python -m tests.run_all -m /path/to/Qwen-2.5-1.5B    # checkpoint required
uv run python -m tests.test_sampling                        # no checkpoint
uv run python -m tests.test_quantization                    # no checkpoint
```

`test_quantization.py` checks roundtrip error vs. a recorded benchmark, device preservation, and measured memory savings (currently 80.2%).

### Benchmarks
`tests/benchmarks/bench_generate.py` measures decode tokens/s and peak VRAM for one long-response prompt.

```bash
uv run python -m tests.benchmarks.bench_generate -m /path/to/Qwen-2.5-1.5B [-q {none,INT4}] [--max-tokens N]
```

Qwen-2.5-1.5B-Instruct, RTX 3070, 150 tokens:

| quantization | tokens/s (decode) | peak VRAM |
|---|---|---|
| `none` | ~26.2 | ~2975 MB |
| `INT4` | ~2.4 | ~1487 MB |

No fused kernel yet, so INT4 trades speed for memory. `quantize_exceptions` isn't populated from `models.py` yet, so norms get quantized too.
