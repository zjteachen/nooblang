# Nooblang

A simple, barely functional inference server.

### Purpose

This project is an inference server with 2 aims:

- Development provides educational value on the subject of AI inference to the developer (me).
- This server acts as a test bed for experimentation with different AI inference optimization techniques.

Longshot goals:

- Add utilities for benchmarking performance and accuracy (for testing techniques such as FP8 KV quantization, speculative decoding).
- Add CUDA/triton backends for fused kernel support.

### Achievements

- Analysis and understanding of Huggingface safetensors format — [`nooblang/inference/load_model.py`](nooblang/inference/load_model.py)
- Full implementation of GQA attention — [`nooblang/inference/layers.py`](nooblang/inference/layers.py)
- Implementation of RoPE — [`nooblang/inference/layers.py`](nooblang/inference/layers.py)
- Full BPE tokenization process tested on at least Qwen-2.5 — [`nooblang/inference/tokenizer.py`](nooblang/inference/tokenizer.py)
- Basic inference generation support — [`nooblang/inference/server.py`](nooblang/inference/server.py)
- Prefill/decode stages with KV cache — [`nooblang/inference/layers.py`](nooblang/inference/layers.py)
- Basic chat interface

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
`-m, --model-path [path-to-safetensors]`: (required) path to the directory containing the model in safetensors format. 
`-d, --device {gpu, cpu}`: device on which to run the server. Default `gpu`.
