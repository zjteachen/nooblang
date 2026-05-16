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

### Compatibility

Give me Qwen-2.5-1.5B or I retire.

### How to run

Still figuring this out.
