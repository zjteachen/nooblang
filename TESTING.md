# Testing Plan

This project hand-rolls every load-bearing piece of a Qwen2.5 forward pass —
tokenizer, RoPE, GQA attention, SwiGLU MLP, RMSNorm, the transformer stack, and
sampling — directly from safetensors weights, with no framework doing the math
for us. That means bugs don't show up as exceptions, they show up as *subtly
wrong numbers*. The only reliable way to catch that is to compare every
load-bearing component against a trusted reference implementation, not just
check that shapes line up.

Reference of choice: **HuggingFace `transformers`' own `modeling_qwen2.py`**
(already a project dependency). For each component we build the corresponding
HF `nn.Module`, load our own extracted tensors into it via `load_state_dict`,
and run both implementations on identical inputs. `load_state_dict(strict=True)`
doubles as a free structural check — if our tensor keys/shapes don't match
HF's own module layout, the test fails before any numerics are even compared.

All tests are standalone argparse scripts (`-m/--model-path`), matching the
existing `tests/test_tokenizer.py` style — no pytest, no mocked weights, run
directly against a real checkpoint (e.g. `Qwen-2.5-1.5B`). Shared helpers for
building/loading reference HF modules live in `tests/common.py`.

## Tiers

### Tier 0 — Weight loading (`tests/test_loader.py`)
Everything downstream depends on `ModelLoader` slicing the right tensors out
of `model.safetensors` for the right layer, with the right key names. This is
regex + string-split based (`load_model.py`), which is exactly the kind of
code that silently misbehaves on edge-index layers.

- **Reference:** raw `safe_open` reads + plain prefix-slicing (`key[len(prefix):]`)
  done independently in the test, not reusing `ModelLoader`'s regex/split logic.
- **Checks:** `load_nonlayer_tensors()` excludes all `model.layers.*` keys;
  `load_layer(i)` returns exactly the expected key set for every layer index
  (not just layer 0/1), with tensor values bit-identical (`torch.equal`) to
  the independently-sliced ground truth; no cross-layer key/value bleed.

### Tier 1 — Component-level numerics (reference = matching HF submodule)
Each hand-written building block in `nooblang/inference/layers.py` is
compared against the equivalent HF module loaded with the *same weights*.
Metrics: max abs/relative diff, cosine similarity, all in float32 for
comparison (computation itself stays in the model's native bf16).

Since `layers.py` now has a KV cache, `Qwen2Layer.forward`/`.attention` take
a `new_kvcache` flag with two distinct code paths that both need coverage:
`new_kvcache=True` (**prefill**: full sequence, causal mask, cache built from
scratch) and `new_kvcache=False` (**decode**: single new token, no mask,
attends over whatever's already in `self.kvcache`). A test that only ever
prefills would silently miss decode-only bugs (e.g. in `apply_rope_single`'s
absolute-position math, or the K/V `torch.cat` onto the cache).

| Component | File | Ours | Reference |
|---|---|---|---|
| RMSNorm | `tests/test_rmsnorm.py` | `Qwen2Layer.normalize` (`torch.rms_norm`) | `Qwen2RMSNorm` |
| RoPE (prefill + decode) | `tests/test_rope.py` | `Qwen2Layer.apply_rope` / `.apply_rope_single` | `Qwen2RotaryEmbedding` + `apply_rotary_pos_emb`, evaluated at matching absolute positions |
| MLP (SwiGLU) | `tests/test_mlp.py` | `Qwen2Layer.mlp` | `Qwen2MLP` |
| GQA Attention (prefill + decode) | `tests/test_attention.py` | `Qwen2Layer.attention` | `Qwen2Attention` (eager) |

For decode-mode attention/RoPE there's no direct decode equivalent in HF (it
doesn't expose a bare single-step cached call at this level), so the
reference is: prefill a prompt through HF normally, then do one more full
HF forward over `[prompt, new_token]` and take only the last position's
output. Our decode step (prompt prefilled into `self.kvcache`, then one
`new_kvcache=False` call) must reproduce that last position exactly -
this is what actually exercises the cache math instead of just the shapes.

### Tier 2 — Integration: one full transformer block (`tests/test_layer.py`)
Component tests can all pass while the *wiring* between them is wrong (e.g.
norm applied post- instead of pre-attention, a missing residual add). This
test runs `Qwen2Layer.forward` against `Qwen2DecoderLayer.forward` end to end
for a single layer, with identical weights, mask, and rotary embeddings -
again for both prefill and decode, using the same "prefill then one more
step, compare to a fresh full HF forward" pattern as Tier 1's attention test.

### Tier 3 — System / end-to-end (`tests/test_logits.py`, `tests/test_tokenizer.py`)
- **Tokenizer** (`tests/test_tokenizer.py`) — our BPE pipeline vs
  `AutoTokenizer.apply_chat_template`, on a curated set of adversarial
  strings (whitespace, unicode, fake special tokens, long input).
- **Full model prefill/decode** (`tests/test_logits.py`) — three checks per
  prompt, using `Qwen2_5.prefill()` / `.decode()` (there is no single-shot
  `predict()` anymore):
  1. `prefill(tokens)` vs `AutoModelForCausalLM`'s last-token logits.
  2. Several `decode()` steps in a row (simulating a real generation loop)
     vs. the corresponding positions from one HF forward pass over
     `[tokens, continuation]` - checks the cache holds up over multiple
     steps, not just one.
  3. **Cache self-consistency**, no HF involved: run `prefill` then a few
     `decode` steps on one model instance, reset its cache
     (`common.reset_kvcache`), then `prefill` the *same* full token sequence
     from scratch on that same instance. The two must agree closely (small
     bf16 reordering slack allowed, since batched vs. sequential matmuls
     round differently) - if they don't, the cache is not actually
     equivalent to recomputation, which is the entire point of having one.

This is the integration backstop: if every component test passes but check
1 or 2 fails, the bug is in how `models.py` composes the layers; if only
check 3 fails, the bug is specifically in cache bookkeeping, not the math.

### Tier 4 — Algorithmic correctness, no model needed (`tests/test_sampling.py`)
`sample_tokens` (top-p/nucleus sampling in `server.py`) is a pure algorithm
and doesn't need a checkpoint to validate. Reference: an independently
written nucleus-mask calculation (same published top-p formula, different
code path — not a copy of the implementation under test). For several fixed
synthetic logit distributions and a fixed RNG seed:
- every token ever drawn (over many samples) must lie inside the reference
  nucleus set (hard invariant, must always hold),
- every token in the reference nucleus set must eventually be drawn (given
  a synthetic distribution engineered so this is true with overwhelming
  probability — not flaky).

## Running everything

Each test is runnable standalone:
```
uv run python -m tests.test_loader     -m /path/to/Qwen-2.5-1.5B
uv run python -m tests.test_rmsnorm    -m /path/to/Qwen-2.5-1.5B
uv run python -m tests.test_rope       -m /path/to/Qwen-2.5-1.5B
uv run python -m tests.test_mlp        -m /path/to/Qwen-2.5-1.5B
uv run python -m tests.test_attention  -m /path/to/Qwen-2.5-1.5B
uv run python -m tests.test_layer      -m /path/to/Qwen-2.5-1.5B
uv run python -m tests.test_tokenizer  -m /path/to/Qwen-2.5-1.5B
uv run python -m tests.test_logits     -m /path/to/Qwen-2.5-1.5B
uv run python -m tests.test_sampling
```
or all at once via the aggregator:
```
uv run python -m tests.run_all -m /path/to/Qwen-2.5-1.5B
```
