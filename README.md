# gemma4-zml-probe

A bit-exact, op-by-op port of **`google/gemma-4-E2B-it`** (text path) to
**[ZML](https://github.com/zml/zml)** — the Zig + MLIR + OpenXLA inference compiler — built and
**proven against HuggingFace Transformers one operation at a time**.

> **Status — port complete.** Prefill, logits, single-token decode, and multi-token generation all
> produce output **identical to HuggingFace**. ~50 atomic gates, each committed and tagged.
> Visual map of the whole port: [`docs/CARTOGRAPHIE_portage.md`](docs/CARTOGRAPHIE_portage.md).

```
prefill (last_hidden ~1e-5 vs HF) → logits (tokens == HF, 0 flip)
  → decode 1 token (last_hidden + logits + argmax == HF)
  → generate N tokens (sequence == HF greedy: [1018, 6398, 25967, 53121])
```

## Why

`gemma-4-E2B-it` already runs everywhere (Ollama, llama.cpp, vLLM, MLX, …). The point of this repo is
**not** "run Gemma 4" — it is a **controlled, op-level reference engine** that:

- reproduces the model **bit-near vs PyTorch** (a proven fp32 baseline you can measure against);
- is a clean substrate to **experiment at the graph level** (custom quantization, KV-cache tricks,
  architecture research) — things turnkey runtimes don't expose;
- adds **Gemma support to the ZML ecosystem** (the upstream ZML repo ships Llama / Qwen / LFM only).

It is a **research baseline**, not a fast production engine: it runs on **CPU in fp32**, without batching,
sampling, or fast-prefill.

## What was ported (the tricky bits of Gemma 4)

- **Per-Layer Embeddings (PLE)** — second embedding table injecting a per-layer residual (`×√256`).
- **Shared KV Cache ("YOCO")** — writers (layers 13 sliding / 14 full) produce K/V reused by readers
  (layers 15–34, Q-only).
- **Two layer types** — sliding (head_dim 256, RoPE θ=1e4, window 512, MLP 6144) and full (head_dim 512,
  **partial RoPE 0.25**, θ=1e6 "proportional", double-wide MLP 12288).
- **GQA** 8 Q / 1 KV head · **RMSNorm** (Llama-style) · `q/k/v_norm` (v without scale) ·
  `gelu_pytorch_tanh` · final softcap `30·tanh(x/30)` · per-layer `layer_scalar`.
- **Incremental decode** — growing KV cache via `scatterSlices(slot, pos)`, absolute `pos_idx`,
  incremental mask, cache threaded step-to-step.

## Method (the discipline)

Every operation is a **gate**: read `modeling_gemma4.py` (assume nothing) → **PyTorch oracle** (the
ground truth) → fixture → **ZML runner** → compare (fixed points + global scan, tolerance 1e-4) →
commit + tag. Multi-tap isolation localizes any drift; an **oracle-independence** rule prevents
shared-assumption false passes; selected milestones were adversarially reviewed.

## Repo layout

```
scripts/      Python oracles (PyTorch / HF) + fixture exporters  (00 → 44)
zml_runner/   ZML runners (.zig) + BUILD.bazel + deploy script
docs/         per-gate notes, precision contract, roadmap, cartography
fixtures/     manifests (the .npy/.pt/.safetensors are regenerable, gitignored)
```

Engine highlights: `zml_runner/gemma4_prefill.zig` (35-layer prefill engine),
`gemma4_logits.zig`, `gemma4_decode{1,2,3,4}.zig` (sliding pilot → full pilot → e2e engine →
generation loop).

## Reproduce

**Prerequisites**

- A Hugging Face account with the **Gemma license accepted** (`huggingface-cli login`).
- Python env (see `requirements.txt`). Tested with **transformers 5.9.0**, **torch 2.12.0**.
- A **ZML** checkout (Bazel) on a compute host. Tested on CPU (`libpjrt_cpu`); runs on a single host.
- `google/gemma-4-E2B-it` weights at `weights/model.safetensors`.

**Run a gate** (oracle → runner)

```bash
# 1. Oracle (PyTorch) produces a fixture under fixtures/
python scripts/40_p5_7_7_decode_pilot_oracle.py

# 2. Build & run the matching ZML runner inside your ZML workspace
#    (deploy sources with zml_runner/deploy_to_3090.sh, configured via env vars)
./bazel.sh build //examples/rqz:gemma4_decode1
./bazel-bin/examples/rqz/gemma4_decode1 weights/model.safetensors fixtures/p5_7_7_decode1.safetensors
```

Each runner prints `max_abs` / `mean_abs` vs the oracle and a PASS/FAIL verdict.

## Limitations / not done (optional extensions)

CPU fp32 only · no batching / sampling / fast-prefill · sliding window > 512 (ring buffer) not handled ·
multimodal (vision/audio) out of scope (text path only) · no independent perf benchmarks.

## License & attribution

Code: **Apache-2.0** (see [`LICENSE`](LICENSE)) — same as ZML and Gemma. © 2026 Régis Rigaud / TheCause.
The Gemma 4 model weights are distributed by Google under the
[Gemma / Apache-2.0 terms](https://huggingface.co/google/gemma-4-E2B-it) — not included here.
