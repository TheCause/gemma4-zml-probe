# gemma4-zml-probe

A bit-exact, op-by-op port of **`google/gemma-4-E2B-it`** (text path) to
**[ZML](https://github.com/zml/zml)** — the Zig + MLIR + OpenXLA inference compiler — built and
**proven against HuggingFace Transformers one operation at a time**, then grown into an autonomous
text→text engine with long-context generation, bf16 fidelity, static batching, and **4-bit weights**.

> **Status — port complete + autonomous runtime + long generation + bf16 + batching + 4-bit weights.**
> Prefill, logits, single-token decode and **1020-token** generation all reproduce HuggingFace
> (token-exact in fp32; within the measured HF-bf16 envelope in bf16). The engine now runs
> **standalone on GPU** (native tokenizer, chat template, EOS early-stop) and carries a modular
> decode socle (`EngineModel(comptime Brick, EngineCfg)`) with proven-neutral bricks.
> ~60 atomic gates, each committed and tagged.
> Visual map of the core port: [`docs/CARTOGRAPHIE_portage.md`](docs/CARTOGRAPHIE_portage.md).
> Full documentation (capabilities, usage, method, 20 pitfalls — in French): [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

```
prefill (last_hidden ~1e-5 vs HF) → logits (tokens == HF, 0 flip)
  → decode 1 token (last_hidden + logits + argmax == HF)
  → generate 1020 tokens [linear / ring-512 / autonomous] (== HF greedy, sliding window 512)
  → autonomous text→text on GPU (tokenizer + chat template + EOS in-graph)
```

## Milestones (all merged to `main`)

| Milestone | Result | Proof |
|---|---|---|
| **Op-by-op port** (text path) | prefill / logits / decode **== HF** | ~50 gates, `docs/CARTOGRAPHIE_portage.md` |
| **Long generation** | 1020 tokens **== HF greedy** (CPU chunked + GPU mono-graph) — **109 tok/s** fp32 GPU, sliding window 512 crossed, non-vacuity proven on logits | PR #1/#3 |
| **bf16 fidelity** | G2 envelope method + **G2.3 per-op sensitivity map** — 12 op families SAFE, combined config at **0.486×** the HF-bf16 envelope | `docs/G2_3_OP_SENSITIVITY.md` |
| **Autonomous runtime** | text→text on GPU: native ZML tokenizer + Gemma chat template + EOS early-stop, engine `engine.zig` untouched | PR #5 |
| **VRAM guard** | refuses to start under a measured free-VRAM threshold (real peak ≈ 16.3 GiB) | PR #6 |
| **L3 in-graph** | gather + `forwardStep` + top-k fused in one compiled graph, host threads a single scalar/step — **113 tok/s** | PR #7 |
| **Static batching** | shape-polymorphic engine (one binary, byte-identical HLO for all B); **113 → 2106 tok/s** (B=64, ×18.5), mono-sequence non-regression 0.999 | PR #8, `docs/BATCHING_RESULTS.md` |
| **4-bit weights (W4)** | `dequantW4` brick (int4 w4a16 compressed-tensors → bf16 in-graph); E2B-W4 decode GPU **48/48 == HF reading the same checkpoint**, **40.9 tok/s**, real VRAM peak **10 524 MiB (−37 % vs bf16)** | PR #9, `docs/W4_RESULTS.md` |

## Why

`gemma-4-E2B-it` already runs everywhere (Ollama, llama.cpp, vLLM, MLX, …). The point of this repo is
**not** "run Gemma 4" — it is a **controlled, op-level reference engine** that:

- reproduces the model **bit-near vs PyTorch** (a proven fp32 baseline you can measure against);
- is a clean substrate to **experiment at the graph level** (custom quantization, KV-cache tricks,
  architecture research) — things turnkey runtimes don't expose. The **modular decode socle**
  (`EngineModel(comptime Brick, EngineCfg)`) lets a brick inject a transformation with a
  byte-identical-HLO neutrality proof; the **4-bit weights** work is the largest brick to date;
- adds **Gemma support to the ZML ecosystem** (the upstream ZML repo ships Llama / Qwen / LFM only).

It began as a **research baseline** (CPU, fp32, op-by-op) and has been grown, gate by gate, into a
GPU engine that generates autonomously in bf16, batches, and runs 4-bit weights — while keeping the
fp32 op-by-op oracle as the correctness ground truth.

## What was ported (the tricky bits of Gemma 4)

- **Per-Layer Embeddings (PLE)** — second embedding table injecting a per-layer residual (`×√256`).
- **Shared KV Cache ("YOCO")** — writers (layers 13 sliding / 14 full) produce K/V reused by readers
  (layers 15–34, Q-only). The E2B checkpoint has **no k/v/k_norm modules on the 20 reader layers**.
- **Two layer types** — sliding (head_dim 256, RoPE θ=1e4, window 512, MLP 6144) and full (head_dim 512,
  **partial RoPE 0.25**, θ=1e6 "proportional", double-wide MLP 12288).
- **GQA** 8 Q / 1 KV head · **RMSNorm** (Llama-style) · `q/k/v_norm` (v without scale) ·
  `gelu_pytorch_tanh` · final softcap `30·tanh(x/30)` · per-layer `layer_scalar`.
- **Incremental decode** — growing KV cache via `scatterSlices(slot, pos)`, absolute `pos_idx`,
  incremental mask, cache threaded step-to-step.
- **4-bit weights** — `weight_packed` i32 [out, in/8] (little-endian nibbles storing q+8) +
  `weight_scale` bf16 [out, in/32]; dequant `(nibble−8)·scale` done **in the graph** so weights
  reside packed in VRAM. Finding: 10 of the 11 linear families are scale-invariant by construction
  (the norms absorb any uniform scale error) — only `gate_proj` (a non-linearity) carries the
  sensitivity (see `docs/W4_RESULTS.md`, pitfall #20).

## Method (the discipline)

Every operation is a **gate**: read `modeling_gemma4.py` (assume nothing) → **PyTorch oracle** (the
ground truth) → fixture → **ZML runner** → compare (fixed points + global scan, tolerance 1e-4) →
commit + tag. Multi-tap isolation localizes any drift; an **oracle-independence** rule prevents
shared-assumption false passes; selected milestones were adversarially reviewed. In fp32 the criterion
is **token-exact == HF**; in bf16 / on recompiled GPU it becomes **≤ 2× the measured HF-bf16 envelope**
(no bit-for-bit between two XLA-GPU compiles — autotuning). Counter-tests are checked on **logits**,
not argmax (greedy is too robust to reveal a masked path).

## Repo layout

```
scripts/      Python oracles (PyTorch / HF) + fixture exporters  (00 → 61)
zml_runner/   ZML runners (.zig) + BUILD.bazel + deploy script
docs/         per-gate notes, precision contract, roadmap, cartography, results
fixtures/     manifests (the .npy/.pt/.safetensors are regenerable, gitignored)
```

Engine highlights: `zml_runner/engine.zig` (modular 35-layer decode socle,
`EngineModel(comptime Brick, EngineCfg)`), `gemma4_gen_auto.zig` (autonomous text→text runtime),
`gemma4_bbatch.zig` (static batching), `w4.zig` + `gemma4_w4auto.zig` (4-bit weights brick + runner),
`gemma4_w4gate.zig` (4-bit unit gates). The historical op-by-op runners
(`gemma4_decode{1,2,3,4}.zig`, `gemma4_logits.zig`, …) remain as the reference trail.

## Reproduce

**Prerequisites**

- A Hugging Face account with the **Gemma license accepted** (`huggingface-cli login`).
- Python env (see `requirements.txt`). Tested with **transformers 5.9.0**, **torch 2.12.0**.
  The 4-bit work adds **llm-compressor** + **compressed-tensors ≥ 0.15** (a separate venv).
- A **ZML** checkout (Bazel) on a compute host. Tested on CPU (`libpjrt_cpu`) and on a single
  GPU (`--@zml//platforms:cuda=true`, RTX 3090).
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

**Autonomous inference on a custom prompt (end-to-end, GPU)**

```bash
# ZML tokenizes, applies the Gemma chat template, generates, detokenizes — no fixture needed.
./bazel.sh run //examples/rqz:gemma4_gen_auto --@zml//platforms:cuda=true -- \
  weights/model.safetensors gemma4-e2b-it-meta/tokenizer.json \
  --prompt "What is the capital of France? Answer in one word." --max-tokens 48
# stdout: "Paris"
```

**4-bit weights (W4)** — quantize E2B to w4a16, then decode it on GPU:

```bash
# 1. Produce the w4a16 checkpoint (data-free RTN, Google's recipe) in the w4quant venv
python scripts/54_w4_quantize.py                        # → weights_w4/ (276 packed linears)
# 2. Decode it on GPU — must match HF reading the SAME w4a16 checkpoint, token-for-token
./bazel.sh run //examples/rqz:gemma4_w4auto --@zml//platforms:cuda=true -- \
  weights_w4/model.safetensors gemma4-e2b-it-meta/tokenizer.json \
  --prompt "What is the capital of France? Answer in one word." --oracle w4_gen48.safetensors
```

Worked example — prompt *"capital of France"* → ZML **48/48 == HF**, decoded text **"Paris"**. In fp32
the engine is token-exact vs HF; the batched and 4-bit paths are validated within the measured
envelope. HF stays the reference oracle; ZML is the validated engine that reproduces it.

## Limitations / not done (optional extensions)

Text path only — **multimodal (vision/audio) out of scope**. No sampling (greedy only), no
continuous batching / serving, no fast-prefill, 256K context not exercised. The static-batch path
assumes equal tokenized prompt lengths. On E2B the 4-bit VRAM gain is bounded by the bf16 embeddings
(expected — the brick targets the 12B, where the linears dominate). No independent perf benchmarks
beyond the reported token-for-token gates.

**Next (at the design stage):** porting **Gemma 4 12B** (`Gemma4Unified`) on the 3090 via the 4-bit
brick — spec drafted, execution gated on a go/no-go decision (see
[`docs/superpowers/specs/2026-07-18-w4-poids-4bit-12b-design.md`](docs/superpowers/specs/2026-07-18-w4-poids-4bit-12b-design.md)).
An upstream-ZML flash-attention path (batch > 1) would require paged KV.

## License & attribution

Code: **Apache-2.0** (see [`LICENSE`](LICENSE)) — same as ZML and Gemma. © 2026 Régis Rigaud / TheCause.
The Gemma 4 model weights are distributed by Google under the
[Gemma / Apache-2.0 terms](https://huggingface.co/google/gemma-4-E2B-it) — not included here.
