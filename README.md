# gemma4-zml-probe

> **⚠ Scope of the "== HF" claim** (nuance pass, `generation_config` work, 29 Jul 2026).
> Throughout this document, "== HF" means **same argmax on the raw logits** — a stricter
> criterion than comparing two `generate()` calls, but **not** the same statement. Until
> 29 Jul the port did **not** apply `generation_config.json` (`suppress_tokens`, multiple
> EOS): the reading "reproduces what `generate()` would produce" was **false**. It became
> true **for the 12B in free-running mode** and remains **false for the E2B runners**.
> Details and figures: `docs/GENERATION_CONFIG_RESULTS.md` · `docs/FINDING_GENERATION_CONFIG.md`.
>
> **⚠ Build mode of published benchmarks (30 Jul 2026).** `-c opt` only sets the C++ backend; the
> Zig frontend mode is a separate Bazel flag. Throughput figures published before that date were
> measured in an unproven mode: exposure is **+7.8 %** on GPU-bound tok/s (the 12B does 9.6-9.7,
> not 9.0) and up to **×8.7** on a pure host timer. **Equivalence claims and paired differentials
> are unaffected.** Audit of the repo's 174 performance claims and the prevention now in place:
> `docs/MODE_BUILD_AUDIT.md`.

A bit-exact, op-by-op port of **`google/gemma-4-E2B-it`** (text path) to
**[ZML](https://github.com/zml/zml)** — the Zig + MLIR + OpenXLA inference compiler — built and
**proven against HuggingFace Transformers one operation at a time**, then grown into an autonomous
text→text engine with long-context generation, bf16 fidelity, static batching, **4-bit weights** —
and now running **Gemma 4 12B** (official QAT w4a16 checkpoint) on a single RTX 3090.

> **Status — port complete + autonomous runtime + long generation + bf16 + batching + 4-bit weights + 12B on a 24 GB GPU + resumable KV-cache + partial prefill (multi-turn).**
> Prefill, logits, single-token decode and **1020-token** generation all reproduce HuggingFace
> (token-exact in fp32; within the measured HF-bf16 envelope in bf16). The engine now runs
> **standalone on GPU** (native tokenizer, chat template, EOS early-stop), carries a modular
> decode socle (`EngineModel(comptime Brick, EngineCfg)`) with proven-neutral bricks, and its
> comptime geometry (`Geom`) runs both E2B and **12B Unified** — whose bf16 weights alone
> (~24 GB) would not fit the GPU. **~70 atomic gates**, each committed and tagged.
> Visual map of the core port: [`docs/CARTOGRAPHIE_portage.md`](docs/CARTOGRAPHIE_portage.md).
> Full documentation (capabilities, usage, method, 29 pitfalls — in French): [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

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
| **Gemma 4 12B on one 3090 (W4-J2)** | official `gemma-4-12B-it-qat-w4a16-ct` (48 layers, heterogeneous GQA/MQA with K=V full layers) decodes in ZML: **1150 tokens @ 9.0 tok/s** (**9.6-9.7 tok/s** re-measured in a proven build mode, see below), real VRAM peak **16 680 MiB** (bf16 weights alone: 24 GB — impossible); teacher-forced **== HF-fp32 STRICT, 48/48 + 1150/1150, zero requalification** (fp32-compute oracle on bf16 storage); E2B engine preserved by **byte-identical HLO** proof | `docs/U_12B_RESULTS.md` |
| **Decoding policy** (`generation_config`) | the port now applies what Google ships: `suppress_tokens` + the **3 EOS**, then host-side `top_k`/`top_p`/`temperature` and **seed-reproducible sampling** — **6 of the 8 keys**, instead of a greedy the model card does not recommend. Graph untouched (byte-identical HLO) | PR #17/#18, `docs/GENERATION_CONFIG_RESULTS.md`, `docs/SAMPLING_RESULTS.md` |
| **Zero host allocation per step (D10)** | the decode loop performs **no Zig allocator call per step** (device→host straight into a persistent buffer, top-k to the stack, hoisted call args, pre-reserved lists) — and the ban is **enforced by a permanent counter gate**, not by code review. Sampling block **3 796 → 908.7 µs/step** (0.86 % of a step). *Pinned memory hypothesis refuted by A/B* | PR #19, `docs/D10_RESULTS.md` |
| **KV-cache dump/restore** | save the state of a running generation (4 KV caches + every fed token + a self-describing manifest) into **one safetensors**, then re-implant it and continue **without re-computing the prefix**. Restoring a 3 927-position state costs **0.898 s warm / 10.8 s cold** where re-computing it costs **449.5 s** — a **×500 speedup warm, ×41 cold** (both measured, neither replaces the other). Continuation is **bit-identical, 32/32** (ids, top-5 indices *and* value bits) intra-process, and **32/32 with zero divergence** across processes. 8 gates, **11 loud refusals** each seen to fire, graph untouched (byte-identical HLO), per-step allocation ban still holds | PR #20, `docs/KVDUMP_RESULTS.md` |
| **Partial prefill** (K5, 10 Aug 2026) | resume a dumped cache **and feed a fresh prompt**: `--load-cache F --prompt "turn 2"` absorbs the new prompt as the next conversation turn at positions `step_next…`, then keeps generating. A run that inherits another process's cache answers **"Your name is Aldebaran."** — the name exists only in the restored cache. Proven **teacher-forced against HF fp32**, never in free decoding: **19/19** and **28/28** context positions match argmax-for-argmax, the latter with the **sliding window actually biting** (T=1064 > 1024). Resuming a 1004-position context costs **9.9 s against 112 s of recompute (×11.3)** — of which **3.5 s is pure dump re-reading**, a fixed cost the ×35.6 position ratio hides. 7 gates, graph untouched (byte-identical HLO, 7th chantier running). ⚠ The mordant is **chiffré, not assumed**: a cache lying by **one** position moves every logit yet flips no argmax — the gate bites from **two** positions on | `docs/K5_RESULTS.md`, `scripts/82_k5_verify.py` |
| **Technical-debt sweep** (10 Aug 2026) | four debts settled with evidence rather than assertions: `applyTopP` and `applyTemperature` now have **GPU coverage** (386 armed steps, **0 disagreement** against a differently-written f64 reference, full antecedent — and two "obvious" mutants proven **vacuous** before being coded); the truncated-dump refusal is **loud** (3 of 7 sites seen firing, the other 4 declared unverified); the README speaks **one language**, with the scope gate extended bilingually at constant strictness; and the **cold** restore is measured — ×41.4, which **requalified** a previously published ≥ ×130 estimate | `docs/SAMPLING_RESULTS.md` §7, `docs/KVDUMP_RESULTS.md` §3 |

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

Two rules were added the hard way, each after a control failed to do its job:

- **A ban with no gate will be broken** — including by the very chantier that writes it. Bans are
  now bound to an instrument (an always-on allocation counter, a measured RSS ceiling) or written
  down as *not enforced*, never left to code review.
- **Benchmarks state their build mode, proven by the binary itself.** `-c opt` only sets the C++
  backend; the Zig frontend mode is a separate Bazel flag. Two concordant measurements once
  "refuted" a build hypothesis while both arms were running debug code (×8.7 artifact). Builds go
  through `zml_runner/build_3090.sh`, every run prints `BUILD: mode=…`, and a gate log without it
  is *unrunnable*, not passing. Full audit of the repo's 174 performance claims:
  `docs/MODE_BUILD_AUDIT.md`.

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

**Decoding policy (`generation_config.json`)** — since 29 Jul 2026, the 12B runner applies the
model's declared `suppress_tokens` and its **three** `eos_token_id`, host-side:

```bash
# Discovered automatically next to the checkpoint (ONE symlink hop). Every run logs what it
# applies — and what it ignores:
#   GENCFG: <path> suppress=[258883,258882] eos=[1,106,50] ignored=[do_sample,top_k,top_p,...]
./bazel-bin/examples/rqz/gemma4_g12auto <ckpt> <tokenizer.json> --prompt "..." --max-tokens 200

--gen-config <FILE>      # force the file (a FILE, never a directory)
--no-gen-config          # restore pre-29-Jul behaviour (the GC4 counter-test instrument)
--selftest-gencfg <fix>  # replay gate GC1 — host-only, no GPU
```

⚠ **This block alone applies 2 of the 8 keys** (`suppress_tokens`, `eos_token_id`); `top_k`,
`top_p`, `temperature` and `do_sample` came with the sampling work below — **6 of 8** today, the
remaining two (`bos_token_id`, `pad_token_id`) being moot at decode time. There is **no silent
fallback**: a runner that cannot find its policy refuses to start. Details, figures and known debt:
[`docs/GENERATION_CONFIG_RESULTS.md`](docs/GENERATION_CONFIG_RESULTS.md).

**Sampling (phase 2, 29 Jul 2026)** — the 12B now runs the sampling configuration Google ships:

```bash
./bazel-bin/examples/rqz/gemma4_g12auto <ckpt> <tokenizer.json> --prompt "..." \
  --top-k 64 --top-p 0.95 --temperature 1.0 --seed 42     # reproducible: same seed, same output

--top-k N / --top-p F / --temperature F / --min-tokens-to-keep N / --seed N
--selftest-sampling <fixture>   # gate S2-U: warpers vs the REAL HF warpers, host-only, no GPU
--selftest-draw <fixture> --draws N --seed S   # gate S2-D: 10k draws on frozen logits
```

Two arming conditions, deliberately distinct: the **full host path** arms as soon as any warper is
requested; the **draw** arms *only* if `--seed` is given. Without a seed, selection stays an
`argmax` — so filtering settings never silently turn on randomness.

Guards are written **as acceptance**, never as rejection: `p <= 0 → reject` would let `NaN`
through (every comparison with `NaN` is false). `--temperature 0` is **rejected like HF**;
`T_MIN = 1e-30` is a **declared, deliberate divergence** (HF accepts `1e-45` and produces `NaN`).

Figures, the 9 known debts, and what is *not* covered:
[`docs/SAMPLING_RESULTS.md`](docs/SAMPLING_RESULTS.md).

**KV-cache dump/restore (10 Aug 2026)** — save the state of a running generation and re-implant it
later, in another process, without re-computing the prefix:

```bash
# 1. generate, then dump the state (4 KV caches + every fed token + a self-describing manifest)
./bazel-bin/examples/rqz/gemma4_g12auto <ckpt> <tok.json> \
  --prompt "..." --max-tokens 16 --dump-cache state.kvdump
#    KVDUMP: state.kvdump l_max=1280 step_next=43 fed_next=1017 ids=43 octets=880804012 xxh64_ok

# 2. ANY later process: re-implant and continue — no prefill
./bazel-bin/examples/rqz/gemma4_g12auto <ckpt> <tok.json> \
  --load-cache state.kvdump --max-tokens 32
#    KVLOAD: ... (reprise sans prefill)   KVLOAD-PERF: ... -> 1er token en 0.898s

--selftest-kvdump-io <DIR>    # gate DC1: file round-trip + built-in mutant, host-only, no GPU
--selftest-kvdump-eq <FILE>   # gate DC2: dump → restore → continuation, bit-exact, one process
scripts/74_kvdump_inspect.py  # inspect / mutate / zero-out / forge a manifest (Python side)
```

The file is **one safetensors**, readable by the stock Python `safetensors`. Its `__metadata__`
carries a checkpoint fingerprint **by content** (size + xxh64 of the weights header — never the
10 GB) and one xxh64 per tensor. **Every mismatch is a loud refusal, and each of the 11 was seen
to fire**: wrong variant, wrong checkpoint, truncated file, bad format, bad shape, inconsistent
state, no room left, and 4 flag combinations. A restore is never silently wrong.

What it costs: at 4k, reaching a 3 927-position state **by computing it** takes **449.5 s**;
restoring it takes **0.898 s** (2.62 GiB read included) — **×500**. What it proves: the
continuation is **bit-identical (32/32)** to a reference within a process, and **32/32 with zero
divergence** across processes. Deliberately *not* covered: PRNG state (so `--dump-cache` with an
armed seed is refused), `--repl`, E2B. Figures, the 5 pre-registered claims and the 8 debts:
[`docs/KVDUMP_RESULTS.md`](docs/KVDUMP_RESULTS.md).
*(“A fresh prompt on a restored cache” was listed here as out of scope until 10 Aug 2026 — it is
now the partial-prefill capability below.)*

**Partial prefill (K5, 10 Aug 2026)** — resume a dumped cache **and feed a fresh prompt**: the
conversation brick. The new prompt is absorbed as the **next turn** at positions `step_next…`,
and generation continues:

```bash
# 1. turn 1: generate and dump
./bazel-bin/examples/rqz/gemma4_g12auto <ckpt> <tok.json> \
  --prompt "My name is Aldebaran and I live in a lighthouse. Tell me a story about my home." \
  --max-tokens 16 --dump-cache state.kvdump

# 2. ANOTHER process: resume that context AND ask something new
./bazel-bin/examples/rqz/gemma4_g12auto <ckpt> <tok.json> \
  --load-cache state.kvdump --prompt "What is my name?" --max-tokens 32
#    K5: prefill partiel — contexte 49 ids + fed_next + clôture 2 + tour2 17 ids = 69 total
#    réponse : "Your name is Aldebaran."          <- the name exists only in the restored cache

--ids-only-turn2               # gate PF6: render turn 2 to ids, host-only, no GPU, no dump
python3 scripts/82_k5_verify.py  # replay all 7 gates from the versioned evidence, no GPU
```

The whole thing is **host-side**: the graph never distinguished prefill from generation (position
≡ `ctrl.step`, in-graph masks, RoPE table covering `L_MAX`), so the HLO is byte-identical — the
same md5 as six chantiers ago. Correctness is proven **teacher-forced against HF fp32**, never in
free decoding: **19/19** then **28/28** context positions match argmax-for-argmax, the latter with
the **sliding window actually biting** (T = 1064 > 1024).

What it costs: resuming a 1004-position context takes **9.9 s against 112 s of recompute (×11.3)**
— and **3.5 s of that is re-reading the dump**, a fixed cost the ×35.6 *position* ratio hides.
Both numbers are published together on purpose. What the gate does **not** catch, measured rather
than assumed: a cache lying about its length by **one** position moves every compared logit yet
flips no argmax — detection starts at **two**. Figures, the 6 pre-registered claims (one refuted
then requalified) and the 5 debts: [`docs/K5_RESULTS.md`](docs/K5_RESULTS.md).

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

Text path only — **multimodal (vision/audio) out of scope**. No continuous batching / serving, no
fast-prefill, 256K context not exercised. **Partial prefill (K5) is 12B-only, and every 12B-only
capability widens the gap with E2B** — the E2B runners still expose no logits outside the graph,
so `generation_config`, sampling, penalty, cache dump/restore and now partial prefill all stop at
the 12B boundary (tracked as K3). Chained turns re-read the whole dump each time (~3.5 s per turn
at 1 k positions): the resident multi-turn path (K4) is what removes that cost, and it is not
built. A cache that lies about its length by a **single** position is not detected by the K5 gate
(measured, `docs/K5_RESULTS.md` §3). The static-batch path assumes equal tokenized prompt
lengths. On E2B the 4-bit VRAM gain is bounded by the bf16 embeddings (expected — the brick targets
the 12B, where the linears dominate). No independent perf benchmarks beyond the reported
token-for-token gates.

**Sampling is no longer a limitation** (12B only): `top_k`/`top_p`/`temperature` + seed-reproducible
draw are implemented host-side and gated. **Neither is losing a generation's state**: it can be
dumped and re-implanted (see above). **Repetition penalty is implemented too** (10 Aug 2026,
12B): HF-compatible, host-side, bit-identical to `RepetitionPenaltyLogitsProcessor` at **0 ULP**
on four penalty values, graph unchanged (same HLO md5 with the penalty armed), zero allocation
per step, and steerable at runtime from the resident REPL (`:penalty`). The runner reproduces
**HF's own trajectory token for token** under penalty (48/48 at both 1.15 and 0.8), and the
counter-test — same fixture with the penalty disarmed — fails as it must (17/48). Measured cost:
**+1.6 % of the host-side block**, which sits below the protocol's resolution floor. All three
corruption counter-tests are seen to fail; the third one first came out **vacuous** on the
reference prompt, which turned out to say more about the antecedent than about the corruption —
the story, and the discriminating prompt that settled it (0/48, diverging at the very first
token), are in `docs/SAMPLING_RESULTS.md` §8.6. Still open, written down rather than hidden: the **E2B** runners don't expose logits so the decoding policy can't apply
there, and C/PJRT-side allocations are *bounded* (< ~450 mallocs/step, measured) rather than
counted. **`applyTopP` is no longer uncovered**: as of 10 Aug 2026 it is
gated on GPU against an independently written reference (386 armed steps, 0 disagreement, full
antecedent) — `docs/SAMPLING_RESULTS.md` §7.

On dump/restore specifically: the **PRNG state is not serialized** (dumping with an armed seed is
refused, not silently approximated), and the **8k variant compiles the same code but no gate
exercises it**. The cold-read debt is **settled** (10 Aug 2026): a cold restore takes **10.823 s**
(~0.264 GiB/s on this VM) for a **×41.4** speedup — the `≥ ×30` claim holds cold, while the earlier
`≥ ×130` estimate assumed 1 GiB/s and has been **requalified**, exactly as the pre-registered
prediction said it would have to be.

**Next (at the design stage):** an upstream-ZML flash-attention path (batch > 1) would require
paged KV; a Triton kernel is the credible route.

## License & attribution

Code: **Apache-2.0** (see [`LICENSE`](LICENSE)) — same as ZML and Gemma. © 2026 Régis Rigaud / TheCause.
The Gemma 4 model weights are distributed by Google under the
[Gemma / Apache-2.0 terms](https://huggingface.co/google/gemma-4-E2B-it) — not included here.
