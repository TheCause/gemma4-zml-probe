"""G2.0 — Oracle ENVELOPPE bf16 : div(HF-bf16, HF-fp32) sur la trajectoire teacher-forcée de 46.

Cf docs/G2_BF16_FIDELITY.md. Le résultat fp32 du projet (« 1020/1020 == HF ») ne dit rien sur le
régime de production bf16 : en basse précision le « bit-à-bit » n'existe plus (kernels cuBLAS,
ordre de réduction). Avant de juger le moteur ZML en bf16 (G2.2), il faut mesurer combien
l'implémentation de RÉFÉRENCE diverge d'elle-même quand elle passe en bf16 : c'est l'enveloppe.

Trois passes, mêmes tokens forcés (`fed` de gen_long.safetensors, produit par 46) :
  - pass A : modèle hybride fp32 (MÊME builder que 46) → logits fp32 par step (memmap ~1,07 Go).
             Sanity : argmax[k] == expected[k] pour tout k (reproduit 46, sinon env changé → stop).
  - pass B : modèle NATIF bf16 (builder du 38 : set_default_dtype(bf16) + load_state_dict, AUCUN
             upcast, tête lm bf16 + softcap bf16 puis .float() — le chemin HF de production).
             Par step : max_abs(logits B−A), KL(A‖B), argmax, marge top1−top2 de A.
  - pass B2 : re-run intégral de B → déterminisme run-to-run (logits bf16 comparés bitwise en u16).

Métriques → fixtures/g2_envelope_metrics.npz + fixtures/g2_envelope_manifest.json.
Leçon capitalisée appliquée : comparer les LOGITS, pas l'argmax (greedy trop robuste).

CLI : python3 scripts/50_bf16_envelope_oracle.py   (3090, venv gemma4-probe, GPU requis).
Prérequis : gen_long.safetensors présent (sinon relancer scripts/46_gen_long_oracle.py).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

ROOT = Path("/data/gemma4-zml-probe")
WEIGHTS = ROOT / "weights" / "model.safetensors"
GEN_LONG_FIXTURE = ROOT / "gen_long.safetensors"
LOGITS_A = ROOT / "g2_logits_a_f32.npy"          # memmap [N, VOC] fp32 (~1,07 Go, régénérable)
LOGITS_B = ROOT / "g2_logits_b_bf16u16.npy"      # memmap [N, VOC] bf16-as-u16 (~0,54 Go)
OUT_METRICS = ROOT / "fixtures" / "g2_envelope_metrics.npz"
OUT_MANIFEST = ROOT / "fixtures" / "g2_envelope_manifest.json"

SEQ_LEN = 4
INPUT_IDS = [2, 105, 2048, 4095]
PREFIX = "model.language_model."
EMBPTL = "embed_tokens_per_layer.weight"

DEVICE = "cuda"


def set_submodule_attr(root, dotted, value):
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], value)


def build_hybrid_model(tc):
    """Pass A — IDENTIQUE à 46 : poids fp32 (sauf embptl bf16), embed_scale fp32. Oracle = vérité."""
    torch.set_default_dtype(torch.bfloat16)
    model = Gemma4TextModel(tc)
    model.train(False)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded = set()
    with safe_open(str(WEIGHTS), framework="pt") as s:
        for k in s.keys():
            if not k.startswith(PREFIX):
                continue
            name = k[len(PREFIX):]
            t = s.get_tensor(k)
            with torch.no_grad():
                if name in params:
                    if name == EMBPTL:
                        params[name].copy_(t)
                    else:
                        set_submodule_attr(model, name, torch.nn.Parameter(t.to(torch.float32), requires_grad=False))
                elif name in buffers:
                    set_submodule_attr(model, name, t.to(torch.float32))
                else:
                    del t
                    continue
            loaded.add(name)
            del t
    ls_missing = [n for n in buffers if n.endswith("layer_scalar") and n not in loaded]
    assert not ls_missing, f"layer_scalar buffers manquants: {ls_missing[:5]}"
    for name, buf in list(model.named_buffers()):
        if buf.dtype == torch.bfloat16:
            set_submodule_attr(model, name, buf.float())
    model.embed_tokens.embed_scale = torch.tensor(1536.0 ** 0.5, dtype=torch.float32)
    torch.set_default_dtype(torch.float32)
    return model


def build_native_bf16_model(tc):
    """Pass B — builder du 38 : tout bf16, load_state_dict direct, AUCUN upcast, embed_scale intact.

    C'est le chemin « utilisateur normal » (≈ from_pretrained(torch_dtype=bfloat16)) : ce que l'on
    mesure est le bruit que HF s'autorise à lui-même en production."""
    torch.set_default_dtype(torch.bfloat16)
    model = Gemma4TextModel(tc)
    model.train(False)
    state = {}
    with safe_open(str(WEIGHTS), framework="pt") as s:
        for k in s.keys():
            if k.startswith(PREFIX):
                state[k[len(PREFIX):]] = s.get_tensor(k)
    missing, _unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "weight" in m or m.endswith("layer_scalar")]
    assert not real_missing, real_missing[:10]
    del state
    torch.set_default_dtype(torch.float32)
    es = model.embed_tokens.embed_scale
    print(f"  embed_scale natif : dtype={es.dtype}, val={float(es):.6f} (√1536={1536.0 ** 0.5:.6f})")
    return model


def head_logits_fp32(out, lm_w, softcap):
    """Tête pass A (comme 46) : tout fp32."""
    lh = out.last_hidden_state.to(torch.float32)[:, -1, :]
    return softcap * torch.tanh((lh @ lm_w.t()) / softcap)          # [1, VOC] fp32


def head_logits_bf16(out, lm_w, softcap):
    """Tête pass B : matmul + softcap en bf16 (chemin HF natif), .float() APRÈS (pour métriques)."""
    lh = out.last_hidden_state[:, -1, :]                            # bf16
    return softcap * torch.tanh((lh @ lm_w.t()) / softcap)          # [1, VOC] bf16


def main() -> None:
    assert torch.cuda.is_available(), "GPU requis (cf docstring)"
    assert WEIGHTS.exists(), WEIGHTS
    assert GEN_LONG_FIXTURE.exists(), f"{GEN_LONG_FIXTURE} absent — relancer scripts/46_gen_long_oracle.py"
    torch.manual_seed(1337)

    with safe_open(str(GEN_LONG_FIXTURE), framework="pt") as s:
        fed = s.get_tensor("fed").tolist()
        expected = s.get_tensor("expected").tolist()
    n_decode = len(fed)
    print(f"Trajectoire 46 chargée : N_DECODE={n_decode}, fed[:4]={fed[:4]}, expected[:4]={expected[:4]}")

    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    softcap = float(getattr(tc, "final_logit_softcapping", 30.0))
    voc = int(tc.vocab_size)
    input_ids = torch.tensor(INPUT_IDS, dtype=torch.long, device=DEVICE).view(1, SEQ_LEN)

    # ---------------- pass A : hybride fp32 (vérité), logits -> memmap ----------------
    print("\n[pass A] build hybride fp32 (builder 46) ...")
    t0 = time.time()
    model = build_hybrid_model(tc).to(DEVICE)
    lm_w = model.embed_tokens.weight.to(torch.float32)
    logits_a = np.lib.format.open_memmap(str(LOGITS_A), mode="w+", dtype=np.float32, shape=(n_decode, voc))
    margin_a = np.zeros(n_decode, dtype=np.float32)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    pkv = out.past_key_values
    s0_a = int(head_logits_fp32(out, lm_w, softcap).argmax(dim=-1).item())
    assert s0_a == fed[0], f"prefill A: argmax {s0_a} != fed[0] {fed[0]} (env changé vs 46 ?)"

    mismatch_a = 0
    for k in range(n_decode):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[fed[k]]], dtype=torch.long, device=DEVICE),
                        past_key_values=pkv, use_cache=True)
        lg = head_logits_fp32(out, lm_w, softcap)                    # [1, VOC] fp32
        top2 = torch.topk(lg, 2, dim=-1).values[0]
        margin_a[k] = float(top2[0] - top2[1])
        if int(lg.argmax(dim=-1).item()) != expected[k]:
            mismatch_a += 1
        logits_a[k] = lg[0].cpu().numpy()
        if (k + 1) % 256 == 0:
            print(f"    A {k + 1}/{n_decode}")
    logits_a.flush()
    assert mismatch_a == 0, f"pass A: {mismatch_a} argmax != expected — l'env ne reproduit plus 46, STOP"
    print(f"[pass A] OK — sanity {n_decode}/{n_decode} == expected ({time.time() - t0:.0f}s)")

    del model, pkv, out, lm_w
    torch.cuda.empty_cache()

    # ---------------- pass B : natif bf16, métriques vs A ----------------
    print("\n[pass B] build natif bf16 (builder 38, zéro upcast) ...")
    t0 = time.time()
    model = build_native_bf16_model(tc).to(DEVICE)
    lm_w_b = model.embed_tokens.weight                               # bf16

    max_abs = np.zeros(n_decode, dtype=np.float32)
    kl = np.zeros(n_decode, dtype=np.float32)
    argmax_b = np.zeros(n_decode, dtype=np.int32)
    match = np.zeros(n_decode, dtype=bool)
    logits_b = np.lib.format.open_memmap(str(LOGITS_B), mode="w+", dtype=np.uint16, shape=(n_decode, voc))

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    pkv = out.past_key_values
    s0_b = int(head_logits_bf16(out, lm_w_b, softcap).float().argmax(dim=-1).item())
    prefill_match = (s0_b == fed[0])
    print(f"  prefill B : argmax={s0_b} vs fed[0]={fed[0]} → {'match' if prefill_match else 'BIFURCATION AU PREFILL'}")

    for k in range(n_decode):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[fed[k]]], dtype=torch.long, device=DEVICE),
                        past_key_values=pkv, use_cache=True)
        lg16 = head_logits_bf16(out, lm_w_b, softcap)                # [1, VOC] bf16
        logits_b[k] = lg16[0].view(torch.uint16).cpu().numpy()
        lg_b = lg16.float()
        lg_a = torch.from_numpy(np.asarray(logits_a[k])).to(DEVICE).view(1, voc)
        max_abs[k] = float((lg_b - lg_a).abs().max())
        lsa = torch.log_softmax(lg_a, dim=-1)
        lsb = torch.log_softmax(lg_b, dim=-1)
        kl[k] = float((lsa.exp() * (lsa - lsb)).sum())
        am = int(lg_b.argmax(dim=-1).item())
        argmax_b[k] = am
        match[k] = (am == expected[k])
        if (k + 1) % 256 == 0:
            print(f"    B {k + 1}/{n_decode}")
    logits_b.flush()
    n_match = int(match.sum())
    p0 = int(np.argmin(match)) if n_match < n_decode else -1        # 1re bifurcation (-1 = aucune)
    print(f"[pass B] OK ({time.time() - t0:.0f}s)")

    # ---------------- pass B2 : déterminisme run-to-run (bitwise u16) ----------------
    print("\n[pass B2] re-run natif bf16 (même process, même modèle) — bitwise vs B1 ...")
    t0 = time.time()
    steps_nondet = 0
    elems_nondet = 0
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    pkv = out.past_key_values
    for k in range(n_decode):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[fed[k]]], dtype=torch.long, device=DEVICE),
                        past_key_values=pkv, use_cache=True)
        row = head_logits_bf16(out, lm_w_b, softcap)[0].view(torch.uint16).cpu().numpy()
        diff = int((row != logits_b[k]).sum())
        if diff:
            steps_nondet += 1
            elems_nondet += diff
        if (k + 1) % 256 == 0:
            print(f"    B2 {k + 1}/{n_decode}")
    deterministic = (steps_nondet == 0)
    print(f"[pass B2] OK ({time.time() - t0:.0f}s)")

    # ---------------- synthèse ----------------
    def pct(a):
        return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)), "max": float(a.max())}

    print("\n" + "=" * 72)
    print("G2.0 — enveloppe bf16 : div(HF-bf16, HF-fp32), trajectoire teacher-forcée 46")
    print("=" * 72)
    print(f"  déterminisme B1==B2   : {'PASS (bitwise)' if deterministic else f'FAIL — {steps_nondet} steps, {elems_nondet} élts'}")
    print(f"  argmax B == expected  : {n_match}/{n_decode} ; première bifurcation p0="
          f"{'aucune' if p0 < 0 else f'step {p0} (pos {SEQ_LEN + p0}, marge A={margin_a[p0]:.4f})'}")
    print(f"  prefill               : {'match' if prefill_match else 'BIFURCATION'}")
    print(f"  max_abs(logits B−A)   : {pct(max_abs)}")
    print(f"  KL(A‖B)               : {pct(kl)}")
    lo_margin = np.sort(margin_a)[:5]
    print(f"  marges A les + basses : {[round(float(v), 4) for v in lo_margin]}")

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_METRICS, max_abs=max_abs, kl=kl, match=match, argmax_b=argmax_b, margin_a=margin_a)
    print(f"\nwrote {OUT_METRICS}")

    manifest = {
        "source": "G2.0 oracle enveloppe bf16 (docs/G2_BF16_FIDELITY.md §3)",
        "trajectory": "teacher-forced sur fed de gen_long.safetensors (46)",
        "n_decode": n_decode, "vocab": voc, "prompt": INPUT_IDS,
        "pass_a_sanity": f"{n_decode}/{n_decode} argmax == expected",
        "determinism_b1_b2": {"pass": deterministic, "steps_nondet": steps_nondet, "elems_nondet": elems_nondet,
                              "note": "même process/modèle ; ne couvre pas rebuild ni autre GPU"},
        "prefill_match": prefill_match,
        "argmax_match": n_match, "p0_first_bifurcation_step": p0,
        "p0_margin_a": None if p0 < 0 else float(margin_a[p0]),
        "max_abs": pct(max_abs), "kl_a_b": pct(kl),
        "criterion_g2_2": "div(ZML-bf16, A) <= 2 x cette enveloppe (p50/p95/max) ; bifurcation pas avant ~p0/2",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}\nG2.0 oracle enveloppe terminé.")


if __name__ == "__main__":
    main()
