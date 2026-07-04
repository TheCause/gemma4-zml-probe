"""G2.2 — Analyse : div(ZML-bf16, HF-fp32) vs 2× l'enveloppe HF-bf16 (cf docs/G2_BF16_FIDELITY.md §3).

Compare le dump logits du bras D (gemma4_gen_long_gpu_bf16, binaire brut f32 [N,VOC]) à la vérité
pass A (g2_logits_a_f32.npy, script 50) et rend le verdict G2.2 contre les seuils chiffrés au G2.0 :
  PASS  : max_abs et KL (p50/p95/max) ≤ 2× enveloppe B  ET  1re bifurcation ≥ p0/2  ET  mismatches ≤ 2× B
  WARN  : ratio dans ]2×, 5×]  → diagnostic P5.7.5 §6 (drift concentré/lisse vs marche) avant verdict
  FAIL  : ratio > 5×, NaN/Inf, ou bifurcation quasi immédiate

CLI : python3 scripts/51_g2_2_analyze.py <zml_logits.bin> [n_steps]   (3090, venv gemma4-probe)
Prérequis : g2_logits_a_f32.npy + fixtures/g2_envelope_manifest.json (script 50).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/gemma4-zml-probe")
LOGITS_A = ROOT / "g2_logits_a_f32.npy"
ENVELOPE = ROOT / "fixtures" / "g2_envelope_manifest.json"
OUT_METRICS = ROOT / "fixtures" / "g2_2_metrics.npz"
OUT_MANIFEST = ROOT / "fixtures" / "g2_2_manifest.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pct(a):
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def main() -> None:
    assert len(sys.argv) >= 2, "Usage: 51_g2_2_analyze.py <zml_logits.bin> [n_steps]"
    zml_path = Path(sys.argv[1])
    assert zml_path.exists(), zml_path
    assert LOGITS_A.exists(), f"{LOGITS_A} absent — relancer scripts/50_bf16_envelope_oracle.py"
    env = json.loads(ENVELOPE.read_text())
    voc = int(env["vocab"])

    logits_a = np.load(LOGITS_A, mmap_mode="r")            # [N, VOC] fp32
    raw = np.memmap(zml_path, dtype=np.float32, mode="r")
    assert raw.size % voc == 0, f"taille dump {raw.size} non multiple de VOC {voc}"
    n_steps = raw.size // voc
    if len(sys.argv) >= 3:
        n_steps = min(n_steps, int(sys.argv[2]))
    logits_d = raw.reshape(-1, voc)[:n_steps]
    assert n_steps <= logits_a.shape[0], (n_steps, logits_a.shape)
    print(f"D = {zml_path.name} [{n_steps},{voc}] vs A = {LOGITS_A.name} ; enveloppe = {ENVELOPE.name}")

    max_abs = np.zeros(n_steps, dtype=np.float32)
    kl = np.zeros(n_steps, dtype=np.float32)
    match = np.zeros(n_steps, dtype=bool)
    finite = True
    for k in range(n_steps):
        a = torch.from_numpy(np.asarray(logits_a[k])).to(DEVICE)
        d = torch.from_numpy(np.asarray(logits_d[k])).to(DEVICE)
        if not torch.isfinite(d).all():
            finite = False
        max_abs[k] = float((d - a).abs().max())
        lsa = torch.log_softmax(a, dim=-1)
        lsd = torch.log_softmax(d, dim=-1)
        kl[k] = float((lsa.exp() * (lsa - lsd)).sum())
        match[k] = bool(int(d.argmax()) == int(a.argmax()))
    n_match = int(match.sum())
    first_div = int(np.argmin(match)) if n_match < n_steps else -1

    # seuils : 2× (PASS) / 5× (FAIL) de l'enveloppe B mesurée au G2.0
    env_ma, env_kl = env["max_abs"], env["kl_a_b"]
    env_mism = env["n_decode"] - env["argmax_match"]        # 4
    p0_b = env["p0_first_bifurcation_step"]                 # 21
    d_ma, d_kl = pct(max_abs), pct(kl)

    def ratio_verdict(d_v, env_v):
        r = {q: (d_v[q] / env_v[q] if env_v[q] > 0 else float("inf")) for q in ("p50", "p95", "max")}
        worst = max(r.values())
        v = "PASS" if worst <= 2.0 else ("WARN" if worst <= 5.0 else "FAIL")
        return r, worst, v

    r_ma, w_ma, v_ma = ratio_verdict(d_ma, env_ma)
    r_kl, w_kl, v_kl = ratio_verdict(d_kl, env_kl)
    v_mism = "PASS" if (n_steps - n_match) <= 2 * env_mism * n_steps / env["n_decode"] else "WARN"
    v_p0 = "PASS" if (first_div < 0 or first_div >= p0_b // 2) else "FAIL"
    v_fin = "PASS" if finite else "FAIL"
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    verdict = max((v_ma, v_kl, v_mism, v_p0, v_fin), key=lambda v: order[v])

    print("=" * 72)
    print("G2.2 — ZML gemm=bf16 vs HF-fp32, comparé à 2× l'enveloppe HF-bf16")
    print("=" * 72)
    print(f"  finitude              : {v_fin}")
    print(f"  max_abs D             : {d_ma}")
    print(f"    vs enveloppe B      : ratios {r_ma} → {v_ma} (pire {w_ma:.2f}×, seuil PASS 2×)")
    print(f"  KL(A‖D)               : {d_kl}")
    print(f"    vs enveloppe B      : ratios {r_kl} → {v_kl} (pire {w_kl:.2f}×)")
    print(f"  argmax D == argmax A  : {n_match}/{n_steps} (B : {env['argmax_match']}/{env['n_decode']}) → {v_mism}")
    print(f"  1re bifurcation       : {'aucune' if first_div < 0 else f'step {first_div}'} (B : p0={p0_b} ; seuil ≥ {p0_b // 2}) → {v_p0}")
    print(f"\n  VERDICT G2.2 : {verdict}")
    if verdict == "WARN":
        print("  → diagnostic P5.7.5 §6 requis avant d'accepter (drift concentré/lisse vs marche).")

    np.savez(OUT_METRICS, max_abs=max_abs, kl=kl, match=match)
    manifest = {
        "source": "G2.2 analyse (docs/G2_BF16_FIDELITY.md §3)",
        "zml_dump": str(zml_path), "n_steps": n_steps,
        "max_abs": d_ma, "kl_a_d": d_kl,
        "ratios_vs_envelope": {"max_abs": r_ma, "kl": r_kl},
        "argmax_match": n_match, "first_divergence_step": first_div,
        "verdicts": {"max_abs": v_ma, "kl": v_kl, "mismatches": v_mism, "p0": v_p0, "finite": v_fin},
        "verdict": verdict,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {OUT_METRICS}\nwrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
