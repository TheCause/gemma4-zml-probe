"""P5.2.E.1 — Export safetensors fixture pour la sous-gate ZML QK scores.

Lit `fixtures/p5_2_e0_attention_oracle_layer15_kv13.pt` (oracle PyTorch figé par E.0)
et extrait le sous-ensemble minimal nécessaire au runner ZML E.1 (QK scores only) :

    q_final     [B, n_heads, S, hd] = [1, 8, 4, 256]   (input, reader layer 15)
    k_final     [B, n_kv,    S, hd] = [1, 1, 4, 256]   (input, writer layer 13, cache layout)
    scores_raw  [B, n_heads, S, S]  = [1, 8, 4, 4]      (oracle attendu, Q·Kᵀ * scaling=1.0)

Sortie : `fixtures/p5_2_e1_qk_scores_layer15_kv13.safetensors` + manifest JSON.

E.1 (ZML) calculera `scores = dot(q, kᵀ) * 1.0` avec broadcast GQA (n_kv 1 -> 8 heads)
et comparera à `scores_raw` (tolérance 1e-4, résidu matmul PJRT-CPU attendu ~1e-5).

Périmètre E.1 : QK scores UNIQUEMENT. Pas de masque, pas de softmax, pas de context.
Le masque/softmax/context relèvent de E.mask et des gates suivantes.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
IN_FIXTURE = ROOT / "fixtures" / "p5_2_e0_attention_oracle_layer15_kv13.pt"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_e1_qk_scores_layer15_kv13.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_e1_qk_scores_layer15_kv13_manifest.json"

EXPECTED_Q_SHAPE = (1, 8, 4, 256)
EXPECTED_K_SHAPE = (1, 1, 4, 256)
EXPECTED_SCORES_SHAPE = (1, 8, 4, 4)
SCALING = 1.0


def main() -> None:
    assert IN_FIXTURE.exists(), f"missing E.0 fixture {IN_FIXTURE}"
    blob = torch.load(str(IN_FIXTURE), map_location="cpu", weights_only=False)

    q_final = blob["q_final"].to(torch.float32).contiguous()
    k_final = blob["k_final"].to(torch.float32).contiguous()
    scores_raw = blob["scores_raw"].to(torch.float32).contiguous()

    for name, t, expected in [
        ("q_final", q_final, EXPECTED_Q_SHAPE),
        ("k_final", k_final, EXPECTED_K_SHAPE),
        ("scores_raw", scores_raw, EXPECTED_SCORES_SHAPE),
    ]:
        assert tuple(t.shape) == expected, f"{name} shape {tuple(t.shape)} != {expected}"
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype} != float32"

    # === Sanity : recompute scores_raw en pur PyTorch (informatif, pas bit-exact). ===
    # Reproduit eager_attention_forward de E.0 (repeat_kv contigu) puis Q·Kᵀ * scaling.
    # NB : le matmul CPU fp32 (réduction sur d=256, addition non-associative) introduit
    # un jitter ~5e-7 même PyTorch-vs-PyTorch -> on vérifie la cohérence à 1e-5, PAS bit-exact.
    # C'est le PLANCHER de bruit que E.1 (ZML) ne pourra pas battre ; tolérance E.1 = 1e-4.
    key_states = (
        k_final[:, :, None, :, :].expand(1, 1, 8, 4, 256).reshape(1, 8, 4, 256)
    )  # repeat_kv(k_final, 8), contigu
    scores_recompute = torch.matmul(q_final, key_states.transpose(2, 3)) * SCALING
    recompute_diff = (scores_recompute - scores_raw).abs().max().item()
    print(f"Sanity recompute Q·Kᵀ vs oracle scores_raw |diff|_max = {recompute_diff:.3e} "
          f"(jitter matmul CPU fp32, attendu < 1e-5 ; plancher de bruit pour E.1)")
    assert recompute_diff < 1e-5, f"oracle scores_raw incohérent au-delà du jitter fp32 (got {recompute_diff})"

    # Fixed points (informational, alignés sur E.0).
    print("Fixed points (scores_raw):")
    for pt in [(0, 0, 0), (0, 0, 3), (0, 7, 3)]:
        vals = scores_raw[pt][:4].tolist()
        print(f"  scores_raw[{','.join(map(str, pt))},:4] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print()

    print("Stats:")
    for name, t in [("q_final", q_final), ("k_final", k_final), ("scores_raw", scores_raw)]:
        print(
            f"  {name:<12} shape={tuple(t.shape)!s:<18} "
            f"mean={t.mean().item(): .6e} std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    tensors = {
        "q_final": q_final,
        "k_final": k_final,
        "scores_raw": scores_raw,
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.E.1 ZML QK scores fixture (slim from E.0 oracle .pt)",
        "derived_from": IN_FIXTURE.name,
        "reader_layer_idx": 15,
        "writer_layer_idx": 13,
        "layer_type": "sliding_attention",
        "scaling": SCALING,
        "scope": "QK scores ONLY — pas de mask, pas de softmax, pas de context",
        "tensors": {name: {"shape": list(t.shape), "dtype": "float32"} for name, t in tensors.items()},
        "zml_pipeline_hint": [
            "q = load q_final  -> tags {.b, .h, .s, .hd}     [1,8,4,256]",
            "k = load k_final  -> tags {.b, .kvh, .s, .hd}   [1,1,4,256]",
            "scores = dot(q, k.T sur .hd) * 1.0  avec broadcast kvh 1 -> h 8  -> {.b,.h,.s,.t}  [1,8,4,4]",
            "compare scores vs scores_raw (oracle), tolérance 1e-4",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": "matmul PJRT-CPU vs PyTorch BLAS ~1e-5 (réduction sur .hd=256)",
        "interdits_p5_2_e1": ["mask", "softmax", "context", "layer 14", "softcap", "scaling 1/sqrt(hd)"],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.E.1 fixture export PASS.")


if __name__ == "__main__":
    main()
