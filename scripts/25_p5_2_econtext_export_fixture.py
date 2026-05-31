"""P5.2.E.context — Export safetensors fixture pour la sous-gate ZML context dot.

Lit `fixtures/p5_2_e0_attention_oracle_layer15_kv13.pt` (oracle PyTorch figé par E.0)
et extrait le sous-ensemble minimal nécessaire au runner ZML E.context (dernier maillon
de l'attention : context = probs @ V) :

    probs    [B, n_heads, S, S]  = [1, 8, 4, 4]    (input, softmax déjà fait en E.softmax)
    v_final  [B, n_kv,    S, hd] = [1, 1, 4, 256]  (input, writer layer 13, V RMSNorm no-scale D.0b)
    context  [B, n_heads, S, hd] = [1, 8, 4, 256]  (oracle attendu : matmul(probs, repeat_kv(v_final,8)))

Sortie : `fixtures/p5_2_econtext_layer15_kv13.safetensors` + manifest JSON.

E.context (ZML) calculera, par GQA (split des têtes Q comme en E.1, broadcast de l'unique
tête KV) :
    probs_split = probs.splitAxis(.h, {.h = v_final.dim(.h)=1, .hq = .auto=8})  [.b,.h=1,.hq=8,.q,.k]
    context     = probs_split.dot(v_final, .k)                                  [.b,.h=1,.hq=8,.q,.hd]
    context     = context.merge({.h = {.h, .hq}})                               [.b,.h=8,.q,.hd]
    context     = context.transpose({.b,.h,.q,.hd})
et comparera à l'oracle `context` (tolérance 1e-4).

Indépendance de l'oracle : `context` de référence vient de torch.matmul (E.0) ; le ZML
utilise sa propre chaîne splitAxis/dot/merge. Aucun code partagé.

GARDE ANTI-RÉGRESSION : v_final doit être le V NORMÉ (RMSNorm sans scale, D.0b), PAS le V
brut `v_after_reshape`. On vérifie max|v_final - v_raw| ~ 0.777 (sinon le bug D.0 est revenu).

Interdits E.context : o_proj (vient APRÈS context), re-softmax, masque, scaling, softcap,
layer 14 full attention.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
IN_FIXTURE = ROOT / "fixtures" / "p5_2_e0_attention_oracle_layer15_kv13.pt"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_econtext_layer15_kv13.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_econtext_layer15_kv13_manifest.json"

EXPECTED_PROBS_SHAPE = (1, 8, 4, 4)
EXPECTED_V_SHAPE = (1, 1, 4, 256)
EXPECTED_CONTEXT_SHAPE = (1, 8, 4, 256)
NUM_KEY_VALUE_GROUPS = 8


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, n_kv, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def main() -> None:
    assert IN_FIXTURE.exists(), f"missing E.0 fixture {IN_FIXTURE}"
    blob = torch.load(str(IN_FIXTURE), map_location="cpu", weights_only=False)

    probs = blob["probs"].to(torch.float32).contiguous()
    v_final = blob["v_final"].to(torch.float32).contiguous()
    context = blob["context"].to(torch.float32).contiguous()

    for name, t, expected in [
        ("probs", probs, EXPECTED_PROBS_SHAPE),
        ("v_final", v_final, EXPECTED_V_SHAPE),
        ("context", context, EXPECTED_CONTEXT_SHAPE),
    ]:
        assert tuple(t.shape) == expected, f"{name} shape {tuple(t.shape)} != {expected}"
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype} != float32"

    # === GARDE ANTI-RÉGRESSION : V normé (D.0b), pas brut — check auto-contenu. ===
    # RMSNorm sans scale impose RMS(v_final sur hd) = sqrt(mean(x²)/(mean(x²)+eps)) ≈ 1.0
    # pour chaque position. Le V brut (non normé) n'aurait pas cette propriété.
    rms = v_final.pow(2).mean(dim=-1).sqrt()            # [1,1,4] : RMS par position
    rms_dev = (rms - 1.0).abs().max().item()
    print(f"V correction check : max|RMS(v_final, hd) - 1| = {rms_dev:.6e}  "
          f"(RMSNorm sans scale impose RMS≈1 — sinon V brut, bug D.0 revenu)")
    assert rms_dev < 1e-3, f"v_final pas RMS-normé (max dev {rms_dev}) : bug V brut D.0 revenu ?"

    # === Sanity : recompute context en pur PyTorch (cohérence interne fixture E.0). ===
    value_states = repeat_kv(v_final, NUM_KEY_VALUE_GROUPS)              # [1,8,4,256]
    assert torch.equal(value_states[0, 0], value_states[0, 7]), "repeat_kv V incohérent"
    context_recompute = torch.matmul(probs, value_states)               # [1,8,4,256]
    recompute_diff = (context_recompute - context).abs().max().item()
    print(f"Sanity recompute matmul(probs, repeat_kv(v,8)) vs oracle context |diff|_max = "
          f"{recompute_diff:.3e}  (cohérence interne fixture E.0, attendu < 1e-5)")
    assert recompute_diff < 1e-5, f"oracle context incohérent (got {recompute_diff})"

    assert not torch.isnan(context).any() and not torch.isinf(context).any(), "oracle context NaN/Inf"

    # Fixed points (alignés sur E.0).
    print("Fixed points (context):")
    for pt in [(0, 0, 0), (0, 0, 3), (0, 7, 3)]:
        vals = context[pt][:8].tolist()
        print(f"  context[{','.join(map(str, pt))},:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print()

    print("Stats:")
    for name, t in [("probs", probs), ("v_final", v_final), ("context", context)]:
        print(
            f"  {name:<10} shape={tuple(t.shape)!s:<18} "
            f"mean={t.mean().item(): .6e} std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    tensors = {
        "probs": probs,
        "v_final": v_final,
        "context": context,
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.E.context ZML context dot fixture (slim from E.0 oracle .pt)",
        "derived_from": IN_FIXTURE.name,
        "reader_layer_idx": 15,
        "writer_layer_idx": 13,
        "layer_type": "sliding_attention",
        "scope": "CONTEXT DOT ONLY (probs @ V, contraction .k) — pas de o_proj, pas de re-softmax/mask/scaling",
        "gqa_note": (
            "value_states = repeat_kv(v_final, 8) côté PyTorch. Côté ZML : split des têtes Q "
            "(.h 8 -> .h=1 batch + .hq=8) partageant l'unique tête KV de v_final (.h=1), puis "
            "dot(.k), merge(.h,.hq), transpose. Miroir exact de la GQA de E.1 (QK scores)."
        ),
        "v_source": "v_final (V RMSNorm sans scale, D.0b) — surtout PAS v_after_reshape (V brut, bug D.0)",
        "tensors": {name: {"shape": list(t.shape), "dtype": "float32"} for name, t in tensors.items()},
        "zml_pipeline_hint": [
            "probs   = load -> tags {.b, .h, .q, .k}     [1,8,4,4]",
            "v_final = load -> tags {.b, .h, .k, .hd}    [1,1,4,256]  (tête KV taggée .h, size 1)",
            "probs_split = probs.splitAxis(.h, {.h = v_final.dim(.h)=1, .hq = .auto=8})  [1,1,8,4,4]",
            "context = probs_split.dot(v_final, .k)   [1,1,8,4,256]",
            "context = context.merge({.h = {.h, .hq}})   [1,8,4,256]",
            "context = context.transpose({.b,.h,.q,.hd})",
            "compare context vs oracle context, tolérance 1e-4",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": (
            "matmul probs@V PJRT-CPU vs PyTorch BLAS (réduction .k=4). Jitter QK E.1 (~2.4e-6) "
            "+ softmax (~0) propagés, plus le résidu de ce matmul. Attendu << 1e-4."
        ),
        "checks": {"v_final_rms_dev": rms_dev, "recompute_diff": recompute_diff},
        "interdits_p5_2_econtext": [
            "o_proj (vient après context)", "re-softmax", "masque (E.mask/E.softmax)",
            "scaling 1/sqrt(head_dim)", "softcap", "layer 14 full attention",
        ],
        "completes": "P5.2.E (chaîne d'attention ZML complète : QK -> mask -> softmax -> context)",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.E.context fixture export PASS.")


if __name__ == "__main__":
    main()
