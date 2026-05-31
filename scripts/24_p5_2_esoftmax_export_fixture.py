"""P5.2.E.softmax — Export safetensors fixture pour la sous-gate ZML softmax.

Lit `fixtures/p5_2_e0_attention_oracle_layer15_kv13.pt` (oracle PyTorch figé par E.0)
et extrait le sous-ensemble minimal nécessaire au runner ZML E.softmax :

    scores_masked  [B, n_heads, S, S] = [1, 8, 4, 4]   (input : scores + masque additif causal)
    probs          [B, n_heads, S, S] = [1, 8, 4, 4]   (oracle attendu : softmax(scores_masked, dim=-1))

Sortie : `fixtures/p5_2_esoftmax_layer15_kv13.safetensors` + manifest JSON.

E.softmax (ZML) calculera `probs = scores_masked.softmax(.k)` (fp32) et comparera à
l'oracle `probs` (tolérance 1e-4). Périmètre : SOFTMAX UNIQUEMENT sur l'axe .k.

Indépendance de l'oracle : le `probs` de référence vient de `torch.softmax` (E.0),
le ZML utilise sa propre implémentation `Tensor.softmax`. Les deux ne partagent AUCUN
code — seul le contrat numérique (input scores_masked, output distribution) est commun.

Interdits E.softmax : context = probs @ V, toute opération sur V, dot avec V, masque
réel S=8/window=3 (testé séparément en E.mask), layer 14 full attention, softcap,
scaling 1/sqrt(head_dim).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
IN_FIXTURE = ROOT / "fixtures" / "p5_2_e0_attention_oracle_layer15_kv13.pt"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_esoftmax_layer15_kv13.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_esoftmax_layer15_kv13_manifest.json"

EXPECTED_SHAPE = (1, 8, 4, 4)
SEQ_LEN = 4


def main() -> None:
    assert IN_FIXTURE.exists(), f"missing E.0 fixture {IN_FIXTURE}"
    blob = torch.load(str(IN_FIXTURE), map_location="cpu", weights_only=False)

    scores_masked = blob["scores_masked"].to(torch.float32).contiguous()
    probs = blob["probs"].to(torch.float32).contiguous()

    for name, t in [("scores_masked", scores_masked), ("probs", probs)]:
        assert tuple(t.shape) == EXPECTED_SHAPE, f"{name} shape {tuple(t.shape)} != {EXPECTED_SHAPE}"
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype} != float32"

    # === Sanity : recompute softmax en pur PyTorch (informatif, pas bit-exact). ===
    # Reproduit l'étape 10 de E.0 : softmax(scores_masked, dim=-1, fp32).
    # On vérifie que l'oracle `probs` du .pt est bien le softmax de `scores_masked`
    # (cohérence interne du fixture E.0), à 1e-6 près.
    probs_recompute = torch.softmax(scores_masked, dim=-1, dtype=torch.float32)
    recompute_diff = (probs_recompute - probs).abs().max().item()
    print(f"Sanity recompute softmax vs oracle probs |diff|_max = {recompute_diff:.3e} "
          f"(cohérence interne fixture E.0, attendu < 1e-6)")
    assert recompute_diff < 1e-6, f"oracle probs incohérent vs softmax(scores_masked) (got {recompute_diff})"

    # === Propriétés de distribution de l'oracle (informatif, garanti par E.0). ===
    prob_sums = probs.sum(dim=-1)                     # [1, 8, 4]
    max_sum_err = (prob_sums - 1.0).abs().max().item()
    print(f"max|sum(probs, dim=-1) - 1| = {max_sum_err:.3e}  (expected < 1e-6)")
    assert max_sum_err < 1e-6, f"oracle probs ne somment pas à 1 (err {max_sum_err})"

    idx = torch.arange(SEQ_LEN)
    future = idx.view(1, SEQ_LEN) > idx.view(SEQ_LEN, 1)   # [4,4] True où k > q (futur masqué)
    future_prob_max = probs[:, :, future].abs().max().item()
    print(f"max proba sur positions futures masquées = {future_prob_max:.3e}  (expected ~0)")
    assert future_prob_max < 1e-9, f"oracle : fuite d'attention sur le futur (max {future_prob_max})"
    assert not torch.isnan(probs).any() and not torch.isinf(probs).any(), "oracle probs NaN/Inf"

    # Fixed points (informational, alignés sur E.0 / handoff).
    print("Fixed points (probs):")
    for pt in [(0, 0, 0), (0, 0, 3), (0, 7, 3)]:
        vals = probs[pt][:4].tolist()
        print(f"  probs[{','.join(map(str, pt))},:4] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print()

    print("Stats:")
    for name, t in [("scores_masked", scores_masked), ("probs", probs)]:
        # min sur scores_masked ignore finfo.min pour rester lisible.
        finite = t[t > torch.finfo(torch.float32).min / 2]
        print(
            f"  {name:<14} shape={tuple(t.shape)!s:<18} "
            f"mean={finite.mean().item(): .6e} std={finite.std().item(): .6e} "
            f"min_finite={finite.min().item(): .6e} max={t.max().item(): .6e}"
        )

    tensors = {
        "scores_masked": scores_masked,
        "probs": probs,
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.E.softmax ZML softmax fixture (slim from E.0 oracle .pt)",
        "derived_from": IN_FIXTURE.name,
        "reader_layer_idx": 15,
        "writer_layer_idx": 13,
        "layer_type": "sliding_attention",
        "scope": "SOFTMAX ONLY (axe .k) — pas de context, pas de V, pas de dot(V)",
        "mask_note": (
            "scores_masked contient le masque causal additif (finfo.min sur le futur). "
            "À S=4 < sliding_window=512 le sliding dégénère en causal (cf E.0). Le vrai "
            "sliding window est testé séparément en E.mask (S=8, window=3) — PAS ici."
        ),
        "tensors": {name: {"shape": list(t.shape), "dtype": "float32"} for name, t in tensors.items()},
        "zml_pipeline_hint": [
            "scores_masked = load -> tags {.b, .h, .q, .k}   [1,8,4,4]",
            "probs = scores_masked.softmax(.k)   (fp32, soustrait le max par ligne)   [1,8,4,4]",
            "compare probs vs oracle probs, tolérance 1e-4",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": (
            "softmax stable (sub max + exp). Le résidu vient du jitter QK propagé "
            "(~2.4e-6 en E.1), softmax peut l'amplifier légèrement mais reste << 1e-4."
        ),
        "distribution_checks": {
            "max_prob_sum_err": max_sum_err,
            "future_prob_max": future_prob_max,
        },
        "interdits_p5_2_esoftmax": [
            "context = probs @ V", "dot(V)", "toute op sur V", "E.context",
            "masque réel S=8/window=3 (E.mask)", "layer 14 full attention",
            "softcap d'attention", "scaling 1/sqrt(head_dim)",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.E.softmax fixture export PASS.")


if __name__ == "__main__":
    main()
