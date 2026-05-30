"""P5.2.D.2b — Export safetensors fixture pour la sous-gate ZML v_norm.

Lit `fixtures/p5_2_d0_kv_oracle_layer13.pt` (oracle D.0b corrige) et extrait
le minimum pour D.2b :

    v_after_proj  [B,S,n_kv*hd] = [1, 4, 256]      (entree : V deja projete)
    v_after_norm  [B,S,n_kv,hd] = [1, 4, 1, 256]   (oracle PyTorch, RMSNorm sans scale)

Sortie : `fixtures/p5_2_d2b_v_norm_layer13.safetensors` + manifest JSON.

D.2b valide UNIQUEMENT la RMSNorm UNSCALED (with_scale=False) sur V. Pas de
v_proj (on part de v_after_proj), pas de mul(weight) (with_scale=False : pas
de poids), pas de RoPE, pas de transpose, pas de cache, pas d'attention.

Depend de D.0b : la cle `v_after_norm` n'existe dans le .pt que si l'oracle a
ete regenere apres le fix v_norm (scripts/14_kv_oracle_layer13.py). Assertion
explicite ci-dessous.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
IN_FIXTURE = ROOT / "fixtures" / "p5_2_d0_kv_oracle_layer13.pt"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_d2b_v_norm_layer13.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_d2b_v_norm_layer13_manifest.json"

LAYER_IDX = 13
LAYER_TYPE = "sliding_attention"
RMS_EPS = 1.0e-6
EXPECTED_V_AFTER_PROJ_SHAPE = (1, 4, 256)
EXPECTED_V_AFTER_NORM_SHAPE = (1, 4, 1, 256)


def main() -> None:
    assert IN_FIXTURE.exists(), f"missing D.0b fixture {IN_FIXTURE}"
    blob = torch.load(str(IN_FIXTURE), map_location="cpu", weights_only=False)

    assert "v_after_norm" in blob, (
        "v_after_norm absent du .pt — l'oracle n'a pas ete regenere en D.0b "
        "(V RMSNorm sans scale). Relancer scripts/14_kv_oracle_layer13.py."
    )
    v_after_proj = blob["v_after_proj"]
    v_after_norm = blob["v_after_norm"]

    # Sanity shape & dtype.
    assert tuple(v_after_proj.shape) == EXPECTED_V_AFTER_PROJ_SHAPE, (
        f"v_after_proj shape {tuple(v_after_proj.shape)} "
        f"!= {EXPECTED_V_AFTER_PROJ_SHAPE}"
    )
    assert tuple(v_after_norm.shape) == EXPECTED_V_AFTER_NORM_SHAPE, (
        f"v_after_norm shape {tuple(v_after_norm.shape)} "
        f"!= {EXPECTED_V_AFTER_NORM_SHAPE}"
    )
    for name, t in [("v_after_proj", v_after_proj), ("v_after_norm", v_after_norm)]:
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype} != float32"

    # Sanity D.0b : la RMSNorm sans scale DOIT modifier V (sinon regression D.0).
    # v_after_norm et v_after_proj ont le meme layout flat (1024 valeurs).
    v_proj_flat = v_after_proj.reshape(-1)
    v_norm_flat = v_after_norm.reshape(-1)
    delta = (v_norm_flat - v_proj_flat).abs().max().item()
    print(f"sanity max|v_after_norm - v_after_proj| = {delta:.6e} (doit etre > 1e-2)")
    assert delta > 1e-2, (
        "v_after_norm ~ v_after_proj : V non norme, oracle D.0b suspect"
    )

    # === Fixed-point blocks (informational, n_kv=1 -> kvh singleton). ===
    print("Fixed points (v_after_norm):")
    for s in (0, 3):
        block = v_after_norm[0, s, 0, :8].tolist()
        block_str = ", ".join(f"{v:.10f}" for v in block)
        print(f"  v_after_norm[0, {s}, 0, :8] = [{block_str}]")
    print()

    # Stats.
    print("Stats:")
    for name, t in [("v_after_proj", v_after_proj), ("v_after_norm", v_after_norm)]:
        print(
            f"  {name:<14} shape={tuple(t.shape)!s:<20} dtype={t.dtype} "
            f"mean={t.mean().item(): .6e} std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    # === Write safetensors fixture (2 tensors). ===
    tensors = {
        "v_after_proj": v_after_proj.contiguous(),
        "v_after_norm": v_after_norm.contiguous(),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.D.2b ZML v_norm-only fixture (slim from D.0b)",
        "derived_from": str(IN_FIXTURE.name),
        "layer_idx": LAYER_IDX,
        "layer_type": LAYER_TYPE,
        "is_writer_producer": True,
        "rms_eps": RMS_EPS,
        "tensors": {
            name: {
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
            }
            for name, t in tensors.items()
        },
        "pipeline": [
            "v_after_proj [B,S,n_kv*head_dim] = [1,4,256] (entree, V deja projete)",
            "v_4d         = v_after_proj.reshape(B,S,n_kv,head_dim).withTags(.{.b,.s,.kvh,.hd})",
            "v_after_norm = rmsNorm(v_4d, axis=.hd, eps=1e-6)   "
            "(UNSCALED, with_scale=False, PAS de mul(weight))",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": (
            "~1e-6 ou moins (pas de matmul amont, RMSNorm pure sur entree fp32)"
        ),
        "interdits_p5_2_d2b": [
            "mul(weight) / v_norm.weight (with_scale=False : pas de poids)",
            "v_proj (on part de v_after_proj)",
            "RoPE",
            "transpose [B,n_kv,S,head_dim]",
            "cache slot",
            "attention",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.D.2b fixture export PASS.")


if __name__ == "__main__":
    main()
