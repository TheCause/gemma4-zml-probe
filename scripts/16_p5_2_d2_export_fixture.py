"""P5.2.D.2 — Export safetensors fixture pour la sous-gate ZML v_proj.

Lit `fixtures/p5_2_d0_kv_oracle_layer13.pt` (artefact figé par D.0) et
extrait un sous-ensemble minimal pour D.2 :

    hidden_input     [B,S,H]       = [1, 4, 1536]
    v_proj_weight    [n_kv*hd, H]  = [256, 1536]
    v_after_proj     [B,S,kv]      = [1, 4, 256]   (oracle PyTorch BLAS)

Sortie : `fixtures/p5_2_d2_v_proj_layer13.safetensors` + manifest JSON.

Ne touche pas D.0. Pas de chargement modèle. Pas de RoPE. Pas de k_norm.
Pas de v_norm (Gemma 4 : V non normé). Reduit le footprint pour le runner
ZML (3 tenseurs au lieu de 16).

Impressions stdout : blocs [0,0,:8], [0,1,:8], [0,3,:8] pour comparaison
humaine (le runner ZML lira l'oracle au runtime, pas besoin de hardcoder).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
IN_FIXTURE = ROOT / "fixtures" / "p5_2_d0_kv_oracle_layer13.pt"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_d2_v_proj_layer13.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_d2_v_proj_layer13_manifest.json"

LAYER_IDX = 13
LAYER_TYPE = "sliding_attention"
EXPECTED_HIDDEN_SHAPE = (1, 4, 1536)
EXPECTED_V_PROJ_W_SHAPE = (256, 1536)
EXPECTED_V_AFTER_PROJ_SHAPE = (1, 4, 256)


def main() -> None:
    assert IN_FIXTURE.exists(), f"missing D.0 fixture {IN_FIXTURE}"
    blob = torch.load(str(IN_FIXTURE), map_location="cpu", weights_only=False)

    hidden_input = blob["hidden_input"]
    v_proj_weight = blob["v_proj_weight"]
    v_after_proj = blob["v_after_proj"]

    # Sanity shape & dtype.
    assert tuple(hidden_input.shape) == EXPECTED_HIDDEN_SHAPE, (
        f"hidden_input shape {tuple(hidden_input.shape)} "
        f"!= {EXPECTED_HIDDEN_SHAPE}"
    )
    assert tuple(v_proj_weight.shape) == EXPECTED_V_PROJ_W_SHAPE, (
        f"v_proj_weight shape {tuple(v_proj_weight.shape)} "
        f"!= {EXPECTED_V_PROJ_W_SHAPE}"
    )
    assert tuple(v_after_proj.shape) == EXPECTED_V_AFTER_PROJ_SHAPE, (
        f"v_after_proj shape {tuple(v_after_proj.shape)} "
        f"!= {EXPECTED_V_AFTER_PROJ_SHAPE}"
    )
    for name, t in [
        ("hidden_input", hidden_input),
        ("v_proj_weight", v_proj_weight),
        ("v_after_proj", v_after_proj),
    ]:
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype} != float32"

    # Sanity contre-confusion K/V : v_proj_weight ne doit PAS être ==
    # k_proj_weight (sinon on a chargé la mauvaise branche).
    k_proj_weight = blob["k_proj_weight"]
    assert not torch.equal(v_proj_weight, k_proj_weight), (
        "v_proj_weight == k_proj_weight — mauvaise branche chargée depuis D.0"
    )

    # === Fixed-point blocks (informational, for human eyeballing). ===
    print("Fixed points (v_after_proj, head 0 implicit since n_kv=1):")
    for s in (0, 1, 3):
        block = v_after_proj[0, s, :8].tolist()
        block_str = ", ".join(f"{v:.10f}" for v in block)
        print(f"  v_after_proj[0, {s}, :8] = [{block_str}]")
    print()

    # Stats over inputs/output.
    print("Stats:")
    for name, t in [
        ("hidden_input", hidden_input),
        ("v_proj_weight", v_proj_weight),
        ("v_after_proj", v_after_proj),
    ]:
        print(
            f"  {name:<14} shape={tuple(t.shape)!s:<20} dtype={t.dtype} "
            f"mean={t.mean().item(): .6e} std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    # === Write safetensors fixture (3 tensors only). ===
    tensors = {
        "hidden_input": hidden_input.contiguous(),
        "v_proj_weight": v_proj_weight.contiguous(),
        "v_after_proj": v_after_proj.contiguous(),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.D.2 ZML v_proj-only fixture (slim from D.0)",
        "derived_from": str(IN_FIXTURE.name),
        "layer_idx": LAYER_IDX,
        "layer_type": LAYER_TYPE,
        "is_writer_producer": True,
        "tensors": {
            name: {
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
            }
            for name, t in tensors.items()
        },
        "pipeline": [
            "hidden_input [B,S,H]",
            "v_after_proj = hidden_input @ v_proj_weight.T  [B,S,n_kv*head_dim]",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": "~5e-6 (matmul PJRT-CPU Eigen-like vs PyTorch BLAS, cf D.1 k_proj)",
        "interdits_p5_2_d2": [
            "k_proj",
            "k_norm",
            "v_norm (absent du checkpoint Gemma 4)",
            "RoPE",
            "reshape [B,S,n_kv,head_dim]",
            "transpose [B,n_kv,S,head_dim]",
            "cache slot",
            "attention",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.D.2 fixture export PASS.")


if __name__ == "__main__":
    main()
