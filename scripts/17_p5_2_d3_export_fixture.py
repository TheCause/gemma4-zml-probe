"""P5.2.D.3 — Export safetensors fixture pour la sous-gate ZML k_norm.

Lit `fixtures/p5_2_d0_kv_oracle_layer13.pt` (artefact figé par D.0) et
extrait un sous-ensemble minimal pour D.3 :

    hidden_input     [B,S,H]            = [1, 4, 1536]
    k_proj_weight    [n_kv*hd, H]       = [256, 1536]
    k_norm_weight    [head_dim]         = [256]
    k_after_norm     [B,S,n_kv,hd]      = [1, 4, 1, 256]   (oracle PyTorch BLAS)

Sortie : `fixtures/p5_2_d3_k_norm_layer13.safetensors` + manifest JSON.

Ne touche pas D.0. Pas de chargement modèle. Pas de RoPE. Pas de v_norm
(absent du checkpoint Gemma 4). Reduit le footprint pour le runner ZML.

Impressions stdout : blocs [0,0,0,:8], [0,3,0,:8] pour comparaison humaine.
Le runner ZML lira l'oracle au runtime, pas besoin de hardcoder.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
IN_FIXTURE = ROOT / "fixtures" / "p5_2_d0_kv_oracle_layer13.pt"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_d3_k_norm_layer13.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_d3_k_norm_layer13_manifest.json"

LAYER_IDX = 13
LAYER_TYPE = "sliding_attention"
RMS_EPS = 1.0e-6
EXPECTED_HIDDEN_SHAPE = (1, 4, 1536)
EXPECTED_K_PROJ_W_SHAPE = (256, 1536)
EXPECTED_K_NORM_W_SHAPE = (256,)
EXPECTED_K_AFTER_NORM_SHAPE = (1, 4, 1, 256)


def main() -> None:
    assert IN_FIXTURE.exists(), f"missing D.0 fixture {IN_FIXTURE}"
    blob = torch.load(str(IN_FIXTURE), map_location="cpu", weights_only=False)

    hidden_input = blob["hidden_input"]
    k_proj_weight = blob["k_proj_weight"]
    k_norm_weight = blob["k_norm_weight"]
    k_after_norm = blob["k_after_norm"]

    # Sanity shape & dtype.
    assert tuple(hidden_input.shape) == EXPECTED_HIDDEN_SHAPE, (
        f"hidden_input shape {tuple(hidden_input.shape)} "
        f"!= {EXPECTED_HIDDEN_SHAPE}"
    )
    assert tuple(k_proj_weight.shape) == EXPECTED_K_PROJ_W_SHAPE, (
        f"k_proj_weight shape {tuple(k_proj_weight.shape)} "
        f"!= {EXPECTED_K_PROJ_W_SHAPE}"
    )
    assert tuple(k_norm_weight.shape) == EXPECTED_K_NORM_W_SHAPE, (
        f"k_norm_weight shape {tuple(k_norm_weight.shape)} "
        f"!= {EXPECTED_K_NORM_W_SHAPE}"
    )
    assert tuple(k_after_norm.shape) == EXPECTED_K_AFTER_NORM_SHAPE, (
        f"k_after_norm shape {tuple(k_after_norm.shape)} "
        f"!= {EXPECTED_K_AFTER_NORM_SHAPE}"
    )
    for name, t in [
        ("hidden_input", hidden_input),
        ("k_proj_weight", k_proj_weight),
        ("k_norm_weight", k_norm_weight),
        ("k_after_norm", k_after_norm),
    ]:
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype} != float32"

    # === Fixed-point blocks (informational, for human eyeballing). ===
    print("Fixed points (k_after_norm, n_kv=1 -> kvh dim is singleton):")
    for s in (0, 3):
        block = k_after_norm[0, s, 0, :8].tolist()
        block_str = ", ".join(f"{v:.10f}" for v in block)
        print(f"  k_after_norm[0, {s}, 0, :8] = [{block_str}]")
    print()

    # Stats.
    print("Stats:")
    for name, t in [
        ("hidden_input", hidden_input),
        ("k_proj_weight", k_proj_weight),
        ("k_norm_weight", k_norm_weight),
        ("k_after_norm", k_after_norm),
    ]:
        print(
            f"  {name:<14} shape={tuple(t.shape)!s:<20} dtype={t.dtype} "
            f"mean={t.mean().item(): .6e} std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    # === Write safetensors fixture (4 tensors). ===
    tensors = {
        "hidden_input": hidden_input.contiguous(),
        "k_proj_weight": k_proj_weight.contiguous(),
        "k_norm_weight": k_norm_weight.contiguous(),
        "k_after_norm": k_after_norm.contiguous(),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.D.3 ZML k_norm-only fixture (slim from D.0)",
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
            "hidden_input [B,S,H]",
            "k_after_proj = hidden_input @ k_proj_weight.T  [B,S,n_kv*head_dim]",
            "k_4d         = k_after_proj.reshape(B,S,n_kv,head_dim).withTags(.{.b,.s,.kvh,.d})",
            "k_normalized = rmsNorm(k_4d, axis=.d, eps=1e-6)",
            "k_after_norm = k_normalized * k_norm_weight.broad(shape)   (pattern Llama, PAS Qwen)",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": "~5-10e-6 (D.1 k_proj résidu ~5e-6 amorti par RMSNorm)",
        "interdits_p5_2_d3": [
            "v_proj",
            "v_norm (absent du checkpoint Gemma 4)",
            "RoPE",
            "transpose [B,n_kv,S,head_dim]",
            "cache slot",
            "attention",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.D.3 fixture export PASS.")


if __name__ == "__main__":
    main()
