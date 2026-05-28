"""P5.2.D.4 — Export safetensors fixture pour la sous-gate ZML RoPE K.

Lit `fixtures/p5_2_d0_kv_oracle_layer13.pt` (artefact figé par D.0) et
extrait un sous-ensemble minimal pour D.4 :

    hidden_input     [B,S,H]            = [1, 4, 1536]
    k_proj_weight    [n_kv*hd, H]       = [256, 1536]
    k_norm_weight    [head_dim]         = [256]
    k_after_norm     [B,S,n_kv,hd]      = [1, 4, 1, 256]
    k_after_rope     [B,S,n_kv,hd]      = [1, 4, 1, 256]   (oracle PyTorch)

Sortie : `fixtures/p5_2_d4_k_rope_layer13.safetensors` + manifest JSON.

Ne touche pas D.0. Pas de chargement modèle. Pas de v_proj. Pas de transpose.
Reduit le footprint pour le runner ZML (5 tenseurs au lieu de 16).

`k_after_norm` est conservé dans la fixture pour permettre la sanity
inline côté runner Zig :
  - pos 0 : RoPE = identité (`|k_rope[0,*,*,:] - k_norm[0,*,*,:]| ≈ 0`)
  - pos 3 : RoPE active (`|k_rope[3,*,*,:] - k_norm[3,*,*,:]| > 1e-3`)

Impressions stdout : blocs [0,0,0,:8], [0,3,0,:8] pour comparaison humaine.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
IN_FIXTURE = ROOT / "fixtures" / "p5_2_d0_kv_oracle_layer13.pt"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_d4_k_rope_layer13.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_d4_k_rope_layer13_manifest.json"

LAYER_IDX = 13
LAYER_TYPE = "sliding_attention"
RMS_EPS = 1.0e-6
ROPE_THETA = 10000.0
EXPECTED_HIDDEN_SHAPE = (1, 4, 1536)
EXPECTED_K_PROJ_W_SHAPE = (256, 1536)
EXPECTED_K_NORM_W_SHAPE = (256,)
EXPECTED_K_AFTER_NORM_SHAPE = (1, 4, 1, 256)
EXPECTED_K_AFTER_ROPE_SHAPE = (1, 4, 1, 256)


def main() -> None:
    assert IN_FIXTURE.exists(), f"missing D.0 fixture {IN_FIXTURE}"
    blob = torch.load(str(IN_FIXTURE), map_location="cpu", weights_only=False)

    hidden_input = blob["hidden_input"]
    k_proj_weight = blob["k_proj_weight"]
    k_norm_weight = blob["k_norm_weight"]
    k_after_norm = blob["k_after_norm"]
    k_after_rope = blob["k_after_rope"]

    # Sanity shape & dtype.
    for name, t, expected in [
        ("hidden_input", hidden_input, EXPECTED_HIDDEN_SHAPE),
        ("k_proj_weight", k_proj_weight, EXPECTED_K_PROJ_W_SHAPE),
        ("k_norm_weight", k_norm_weight, EXPECTED_K_NORM_W_SHAPE),
        ("k_after_norm", k_after_norm, EXPECTED_K_AFTER_NORM_SHAPE),
        ("k_after_rope", k_after_rope, EXPECTED_K_AFTER_ROPE_SHAPE),
    ]:
        assert tuple(t.shape) == expected, (
            f"{name} shape {tuple(t.shape)} != {expected}"
        )
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype} != float32"

    # Sanity RoPE (rappel des invariants D.0).
    pos0_diff = (
        k_after_rope[0, 0, 0, :] - k_after_norm[0, 0, 0, :]
    ).abs().max().item()
    pos3_diff = (
        k_after_rope[0, 3, 0, :] - k_after_norm[0, 3, 0, :]
    ).abs().max().item()
    print(
        f"Sanity RoPE pos 0 |k_rope - k_norm|_max : {pos0_diff:.6e}  "
        f"(expected ~0 strict, identity at pos 0)"
    )
    print(
        f"Sanity RoPE pos 3 |k_rope - k_norm|_max : {pos3_diff:.6e}  "
        f"(expected > 1e-3, RoPE active)"
    )
    assert pos0_diff < 1e-6, (
        f"RoPE pos 0 should be identity but got {pos0_diff}"
    )
    assert pos3_diff > 1e-3, (
        f"RoPE pos 3 should differ from k_norm but got {pos3_diff}"
    )
    print()

    # === Fixed-point blocks (informational). ===
    print("Fixed points (k_after_rope):")
    for s in (0, 3):
        block = k_after_rope[0, s, 0, :8].tolist()
        block_str = ", ".join(f"{v:.10f}" for v in block)
        print(f"  k_after_rope[0, {s}, 0, :8] = [{block_str}]")
    print()

    # Stats.
    print("Stats:")
    for name, t in [
        ("hidden_input", hidden_input),
        ("k_proj_weight", k_proj_weight),
        ("k_norm_weight", k_norm_weight),
        ("k_after_norm", k_after_norm),
        ("k_after_rope", k_after_rope),
    ]:
        print(
            f"  {name:<14} shape={tuple(t.shape)!s:<20} dtype={t.dtype} "
            f"mean={t.mean().item(): .6e} std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    # === Write safetensors fixture (5 tensors). ===
    tensors = {
        "hidden_input": hidden_input.contiguous(),
        "k_proj_weight": k_proj_weight.contiguous(),
        "k_norm_weight": k_norm_weight.contiguous(),
        "k_after_norm": k_after_norm.contiguous(),
        "k_after_rope": k_after_rope.contiguous(),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.D.4 ZML RoPE K-only fixture (slim from D.0)",
        "derived_from": str(IN_FIXTURE.name),
        "layer_idx": LAYER_IDX,
        "layer_type": LAYER_TYPE,
        "is_writer_producer": True,
        "rms_eps": RMS_EPS,
        "rope_theta": ROPE_THETA,
        "rope_layout": "sequential (HF style, split-half)",
        "rope_scaling": "default (attention_scaling=1.0)",
        "rope_partial_rotary_factor": 1.0,
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
            "k_4d         = k_after_proj.reshape(B,S,n_kv,head_dim).withTags(.{.b,.s,.kvh,.hd})",
            "k_normalized = rmsNorm(k_4d, axis=.hd, eps=1e-6)",
            "k_after_norm = k_normalized * k_norm_weight.broad(shape)",
            "k_after_rope = zml.nn.rope(k_after_norm, null, .{ .layout=.sequential, .scaling=.{.default=.{.rope_theta=10000}} })",
        ],
        "rope_sanity": {
            "pos0_max_abs_diff_k_rope_vs_k_norm": pos0_diff,
            "pos3_max_abs_diff_k_rope_vs_k_norm": pos3_diff,
            "pos0_threshold_lt": 1e-6,
            "pos3_threshold_gt": 1e-3,
        },
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": "~ D.3 k_norm résidu ~5e-7 (RoPE orthogonale, préserve)",
        "interdits_p5_2_d4": [
            "v_proj",
            "v_norm",
            "transpose [B,n_kv,S,head_dim]",
            "cache slot",
            "attention",
            "layer 14 full attention (proportional RoPE)",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.D.4 fixture export PASS.")


if __name__ == "__main__":
    main()
