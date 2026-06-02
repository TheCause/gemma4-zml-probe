"""P5.6.K — PyTorch oracle + fixture : RoPE MANUELLE partielle full_attention, K-path (layer 14).

Ferme le seul gap de l'audit P5.6.closeout : P5.6 avait validé le Q full-rope ; le K full-rope est
la MÊME technique (RoPE manuelle partielle, cos/sin 512-wide identiques) appliquée à K (1 tête kv,
head_dim 512). Aucun nouveau type de complexité — gate de complétude.

Pipeline (K-path layer 14, miroir Gemma4TextAttention.forward) :
    k_after_proj = k_proj(hidden_input).view(1,4,1,512)        # 1 tête kv × 512
    k_after_norm = k_norm(k_after_proj)                         # Gemma4RMSNorm(512, with_scale)
    k_after_rope = apply_rotary_pos_emb(k_after_norm, cos, sin, unsqueeze_dim=2)   # MÊMES cos/sin que Q
Fixture : hidden_input, k_proj_weight, k_norm_weight, cos_full, sin_full (inputs), k_after_rope (oracle).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm, Gemma4TextRotaryEmbedding, apply_rotary_pos_emb,
)


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_6k_full_krope_layer14.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_6k_full_krope_layer14_manifest.json"

LAYER_IDX = 14
HIDDEN = 1536
N_KV = 1
FULL_HEAD_DIM = 512
SEQ_LEN = 4
RMS_EPS = 1e-6
PFX = f"model.language_model.layers.{LAYER_IDX}.self_attn"


def main() -> None:
    assert WEIGHTS.exists()
    torch.manual_seed(1337)
    hidden_input = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float32)

    with safe_open(str(WEIGHTS), framework="pt") as s:
        k_proj_weight = s.get_tensor(f"{PFX}.k_proj.weight").to(torch.float32).contiguous()
        k_norm_weight = s.get_tensor(f"{PFX}.k_norm.weight").to(torch.float32).contiguous()
    assert tuple(k_proj_weight.shape) == (N_KV * FULL_HEAD_DIM, HIDDEN), k_proj_weight.shape
    assert tuple(k_norm_weight.shape) == (FULL_HEAD_DIM,), k_norm_weight.shape

    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    rot = Gemma4TextRotaryEmbedding(tc)
    pos = torch.arange(SEQ_LEN).unsqueeze(0)
    cos_full, sin_full = rot(hidden_input, pos, layer_type="full_attention")

    print("=" * 70)
    print(f"P5.6.K — PyTorch oracle full_attention K-rope (layer {LAYER_IDX}, head_dim {FULL_HEAD_DIM} partial)")
    print("=" * 70)

    k_proj = torch.nn.Linear(HIDDEN, N_KV * FULL_HEAD_DIM, bias=False)
    with torch.no_grad():
        k_proj.weight.copy_(k_proj_weight)
    k_norm = Gemma4RMSNorm(FULL_HEAD_DIM, eps=RMS_EPS)
    with torch.no_grad():
        k_norm.weight.copy_(k_norm_weight)

    k_after_proj = k_proj(hidden_input).view(1, SEQ_LEN, N_KV, FULL_HEAD_DIM)
    k_after_norm = k_norm(k_after_proj)
    k_after_rope = apply_rotary_pos_emb(k_after_norm, cos_full, sin_full, unsqueeze_dim=2)
    assert tuple(k_after_rope.shape) == (1, SEQ_LEN, N_KV, FULL_HEAD_DIM)
    assert not torch.isnan(k_after_rope).any()

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    c = cos_full.unsqueeze(2); sn = sin_full.unsqueeze(2)
    manual = k_after_norm * c + rotate_half(k_after_norm) * sn
    man_diff = (manual - k_after_rope).abs().max().item()
    print(f"Sanity RoPE manuelle vs apply_rotary_pos_emb |diff|_max = {man_diff:.3e} (attendu 0)")
    assert man_diff < 1e-6
    pos0 = (k_after_rope[0, 0] - k_after_norm[0, 0]).abs().max().item()
    pos3 = (k_after_rope[0, 3] - k_after_norm[0, 3]).abs().max().item()
    print(f"pos0 |Δ|={pos0:.3e} (≈0 identité) | pos3 |Δ|={pos3:.4f} (>0 active, partiel)")
    assert pos0 < 1e-6 and pos3 > 1e-3

    print("\nFixed points (k_after_rope[0,pos,0,:8]):")
    for q in [0, 3]:
        vals = k_after_rope[0, q, 0, :8].tolist()
        print(f"  k_after_rope[0,{q},0,:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")

    tensors = {
        "hidden_input": hidden_input.contiguous(),
        "k_proj_weight": k_proj_weight,
        "k_norm_weight": k_norm_weight,
        "cos_full": cos_full.contiguous(),
        "sin_full": sin_full.contiguous(),
        "k_after_rope": k_after_rope.contiguous(),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print(f"\nwrote {OUT_FIXTURE}  ({sum(t.numel()*t.element_size() for t in tensors.values())} bytes)")

    manifest = {
        "source": "P5.6.K PyTorch oracle full_attention K-rope (layer 14, partial rotary manuelle)",
        "spec_refs": ["miroir P5.6 (Q) sur K : mêmes cos/sin full 512-wide, 1 tête kv"],
        "layer_idx": LAYER_IDX,
        "config": {"hidden": HIDDEN, "n_kv": N_KV, "head_dim": FULL_HEAD_DIM, "seq_len": SEQ_LEN,
                   "rope": "proportional partial 0.25 theta1e6 scaling1.0"},
        "tensors": {n: {"shape": list(t.shape), "dtype": "float32"} for n, t in tensors.items()},
        "zml_pipeline_hint": [
            "k = hidden_input.dot(k_proj_weight,.h) -> reshape [1,4,1,512] withTags {.b,.s,.nh,.hd}",
            "k = rmsNorm(k,.hd,1e-6).mul(k_norm_weight.broad)",
            "rh = concat(-k.split(.hd,{256,256})[1], k.split(.hd,{256,256})[0], .hd)",
            "k_rope = k.mul(cos.broad).add(rh.mul(sin.broad))   # cos/sin {.b,.s,.hd=512}",
            "compare vs k_after_rope [1,4,1,512], tol 1e-4",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "checks": {"manual_vs_apply": man_diff, "pos0_identity": pos0, "pos3_active": pos3},
        "note": "ferme le gap K-full-rope de l'audit P5.6.closeout (technique identique à P5.6 Q).",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print("\nP5.6.K oracle + fixture export PASS.")


if __name__ == "__main__":
    main()
