"""P5.6 — PyTorch oracle + fixture : RoPE MANUELLE partielle, full_attention (layer 14, Q-path).

Dé-risque les couches full_attention (4,9,14,19,24,29,34), structurellement différentes du sliding :
- **head_dim = 512** (global_head_dim), pas 256. q_proj [4096,1536]=8×512, q_norm [512].
- **partial rotary** : rope_type=proportional, partial_rotary_factor=0.25, rope_theta=1e6.
  Les cos/sin sont **512-wide** mais 384/512 entrées = identité (cos=1, sin=0) ; seules 128 dims
  tournent (0.25×512). attention_scaling=1.0.
- `zml.nn.rope` ne couvre PAS proportional → RoPE MANUELLE : on exporte cos/sin (réels, du module
  Gemma4TextRotaryEmbedding) et le ZML applique `q*cos + rotate_half(q)*sin` à la main
  (rotate_half(x) = cat(-x[256:512], x[0:256]) sur .hd). La structure partielle est portée par
  les valeurs de cos/sin → le ZML n'a pas besoin de connaître proportional.

Pipeline oracle (Q-path layer 14, miroir Gemma4TextAttention.forward L1226-1229) :
    q_after_proj = q_proj(hidden_input).view(1,4,8,512)
    q_after_norm = q_norm(q_after_proj)                       # Gemma4RMSNorm(512, with_scale)
    q_after_rope = apply_rotary_pos_emb(q_after_norm, cos, sin, unsqueeze_dim=2)   # (x*cos)+(rotate_half(x)*sin)
    q_final      = q_after_rope.transpose(1,2)                # [1,8,4,512]

cos,sin via Gemma4TextRotaryEmbedding(layer_type="full_attention") (proportional, head_dim_key=global_head_dim).

Fixture (5 tenseurs) : hidden_input, q_proj_weight, q_norm_weight, cos_full, sin_full (inputs),
q_after_rope (oracle). + q_after_norm pour sanity inline.

Valide la TECHNIQUE (RoPE manuelle partielle, head_dim 512). Le reste du chemin full (K, QK,
softmax, context, o_proj) est identique au sliding (dims 512) — couvert par E/F mécaniquement.

Interdits P5.6 : K/V (Q-path only), QK/softmax/context, o_proj, MLP, sliding.
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
OUT_FIXTURE = ROOT / "fixtures" / "p5_6_full_qrope_layer14.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_6_full_qrope_layer14_manifest.json"

LAYER_IDX = 14
HIDDEN = 1536
N_HEADS = 8
FULL_HEAD_DIM = 512   # global_head_dim
SEQ_LEN = 4
RMS_EPS = 1e-6
PFX = f"model.language_model.layers.{LAYER_IDX}.self_attn"


def main() -> None:
    assert WEIGHTS.exists(), f"missing weights {WEIGHTS}"
    torch.manual_seed(1337)
    hidden_input = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float32)

    with safe_open(str(WEIGHTS), framework="pt") as s:
        q_proj_weight = s.get_tensor(f"{PFX}.q_proj.weight").to(torch.float32).contiguous()
        q_norm_weight = s.get_tensor(f"{PFX}.q_norm.weight").to(torch.float32).contiguous()
    assert tuple(q_proj_weight.shape) == (N_HEADS * FULL_HEAD_DIM, HIDDEN), q_proj_weight.shape
    assert tuple(q_norm_weight.shape) == (FULL_HEAD_DIM,), q_norm_weight.shape

    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    rot = Gemma4TextRotaryEmbedding(tc)
    pos = torch.arange(SEQ_LEN).unsqueeze(0)                       # [1,4]
    cos_full, sin_full = rot(hidden_input, pos, layer_type="full_attention")   # [1,4,512]
    assert tuple(cos_full.shape) == (1, SEQ_LEN, FULL_HEAD_DIM), cos_full.shape

    print("=" * 70)
    print(f"P5.6 — PyTorch oracle full_attention Q-rope (layer {LAYER_IDX}, head_dim={FULL_HEAD_DIM} partial)")
    print("=" * 70)
    ones = (cos_full[0, 2].sub(1.0).abs() < 1e-6).sum().item()
    print(f"cos_full shape {tuple(cos_full.shape)} | identité (cos==1) à pos=2 : {ones}/512 "
          f"-> {512 - ones} dims tournent (partial 0.25×512=128) | scaling={getattr(rot,'full_attention_attention_scaling')}")

    q_proj = torch.nn.Linear(HIDDEN, N_HEADS * FULL_HEAD_DIM, bias=False)
    with torch.no_grad():
        q_proj.weight.copy_(q_proj_weight)
    q_norm = Gemma4RMSNorm(FULL_HEAD_DIM, eps=RMS_EPS)
    with torch.no_grad():
        q_norm.weight.copy_(q_norm_weight)

    q_after_proj = q_proj(hidden_input).view(1, SEQ_LEN, N_HEADS, FULL_HEAD_DIM)
    q_after_norm = q_norm(q_after_proj)                                       # [1,4,8,512]
    q_after_rope = apply_rotary_pos_emb(q_after_norm, cos_full, sin_full, unsqueeze_dim=2)
    assert tuple(q_after_rope.shape) == (1, SEQ_LEN, N_HEADS, FULL_HEAD_DIM)
    assert not torch.isnan(q_after_rope).any()

    # Sanity 1 : RoPE manuelle (q*cos + rotate_half(q)*sin) == apply_rotary_pos_emb.
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    c = cos_full.unsqueeze(2); sn = sin_full.unsqueeze(2)                     # unsqueeze_dim=2
    manual = q_after_norm * c + rotate_half(q_after_norm) * sn
    man_diff = (manual - q_after_rope).abs().max().item()
    print(f"Sanity RoPE manuelle vs apply_rotary_pos_emb |diff|_max = {man_diff:.3e} (attendu 0 — formule ZML)")
    assert man_diff < 1e-6

    # Sanity 2 : pos 0 identité (cos=1,sin=0) -> q_after_rope == q_after_norm.
    pos0 = (q_after_rope[0, 0] - q_after_norm[0, 0]).abs().max().item()
    pos3 = (q_after_rope[0, 3] - q_after_norm[0, 3]).abs().max().item()
    print(f"Sanity RoPE : pos0 |Δ|={pos0:.3e} (≈0 identité) | pos3 |Δ|={pos3:.4f} (>0 active, partiel)")
    assert pos0 < 1e-6 and pos3 > 1e-3

    q_final = q_after_rope.transpose(1, 2).contiguous()                      # [1,8,4,512]

    print("\nFixed points (q_after_rope[0,pos,head0,:8]):")
    for q in [0, 3]:
        vals = q_after_rope[0, q, 0, :8].tolist()
        print(f"  q_after_rope[0,{q},0,:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")

    tensors = {
        "hidden_input": hidden_input.contiguous(),
        "q_proj_weight": q_proj_weight,
        "q_norm_weight": q_norm_weight,
        "cos_full": cos_full.contiguous(),
        "sin_full": sin_full.contiguous(),
        "q_after_norm": q_after_norm.contiguous(),     # sanity inline
        "q_after_rope": q_after_rope.contiguous(),     # oracle principal [1,4,8,512]
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print(f"\nwrote {OUT_FIXTURE}  ({sum(t.numel()*t.element_size() for t in tensors.values())} bytes)")

    manifest = {
        "source": "P5.6 PyTorch oracle full_attention Q-rope (layer 14, partial rotary manuelle)",
        "spec_refs": [
            "modeling_gemma4 L1226-1229 (q_proj.view -> q_norm -> apply_rotary_pos_emb)",
            "apply_rotary_pos_emb L770-789 ((x*cos)+(rotate_half(x)*sin)), rotate_half L763",
            "rope_parameters full_attention = {proportional, theta=1e6, partial_rotary_factor=0.25, global_head_dim=512}",
        ],
        "layer_idx": LAYER_IDX,
        "config": {"hidden": HIDDEN, "n_heads": N_HEADS, "head_dim": FULL_HEAD_DIM,
                   "seq_len": SEQ_LEN, "rms_eps": RMS_EPS, "rope": "proportional partial 0.25 theta1e6 scaling1.0"},
        "tensors": {n: {"shape": list(t.shape), "dtype": "float32"} for n, t in tensors.items()},
        "zml_pipeline_hint": [
            "q = hidden_input.dot(q_proj_weight,.h) -> reshape [1,4,8,512] withTags {.b,.s,.nh,.hd}",
            "q = rmsNorm(q,.hd,1e-6).mul(q_norm_weight.broad)",
            "rh = concat(-q.split(.hd,{256,256})[1], q.split(.hd,{256,256})[0], .hd)  # rotate_half",
            "q_rope = q.mul(cos.broad).add(rh.mul(sin.broad))   # cos/sin {.s,.hd=512}",
            "compare vs q_after_rope (oracle, avant transpose) [1,4,8,512], tol 1e-4",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "checks": {"manual_vs_apply": man_diff, "pos0_identity": pos0, "pos3_active": pos3},
        "interdits_p5_6": ["K/V (Q-path only)", "QK/softmax/context", "o_proj", "MLP", "sliding"],
        "note": "head_dim full=512 ; partial rotary porté par cos/sin (384/512 identité). zml.nn.rope inutilisable -> rope manuelle.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print("\nP5.6 oracle + fixture export PASS.")


if __name__ == "__main__":
    main()
