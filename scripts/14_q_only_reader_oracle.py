"""P5.2.C.0 — PyTorch oracle Q-only reader, layer 15 sliding.

Computes q_proj -> q_norm -> RoPE for a sliding reader layer (layer 15),
on synthetic deterministic input. No K/V. No attention. No cache.

Spec refs : transformers/models/gemma4/modeling_gemma4.py
  Gemma4TextAttention.forward L811-L824 (q_proj -> view -> q_norm -> RoPE -> transpose)

Pas de chargement du modele complet : load uniquement q_proj.weight et
q_norm.weight depuis safetensors brut. Rotary instancie depuis config
(pas de poids). Memory minimale, reproductible.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextRotaryEmbedding,
    Gemma4RMSNorm,
    apply_rotary_pos_emb,
)


REPO = "google/gemma-4-E2B-it"
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "q_only_reader_layer15.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "q_only_reader_layer15_manifest.json"

LAYER_IDX = 15
LAYER_TYPE = "sliding_attention"
SEED = 1337
B, S = 1, 4


def main() -> None:
    cfg = AutoConfig.from_pretrained(REPO)
    tc = cfg.text_config

    # Sanity 1 : layer 15 is sliding reader
    actual_lt = tc.layer_types[LAYER_IDX]
    assert actual_lt == LAYER_TYPE, (
        f"layer {LAYER_IDX} expected {LAYER_TYPE}, got {actual_lt}"
    )
    first_kv = tc.num_hidden_layers - tc.num_kv_shared_layers
    assert LAYER_IDX >= first_kv, (
        f"layer {LAYER_IDX} expected reader (>= {first_kv})"
    )

    # Architecture facts
    head_dim = tc.head_dim  # sliding uses head_dim (not global_head_dim)
    n_heads = tc.num_attention_heads
    hidden_size = tc.hidden_size
    rms_eps = tc.rms_norm_eps

    print(f"layer_idx                : {LAYER_IDX}")
    print(f"layer_type               : {LAYER_TYPE}")
    print(f"is_reader                : True (first_kv_shared = {first_kv})")
    print(f"head_dim                 : {head_dim}")
    print(f"num_attention_heads      : {n_heads}")
    print(f"hidden_size              : {hidden_size}")
    print(f"rms_norm_eps             : {rms_eps}")
    print(f"q_proj out features      : {n_heads * head_dim}")
    print()

    # Load q_proj + q_norm weights from raw safetensors (no full model).
    # Note : checkpoint full multi-modal -> prefixe `model.language_model.`
    # (cf P5.0 weight map). Layer 15 K/V/k_norm/v_norm sont presents sur disque
    # bien que non instancies en Python runtime (P5.0 § 4 nota bene).
    q_proj_key = f"model.language_model.layers.{LAYER_IDX}.self_attn.q_proj.weight"
    q_norm_key = f"model.language_model.layers.{LAYER_IDX}.self_attn.q_norm.weight"

    with safe_open(WEIGHTS, framework="pt", device="cpu") as f:
        q_proj_weight = f.get_tensor(q_proj_key).float()
        q_norm_weight = f.get_tensor(q_norm_key).float()

    print(f"q_proj_weight shape      : {tuple(q_proj_weight.shape)}")
    print(f"q_norm_weight shape      : {tuple(q_norm_weight.shape)}")
    assert q_proj_weight.shape == (n_heads * head_dim, hidden_size), (
        f"q_proj_weight shape {tuple(q_proj_weight.shape)} != ({n_heads * head_dim}, {hidden_size})"
    )
    assert q_norm_weight.shape == (head_dim,), (
        f"q_norm_weight shape {tuple(q_norm_weight.shape)} != ({head_dim},)"
    )

    # Build minimal modules with loaded weights
    q_proj = torch.nn.Linear(hidden_size, n_heads * head_dim, bias=False)
    q_proj.weight = torch.nn.Parameter(q_proj_weight)

    q_norm = Gemma4RMSNorm(head_dim, eps=rms_eps)
    q_norm.weight = torch.nn.Parameter(q_norm_weight)

    # Rotary embedding (pas de poids, pure compute from config)
    rotary = Gemma4TextRotaryEmbedding(tc)

    # Synthetic deterministic input (no K/V flow, no real prompt)
    torch.manual_seed(SEED)
    hidden = torch.randn(B, S, hidden_size, dtype=torch.float32)
    print(
        f"hidden input             : shape={tuple(hidden.shape)} "
        f"sum={hidden.sum().item():.6f}"
    )
    print()

    # === Q-only pipeline ===
    with torch.no_grad():
        # Step A : q_proj (Gemma4TextAttention.forward L821)
        q_after_proj = q_proj(hidden)  # [B, S, n_heads*head_dim]
        print(f"A) q_after_proj shape    : {tuple(q_after_proj.shape)}")

        # Step B : view + q_norm (L821 .view, L822 q_norm)
        q_view = q_after_proj.view(B, S, n_heads, head_dim)
        q_after_norm = q_norm(q_view)  # RMSNorm on last dim
        print(f"B) q_after_norm shape    : {tuple(q_after_norm.shape)}")

        # Step C : RoPE (L823)
        position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0)  # [1, S]
        cos, sin = rotary(hidden, position_ids, layer_type=LAYER_TYPE)
        print(f"C) cos/sin shape         : {tuple(cos.shape)}")
        q_after_rope = apply_rotary_pos_emb(
            q_after_norm, cos, sin, unsqueeze_dim=2
        )
        print(f"   q_after_rope shape    : {tuple(q_after_rope.shape)}")

        # Step D : transpose (L824)
        q_final = q_after_rope.transpose(1, 2).contiguous()
        print(f"D) q_final shape         : {tuple(q_final.shape)} (transposed)")

    # Sample values (head 0, position 0)
    print()
    print("Sample values (head 0, position 0, first 4 dims):")
    print(f"  q_after_proj[0, 0, 0:4]    = {q_after_proj[0, 0, 0:4].tolist()}")
    print(f"  q_after_norm[0, 0, 0, 0:4] = {q_after_norm[0, 0, 0, 0:4].tolist()}")
    print(f"  q_after_rope[0, 0, 0, 0:4] = {q_after_rope[0, 0, 0, 0:4].tolist()}")
    print(f"  q_final[0, 0, 0, 0:4]      = {q_final[0, 0, 0, 0:4].tolist()}")
    print()
    print("Sanity stats:")
    for name, t in [
        ("q_after_proj", q_after_proj),
        ("q_after_norm", q_after_norm),
        ("q_after_rope", q_after_rope),
        ("q_final", q_final),
    ]:
        print(
            f"  {name:<14} mean={t.mean().item(): .6e} "
            f"std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    # Serialize fixture
    tensors = {
        "hidden_input": hidden,
        "q_proj_weight": q_proj_weight,
        "q_norm_weight": q_norm_weight,
        "rotary_cos": cos,
        "rotary_sin": sin,
        "q_after_proj": q_after_proj,
        "q_after_norm": q_after_norm,
        "q_after_rope": q_after_rope,
        "q_final": q_final,
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes)")

    manifest = {
        "source": "P5.2.C.0 PyTorch oracle Q-only reader layer 15 sliding",
        "spec_refs": [
            "transformers/models/gemma4/modeling_gemma4.py L811-L824 "
            "(Gemma4TextAttention.forward Q path)"
        ],
        "layer_idx": LAYER_IDX,
        "layer_type": LAYER_TYPE,
        "is_reader": True,
        "seed": SEED,
        "batch_size": B,
        "seq_len": S,
        "config": {
            "num_hidden_layers": tc.num_hidden_layers,
            "num_kv_shared_layers": tc.num_kv_shared_layers,
            "first_kv_shared_layer_idx": first_kv,
            "head_dim": head_dim,
            "num_attention_heads": n_heads,
            "hidden_size": hidden_size,
            "rms_norm_eps": rms_eps,
            "sliding_window": tc.sliding_window,
            "rope_parameters_sliding": tc.rope_parameters.get(LAYER_TYPE, {}),
        },
        "tensors": {
            name: {
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
            }
            for name, t in tensors.items()
        },
        "pipeline": [
            "hidden_input [B,S,H]",
            "A) q_after_proj = q_proj(hidden_input) [B,S,n_heads*head_dim]",
            "B) q_view = q_after_proj.view(B,S,n_heads,head_dim) ; "
            "q_after_norm = q_norm(q_view) [B,S,n_heads,head_dim]",
            "C) (cos, sin) = rotary(hidden_input, position_ids=arange(S), "
            "layer_type='sliding_attention') ; "
            "q_after_rope = apply_rotary_pos_emb(q_after_norm, cos, sin, "
            "unsqueeze_dim=2)",
            "D) q_final = q_after_rope.transpose(1,2).contiguous() "
            "[B,n_heads,S,head_dim]",
        ],
        "interdits_p5_2_c0": [
            "k_proj", "v_proj", "k_norm", "v_norm",
            "attention scores", "matmul QK", "softmax",
            "cache", "sliding mask",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.C.0 PASS: PyTorch oracle for Q-only reader layer 15 (sliding) generated.")


if __name__ == "__main__":
    main()
