"""P5.2.D.0 — PyTorch oracle K/V producer + writer sliding, layer 13.

Computes k_proj / v_proj -> view -> k_norm -> RoPE(K) on a sliding writer
layer (layer 13), on synthetic deterministic input. Q path absent.
Attention scores absent. QK matmul absent. Softmax absent. Cache absent.

Spec refs : transformers/models/gemma4/modeling_gemma4.py
  Gemma4TextAttention.forward L811-L829 (k_proj / v_proj -> view -> k_norm
  -> RoPE(K) -> transpose, V not normed, V not roped)

Pas de chargement du modele complet : load uniquement k_proj.weight,
v_proj.weight et k_norm.weight depuis safetensors brut. Verifie que
v_norm n'est pas present dans le checkpoint (Gemma 4 : Q et K seulement).
Rotary instancie depuis config (pas de poids). Reproductibilite C.0 :
meme seed, meme hidden_input.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextRotaryEmbedding,
    Gemma4RMSNorm,
    apply_rotary_pos_emb,
)


REPO = "google/gemma-4-E2B-it"
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_d0_kv_oracle_layer13.pt"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_d0_kv_oracle_layer13_manifest.json"

LAYER_IDX = 13
LAYER_TYPE = "sliding_attention"
SEED = 1337
B, S = 1, 4


def stats(name: str, t: torch.Tensor) -> dict:
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "min": t.min().item(),
        "max": t.max().item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "sum": t.sum().item(),
    }


def print_stats(s: dict) -> None:
    print(
        f"  {s['name']:<18} shape={tuple(s['shape'])!s:<20} "
        f"dtype={s['dtype']:<7} "
        f"mean={s['mean']: .6e} std={s['std']: .6e} "
        f"min={s['min']: .6e} max={s['max']: .6e} "
        f"sum={s['sum']: .6e}"
    )


def main() -> None:
    cfg = AutoConfig.from_pretrained(REPO)
    tc = cfg.text_config

    # Sanity 1 : layer 13 is sliding producer (writer) — pre-shared boundary
    actual_lt = tc.layer_types[LAYER_IDX]
    assert actual_lt == LAYER_TYPE, (
        f"layer {LAYER_IDX} expected {LAYER_TYPE}, got {actual_lt}"
    )
    first_kv = tc.num_hidden_layers - tc.num_kv_shared_layers
    assert LAYER_IDX < first_kv, (
        f"layer {LAYER_IDX} expected producer/writer (< {first_kv})"
    )

    # Architecture facts (sliding uses head_dim, not global_head_dim).
    head_dim = tc.head_dim
    n_heads = tc.num_attention_heads
    n_kv = tc.num_key_value_heads
    hidden_size = tc.hidden_size
    rms_eps = tc.rms_norm_eps

    print(f"layer_idx                : {LAYER_IDX}")
    print(f"layer_type               : {LAYER_TYPE}")
    print(f"is_writer (producer)     : True (first_kv_shared = {first_kv})")
    print(f"head_dim                 : {head_dim}")
    print(f"num_attention_heads      : {n_heads}")
    print(f"num_key_value_heads      : {n_kv}")
    print(f"hidden_size              : {hidden_size}")
    print(f"rms_norm_eps             : {rms_eps}")
    print(f"k_proj/v_proj out feats  : {n_kv * head_dim}")
    print()

    # Sanity 2 : check which K/V-related weights actually exist on disk.
    prefix = f"model.language_model.layers.{LAYER_IDX}.self_attn."
    keys_of_interest = [
        "k_proj.weight",
        "v_proj.weight",
        "k_norm.weight",
        "v_norm.weight",
    ]
    presence = {}
    with safe_open(WEIGHTS, framework="pt", device="cpu") as f:
        all_keys = set(f.keys())
        for suffix in keys_of_interest:
            key = prefix + suffix
            presence[suffix] = key in all_keys

    print("Checkpoint key presence (layer 13 self_attn) :")
    for suffix, present in presence.items():
        print(f"  {suffix:<14}  present={present}")
    print()
    # Hard contract for Gemma 4 K/V path : k_proj, v_proj, k_norm yes ;
    # v_norm absent. If a future Gemma variant ships v_norm we want to fail
    # loud here rather than silently misuse it.
    assert presence["k_proj.weight"], "missing k_proj.weight"
    assert presence["v_proj.weight"], "missing v_proj.weight"
    assert presence["k_norm.weight"], "missing k_norm.weight"
    assert not presence["v_norm.weight"], (
        "v_norm.weight found on disk — D.0 contract says K-norm only ; "
        "investigate before using"
    )

    # Load K/V projection + k_norm weights from raw safetensors (no full model).
    k_proj_key = prefix + "k_proj.weight"
    v_proj_key = prefix + "v_proj.weight"
    k_norm_key = prefix + "k_norm.weight"

    with safe_open(WEIGHTS, framework="pt", device="cpu") as f:
        k_proj_weight = f.get_tensor(k_proj_key).float()
        v_proj_weight = f.get_tensor(v_proj_key).float()
        k_norm_weight = f.get_tensor(k_norm_key).float()

    print(f"k_proj_weight shape      : {tuple(k_proj_weight.shape)}")
    print(f"v_proj_weight shape      : {tuple(v_proj_weight.shape)}")
    print(f"k_norm_weight shape      : {tuple(k_norm_weight.shape)}")
    assert k_proj_weight.shape == (n_kv * head_dim, hidden_size), (
        f"k_proj_weight shape {tuple(k_proj_weight.shape)} "
        f"!= ({n_kv * head_dim}, {hidden_size})"
    )
    assert v_proj_weight.shape == (n_kv * head_dim, hidden_size), (
        f"v_proj_weight shape {tuple(v_proj_weight.shape)} "
        f"!= ({n_kv * head_dim}, {hidden_size})"
    )
    assert k_norm_weight.shape == (head_dim,), (
        f"k_norm_weight shape {tuple(k_norm_weight.shape)} != ({head_dim},)"
    )

    # Build minimal modules with loaded weights.
    k_proj = torch.nn.Linear(hidden_size, n_kv * head_dim, bias=False)
    k_proj.weight = torch.nn.Parameter(k_proj_weight)

    v_proj = torch.nn.Linear(hidden_size, n_kv * head_dim, bias=False)
    v_proj.weight = torch.nn.Parameter(v_proj_weight)

    k_norm = Gemma4RMSNorm(head_dim, eps=rms_eps)
    k_norm.weight = torch.nn.Parameter(k_norm_weight)

    # Rotary embedding (pas de poids, pure compute from config).
    rotary = Gemma4TextRotaryEmbedding(tc)

    # Synthetic deterministic input — same recipe as C.0 (seed=1337, B=1, S=4).
    torch.manual_seed(SEED)
    hidden = torch.randn(B, S, hidden_size, dtype=torch.float32)
    print(
        f"hidden input             : shape={tuple(hidden.shape)} "
        f"sum={hidden.sum().item():.6f}"
    )
    print()

    # === K/V pipeline ===
    with torch.no_grad():
        # Step A : k_proj / v_proj (Gemma4TextAttention.forward L821-L822).
        # Raw linear output is [B, S, n_kv*head_dim] = [1, 4, 256] for layer 13.
        k_after_proj = k_proj(hidden)
        v_after_proj = v_proj(hidden)
        print(f"A) k_after_proj shape    : {tuple(k_after_proj.shape)}")
        print(f"   v_after_proj shape    : {tuple(v_after_proj.shape)}")

        # Step B : view to [B, S, n_kv, head_dim] = [1, 4, 1, 256].
        k_after_reshape = k_after_proj.view(B, S, n_kv, head_dim)
        v_after_reshape = v_after_proj.view(B, S, n_kv, head_dim)
        print(f"B) k_after_reshape shape : {tuple(k_after_reshape.shape)}")
        print(f"   v_after_reshape shape : {tuple(v_after_reshape.shape)}")

        # Step C : k_norm (RMSNorm Llama pure-weight pattern, last-dim head_dim).
        # V is NOT normed in Gemma 4 (no v_norm in checkpoint, see Sanity 2).
        k_after_norm = k_norm(k_after_reshape)
        print(f"C) k_after_norm shape    : {tuple(k_after_norm.shape)}")

        # Step D : RoPE on K only (V is not roped in Gemma 4).
        # Sliding layer => rotary with layer_type='sliding_attention',
        # theta=10000 (cf rope_parameters[sliding_attention]).
        position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0)  # [1, S]
        cos, sin = rotary(hidden, position_ids, layer_type=LAYER_TYPE)
        print(f"D) cos/sin shape         : {tuple(cos.shape)}")
        k_after_rope = apply_rotary_pos_emb(
            k_after_norm, cos, sin, unsqueeze_dim=2
        )
        print(f"   k_after_rope shape    : {tuple(k_after_rope.shape)}")

        # Step E : optional transpose to [B, n_kv, S, head_dim] = [1, 1, 4, 256].
        # Matches Gemma4TextAttention output layout before the cache write.
        k_final = k_after_rope.transpose(1, 2).contiguous()
        v_final = v_after_reshape.transpose(1, 2).contiguous()
        print(f"E) k_final shape         : {tuple(k_final.shape)} (transposed)")
        print(f"   v_final shape         : {tuple(v_final.shape)} (transposed)")

    print()

    # === Sanity 3 : RoPE identity at position 0, modification at position 3 ===
    # cos(0)=1, sin(0)=0 => K_rope[pos=0] must equal K_norm[pos=0] bit-exact.
    pos0_diff = (
        k_after_rope[0, 0, 0, :] - k_after_norm[0, 0, 0, :]
    ).abs().max().item()
    pos3_diff = (
        k_after_rope[0, 3, 0, :] - k_after_norm[0, 3, 0, :]
    ).abs().max().item()
    print(
        f"RoPE pos 0 |k_rope - k_norm|_max  : {pos0_diff:.6e}  "
        f"(expected ~0, identity at pos 0)"
    )
    print(
        f"RoPE pos 3 |k_rope - k_norm|_max  : {pos3_diff:.6e}  "
        f"(expected > 1e-3, RoPE active)"
    )
    assert pos0_diff < 1e-6, (
        f"RoPE pos 0 should be identity but got max abs diff {pos0_diff}"
    )
    assert pos3_diff > 1e-3, (
        f"RoPE pos 3 should differ from k_norm but got max abs diff {pos3_diff}"
    )

    print()

    # === Fixed points : first 8 dims at pos 0 and pos 3 ===
    print("Fixed points (head 0):")
    print(
        f"  k_after_proj   [0,0,:8]    = {k_after_proj[0, 0, :8].tolist()}"
    )
    print(
        f"  k_after_proj   [0,3,:8]    = {k_after_proj[0, 3, :8].tolist()}"
    )
    print(
        f"  v_after_proj   [0,0,:8]    = {v_after_proj[0, 0, :8].tolist()}"
    )
    print(
        f"  v_after_proj   [0,3,:8]    = {v_after_proj[0, 3, :8].tolist()}"
    )
    print(
        f"  k_after_norm   [0,0,0,:8]  = {k_after_norm[0, 0, 0, :8].tolist()}"
    )
    print(
        f"  k_after_norm   [0,3,0,:8]  = {k_after_norm[0, 3, 0, :8].tolist()}"
    )
    print(
        f"  k_after_rope   [0,0,0,:8]  = {k_after_rope[0, 0, 0, :8].tolist()}"
    )
    print(
        f"  k_after_rope   [0,3,0,:8]  = {k_after_rope[0, 3, 0, :8].tolist()}"
    )
    print(
        f"  v_after_reshape[0,0,0,:8]  = {v_after_reshape[0, 0, 0, :8].tolist()}"
    )
    print(
        f"  v_after_reshape[0,3,0,:8]  = {v_after_reshape[0, 3, 0, :8].tolist()}"
    )
    print(
        f"  k_final        [0,0,0,:8]  = {k_final[0, 0, 0, :8].tolist()}  "
        f"(transposed : head 0, pos 0)"
    )
    print(
        f"  v_final        [0,0,0,:8]  = {v_final[0, 0, 0, :8].tolist()}  "
        f"(transposed : head 0, pos 0)"
    )
    print()

    # === Stats over every tensor ===
    print("Sanity stats:")
    stat_list = []
    for name, t in [
        ("hidden_input", hidden),
        ("k_after_proj", k_after_proj),
        ("v_after_proj", v_after_proj),
        ("k_after_reshape", k_after_reshape),
        ("v_after_reshape", v_after_reshape),
        ("k_after_norm", k_after_norm),
        ("k_after_rope", k_after_rope),
        ("k_final", k_final),
        ("v_final", v_final),
    ]:
        s = stats(name, t)
        print_stats(s)
        stat_list.append(s)
    print()

    # === Serialize fixture (torch .pt) ===
    fixture = {
        "layer_idx": LAYER_IDX,
        "layer_type": LAYER_TYPE,
        "input_ids": None,
        "hidden_input": hidden,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "k_norm_weight": k_norm_weight,
        "rotary_cos": cos,
        "rotary_sin": sin,
        "k_after_proj": k_after_proj,
        "v_after_proj": v_after_proj,
        "k_after_reshape": k_after_reshape,
        "v_after_reshape": v_after_reshape,
        "k_after_norm": k_after_norm,
        "k_after_rope": k_after_rope,
        "k_final": k_final,
        "v_final": v_final,
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, str(OUT_FIXTURE))
    total_bytes = sum(
        t.numel() * t.element_size() for t in fixture.values()
        if isinstance(t, torch.Tensor)
    )
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.D.0 PyTorch oracle K/V producer+writer layer 13 sliding",
        "spec_refs": [
            "transformers/models/gemma4/modeling_gemma4.py L811-L829 "
            "(Gemma4TextAttention.forward K/V path : k_proj/v_proj -> view "
            "-> k_norm -> apply_rotary_pos_emb on K only -> transpose ; "
            "V not normed, V not roped)"
        ],
        "layer_idx": LAYER_IDX,
        "layer_type": LAYER_TYPE,
        "is_writer_producer": True,
        "seed": SEED,
        "batch_size": B,
        "seq_len": S,
        "config": {
            "num_hidden_layers": tc.num_hidden_layers,
            "num_kv_shared_layers": tc.num_kv_shared_layers,
            "first_kv_shared_layer_idx": first_kv,
            "head_dim": head_dim,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv,
            "hidden_size": hidden_size,
            "rms_norm_eps": rms_eps,
            "sliding_window": tc.sliding_window,
            "rope_parameters_sliding": tc.rope_parameters.get(LAYER_TYPE, {}),
        },
        "checkpoint_key_presence": presence,
        "tensors": {
            name: {
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
            }
            for name, t in fixture.items() if isinstance(t, torch.Tensor)
        },
        "stats": stat_list,
        "rope_sanity": {
            "pos0_max_abs_diff_k_rope_vs_k_norm": pos0_diff,
            "pos3_max_abs_diff_k_rope_vs_k_norm": pos3_diff,
            "pos0_threshold_lt": 1e-6,
            "pos3_threshold_gt": 1e-3,
        },
        "pipeline": [
            "hidden_input [B,S,H]",
            "A) k_after_proj = k_proj(hidden_input) [B,S,n_kv*head_dim]",
            "   v_after_proj = v_proj(hidden_input) [B,S,n_kv*head_dim]",
            "B) k_after_reshape = k_after_proj.view(B,S,n_kv,head_dim) "
            "[B,S,n_kv,head_dim]",
            "   v_after_reshape = v_after_proj.view(B,S,n_kv,head_dim) "
            "[B,S,n_kv,head_dim]",
            "C) k_after_norm = k_norm(k_after_reshape) [B,S,n_kv,head_dim] "
            "(V not normed in Gemma 4)",
            "D) (cos, sin) = rotary(hidden_input, position_ids=arange(S), "
            "layer_type='sliding_attention') ; "
            "k_after_rope = apply_rotary_pos_emb(k_after_norm, cos, sin, "
            "unsqueeze_dim=2) (V not roped)",
            "E) k_final = k_after_rope.transpose(1,2).contiguous() "
            "[B,n_kv,S,head_dim] ; "
            "v_final = v_after_reshape.transpose(1,2).contiguous() "
            "[B,n_kv,S,head_dim]",
        ],
        "interdits_p5_2_d0": [
            "q_proj", "q_norm",
            "attention scores", "matmul QK", "softmax",
            "cache", "sliding mask",
            "layer 14 (full attention)",
            "p-RoPE proportional",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print(
        "P5.2.D.0 PASS: PyTorch oracle for K/V producer+writer "
        "layer 13 (sliding) generated."
    )


if __name__ == "__main__":
    main()
