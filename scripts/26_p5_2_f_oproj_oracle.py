"""P5.2.F — PyTorch oracle + fixture pour o_proj (projection de sortie de l'attention).

Dernier maillon du BLOC attention (après context, avant le résiduel/MLP). Reproduit la fin
de `Gemma4TextAttention.forward` (modeling_gemma4.py, transformers 5.9.0, L1272-1273) :

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()   # [b, q, n_heads*head_dim]
    attn_output = self.o_proj(attn_output)                             # nn.Linear, bias=False

ATTENTION (discipline oracle = source de vérité) :
- `eager_attention_forward` (L833-834) renvoie `matmul(probs, value_states).transpose(1, 2)`,
  soit [b, q, n_heads, head_dim]. Notre fixture E.0 stocke `context` AVANT ce transpose, en
  [b, n_heads, q, head_dim] = [1, 8, 4, 256]. Il faut donc transposer (1,2) ici.
- `o_proj` = `nn.Linear` SIMPLE (PAS `Gemma4ClippableLinear` — ça c'est l'attention VISION).
  `attention_bias = False` → pas de biais. Poids `o_proj.weight` [hidden=1536, n_heads*head_dim=2048].
- o_proj de la layer 15 (le reader dont on a calculé l'attention en E ; Q de 15, KV de 13).

Pipeline oracle (depuis le `context` de E.0) :
    attn_output = context.transpose(1, 2)                  [1,8,4,256] -> [1,4,8,256]
    attn_output = attn_output.reshape(1, 4, 2048)          concat des têtes (h-major, hd-mineur)
    o_out       = F.linear(attn_output, o_proj_weight)     [1,4,2048] @ [1536,2048]^T -> [1,4,1536]

Fixture exportée (3 tenseurs) : `context` (input), `o_proj_weight` (input), `o_proj_out` (oracle).
Le ZML fera transpose -> merge(têtes) -> dot et comparera à `o_proj_out` (tolérance 1e-4).

Interdits P5.2.F : résiduel, post_attention_layernorm, MLP, layer 14, re-attention.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
E0_FIXTURE = ROOT / "fixtures" / "p5_2_e0_attention_oracle_layer15_kv13.pt"
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_f_oproj_layer15.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_f_oproj_layer15_manifest.json"

LAYER_IDX = 15
HIDDEN = 1536
N_HEADS = 8
HEAD_DIM = 256
SEQ_LEN = 4
M = N_HEADS * HEAD_DIM  # 2048
O_PROJ_KEY = f"model.language_model.layers.{LAYER_IDX}.self_attn.o_proj.weight"

EXPECTED_CONTEXT_SHAPE = (1, N_HEADS, SEQ_LEN, HEAD_DIM)   # [1,8,4,256]
EXPECTED_W_SHAPE = (HIDDEN, M)                            # [1536,2048]
EXPECTED_OUT_SHAPE = (1, SEQ_LEN, HIDDEN)                 # [1,4,1536]


def main() -> None:
    assert E0_FIXTURE.exists(), f"missing E.0 fixture {E0_FIXTURE}"
    assert WEIGHTS.exists(), f"missing weights {WEIGHTS}"

    blob = torch.load(str(E0_FIXTURE), map_location="cpu", weights_only=False)
    context = blob["context"].to(torch.float32).contiguous()
    assert tuple(context.shape) == EXPECTED_CONTEXT_SHAPE, f"context {tuple(context.shape)} != {EXPECTED_CONTEXT_SHAPE}"

    with safe_open(str(WEIGHTS), framework="pt") as s:
        assert O_PROJ_KEY in s.keys(), f"missing {O_PROJ_KEY} in checkpoint"
        o_proj_weight = s.get_tensor(O_PROJ_KEY).to(torch.float32).contiguous()
    assert tuple(o_proj_weight.shape) == EXPECTED_W_SHAPE, f"o_proj_weight {tuple(o_proj_weight.shape)} != {EXPECTED_W_SHAPE}"

    print("=" * 70)
    print(f"P5.2.F — PyTorch oracle o_proj (layer {LAYER_IDX} reader)")
    print("=" * 70)
    print(f"o_proj = nn.Linear({M}, {HIDDEN}, bias=False)  [PAS de clipping — attention TEXTE]")
    print(f"Loaded context {tuple(context.shape)} (E.0, [b,h,q,hd]) + o_proj_weight {tuple(o_proj_weight.shape)}")
    print()

    # === Pipeline o_proj (miroir verbatim L1272-1273 + transpose de eager_attention_forward L834) ===
    attn_output = context.transpose(1, 2).contiguous()          # [1,8,4,256] -> [1,4,8,256]
    assert tuple(attn_output.shape) == (1, SEQ_LEN, N_HEADS, HEAD_DIM)
    attn_output = attn_output.reshape(1, SEQ_LEN, M).contiguous()  # [1,4,2048] concat têtes
    o_out = F.linear(attn_output, o_proj_weight)                 # [1,4,2048] @ [1536,2048]^T -> [1,4,1536]
    assert tuple(o_out.shape) == EXPECTED_OUT_SHAPE
    assert not torch.isnan(o_out).any() and not torch.isinf(o_out).any()

    # Sanity : recompute via einsum indépendant (ordre de concat h-major explicite).
    # o_out[b,q,o] = sum_{h,d} context[b,h,q,d] * W[o, h*head_dim + d]
    w_resh = o_proj_weight.reshape(HIDDEN, N_HEADS, HEAD_DIM)    # [1536,8,256] : o, h, d
    o_out_check = torch.einsum("bhqd,ohd->bqo", context, w_resh)
    recompute_diff = (o_out_check - o_out).abs().max().item()
    print(f"Sanity einsum (concat h-major explicite) vs F.linear |diff|_max = {recompute_diff:.3e} "
          f"(confirme l'ordre de concaténation des têtes, attendu < 1e-4)")
    assert recompute_diff < 1e-4, f"désaccord concat têtes (got {recompute_diff})"
    print()

    print("Fixed points (o_proj_out):")
    for q in [0, 3]:
        vals = o_out[0, q, :8].tolist()
        print(f"  o_proj_out[0,{q},:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print()

    print("Stats:")
    for name, t in [("context", context), ("o_proj_weight", o_proj_weight), ("o_proj_out", o_out)]:
        print(
            f"  {name:<14} shape={tuple(t.shape)!s:<16} "
            f"mean={t.mean().item(): .6e} std={t.std().item(): .6e} "
            f"min={t.min().item(): .6e} max={t.max().item(): .6e}"
        )

    tensors = {
        "context": context.contiguous(),
        "o_proj_weight": o_proj_weight.contiguous(),
        "o_proj_out": o_out.contiguous(),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.F PyTorch oracle o_proj (layer 15 reader, attention output projection)",
        "spec_refs": [
            "modeling_gemma4.py (5.9.0) Gemma4TextAttention.forward L1272-1273 (reshape + o_proj)",
            "eager_attention_forward L833-834 (matmul + transpose(1,2)) — context E.0 est PRE-transpose",
            "Gemma4TextAttention.__init__ L1209 : o_proj = nn.Linear(2048, 1536, bias=attention_bias=False)",
        ],
        "layer_idx": LAYER_IDX,
        "note_no_clipping": "o_proj texte = nn.Linear simple. Gemma4ClippableLinear = attention VISION (L911), pas ici.",
        "config": {"hidden": HIDDEN, "n_heads": N_HEADS, "head_dim": HEAD_DIM, "seq_len": SEQ_LEN, "m": M, "bias": False},
        "pipeline": [
            "attn_output = context.transpose(1,2)            [1,8,4,256] -> [1,4,8,256]",
            "attn_output = attn_output.reshape(1,4,2048)     concat têtes (h-major, hd-mineur)",
            "o_out       = F.linear(attn_output, o_proj_weight)   -> [1,4,1536]",
        ],
        "tensors": {name: {"shape": list(t.shape), "dtype": "float32"} for name, t in tensors.items()},
        "zml_pipeline_hint": [
            "context = load -> tags {.b,.h,.q,.hd}   [1,8,4,256]",
            "o_proj_weight = load -> tags {.o,.m}    [1536,2048]  (o=hidden out, m=n_heads*head_dim in)",
            "attn = context.transpose({.b,.q,.h,.hd}).merge({.m={.h,.hd}})   [1,4,2048]",
            "o_out = attn.dot(o_proj_weight, .m)   [1,4,1536]",
            "compare o_out vs oracle o_proj_out, tolérance 1e-4",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "expected_zml_residual_hint": "matmul réduction .m=2048 (longue) -> résidu attendu ~1e-5 (cf q_proj C.1 1.14e-5)",
        "checks": {"recompute_diff": recompute_diff},
        "interdits_p5_2_f": ["résiduel", "post_attention_layernorm", "MLP", "layer 14", "re-attention", "clipping (vision only)"],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.F oracle + fixture export PASS.")


if __name__ == "__main__":
    main()
