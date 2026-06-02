"""P5.3 — PyTorch oracle + fixture : COUCHE DÉCODEUR sliding COMPLÈTE (layer 13, producer).

Capstone de composition : assemble en UNE couche tous les maillons validés (E/F/G/H) + input_layernorm
+ bloc PLE per-layer + layer_scalar. Oracle = module RÉEL `Gemma4TextDecoderLayer(config, 13)`
(producer sliding, intermediate 6144, calcule sa propre KV). On NE ré-dérive PAS : on appelle le
forward réel.

Forward réel (modeling_gemma4 L1395-1438) :
    residual = h ; h = input_layernorm(h) ; h = self_attn(h, ...) ; h = post_attn_norm(h) ; h += residual
    residual = h ; h = pre_ff_norm(h) ; h = mlp(h) ; h = post_ff_norm(h) ; h += residual
    if hidden_size_per_layer_input:                      # bloc PLE per-layer
        residual = h ; h = per_layer_projection(act_fn(per_layer_input_gate(h)) * per_layer_input)
        h = post_per_layer_input_norm(h) ; h += residual
    h *= layer_scalar

Inputs : layer_input [1,4,1536] (synthétique seed 1337, pré-input_layernorm), per_layer_input
[1,4,256] (synthétique seed, simule la sortie du pipeline PLE), cos/sin sliding, mask additif causal.

Fixture : layer_input, per_layer_input, cos_sliding, sin_sliding, attn_mask, layer_scalar,
+ TOUS les poids de layer 13 (17 tenseurs), + layer_out (oracle). Le ZML composera tout.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding,
)


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_3_layer13.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_3_layer13_manifest.json"

LAYER_IDX = 13
HIDDEN = 1536
SEQ_LEN = 4
PLE_DIM = 256
PFX = f"model.language_model.layers.{LAYER_IDX}."

WEIGHT_SUBKEYS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.q_norm.weight",
    "self_attn.k_proj.weight", "self_attn.k_norm.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "post_feedforward_layernorm.weight",
    "per_layer_input_gate.weight", "per_layer_projection.weight", "post_per_layer_input_norm.weight",
    "layer_scalar",
]


def main() -> None:
    assert WEIGHTS.exists()
    torch.manual_seed(1337)
    layer_input = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float32)
    per_layer_input = torch.randn(1, SEQ_LEN, PLE_DIM, dtype=torch.float32)

    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    tc_dtype = tc.torch_dtype
    tc.torch_dtype = torch.float32

    layer = Gemma4TextDecoderLayer(tc, LAYER_IDX).to(torch.float32)
    layer.train(False)
    assert getattr(layer, "is_kv_shared_layer", False) is False, "layer 13 doit être producer"

    state, raw = {}, {}
    with safe_open(str(WEIGHTS), framework="pt") as s:
        keys = set(s.keys())
        for sub in WEIGHT_SUBKEYS:
            k = PFX + sub
            assert k in keys, f"missing {k}"
            t = s.get_tensor(k).to(torch.float32).contiguous()
            state[sub] = t
            raw[sub] = t
    missing, unexpected = layer.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if m.endswith(".weight") or m == "layer_scalar"]
    assert not real_missing, f"poids manquants: {real_missing}"
    print(f"load_state_dict: {len(state)} tenseurs ; missing(non-poids)={len(missing)} unexpected={len(unexpected)}")

    rot = Gemma4TextRotaryEmbedding(tc)
    pos = torch.arange(SEQ_LEN).unsqueeze(0)
    cos, sin = rot(layer_input, pos, layer_type="sliding_attention")
    min_val = torch.finfo(torch.float32).min
    idx = torch.arange(SEQ_LEN)
    causal = (idx.view(SEQ_LEN, 1) >= idx.view(1, SEQ_LEN))
    attn_mask = torch.where(causal, torch.zeros(()), torch.full((), min_val)).view(1, 1, SEQ_LEN, SEQ_LEN)

    print("=" * 70)
    print(f"P5.3 — PyTorch oracle COUCHE DÉCODEUR sliding complète (layer {LAYER_IDX}, producer)")
    print("=" * 70)
    print(f"layer_scalar = {raw['layer_scalar'].tolist()}")

    shared_kv = {}
    with torch.no_grad():
        layer_out = layer(
            layer_input,
            per_layer_input=per_layer_input,
            shared_kv_states=shared_kv,
            position_embeddings=(cos, sin),
            attention_mask=attn_mask,
            position_ids=pos,
            past_key_values=None,
        )
    if isinstance(layer_out, tuple):
        layer_out = layer_out[0]
    layer_out = layer_out.to(torch.float32).contiguous()
    assert tuple(layer_out.shape) == (1, SEQ_LEN, HIDDEN), layer_out.shape
    assert not torch.isnan(layer_out).any()
    tc.torch_dtype = tc_dtype

    print("\nFixed points (layer_out):")
    for q in [0, 3]:
        vals = layer_out[0, q, :8].tolist()
        print(f"  layer_out[0,{q},:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print(f"\nStats layer_out: mean={layer_out.mean():.4e} std={layer_out.std():.4e} "
          f"min={layer_out.min():.4e} max={layer_out.max():.4e}")

    tensors = {
        "layer_input": layer_input.contiguous(),
        "per_layer_input": per_layer_input.contiguous(),
        "cos_sliding": cos.contiguous(),
        "sin_sliding": sin.contiguous(),
        "attn_mask": attn_mask.contiguous(),
        "layer_out": layer_out,
    }
    for sub, t in raw.items():
        tensors["w__" + sub.replace(".", "__")] = t.contiguous()

    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print(f"\nwrote {OUT_FIXTURE}  ({sum(t.numel()*t.element_size() for t in tensors.values())} bytes, {len(tensors)} tenseurs)")

    manifest = {
        "source": "P5.3 PyTorch oracle couche décodeur sliding complète (layer 13 producer)",
        "spec_refs": ["Gemma4TextDecoderLayer.forward L1395-1438 (module RÉEL, pas ré-dérivation)"],
        "layer_idx": LAYER_IDX,
        "config": {"hidden": HIDDEN, "intermediate": 6144, "head_dim": 256, "n_heads": 8, "n_kv": 1,
                   "ple_dim": PLE_DIM, "seq_len": SEQ_LEN, "sliding_window": 512, "layer_type": "sliding"},
        "layer_scalar": raw["layer_scalar"].tolist(),
        "tensors": {n: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for n, t in tensors.items()},
        "pipeline": [
            "residual=h; h=input_layernorm(h); h=self_attn(h,cos,sin,mask); h=post_attn_norm(h); h+=residual",
            "residual=h; h=pre_ff_norm(h); h=mlp(h); h=post_ff_norm(h); h+=residual",
            "residual=h; h=post_per_layer_input_norm(per_layer_projection(gelu(per_layer_input_gate(h))*per_layer_input)); h+=residual",
            "h*=layer_scalar",
        ],
        "expected_zml_max_abs_le": 5.0e-4,
        "note": "capstone composition (attention E/F/G + MLP H + input_ln + bloc PLE per-layer). Producer = KV local.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print("\nP5.3 oracle + fixture export PASS.")


if __name__ == "__main__":
    main()
