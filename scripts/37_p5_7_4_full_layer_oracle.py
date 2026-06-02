"""P5.7.4 — Oracle couche décodeur FULL attention (layer 14, producer) — dispatcher full path.

Généralise P5.3 (couche sliding) à la full_attention : head_dim 512, RoPE manuelle partielle
(proportional θ1e6, partial 0.25). Oracle = module RÉEL Gemma4TextDecoderLayer(config, 14)
(producer full, KV local, intermediate 6144), rotary layer_type='full_attention', masque causal.

Exporte cos/sin full (512-wide) pour la RoPE manuelle ZML (le reste de la couche = identique
au sliding, dims 512). Fixture = inputs + cos/sin + 17 poids layer 14 + layer_out (oracle).
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
OUT_FIXTURE = ROOT / "fixtures" / "p5_7_4_full_layer14.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_7_4_full_layer14_manifest.json"

LAYER_IDX = 14
HIDDEN = 1536
SEQ_LEN = 4
PLE_DIM = 256
FULL_HD = 512

WEIGHT_SUBKEYS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.q_norm.weight",
    "self_attn.k_proj.weight", "self_attn.k_norm.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "post_feedforward_layernorm.weight",
    "per_layer_input_gate.weight", "per_layer_projection.weight", "post_per_layer_input_norm.weight",
    "layer_scalar",
]
PFX = f"model.language_model.layers.{LAYER_IDX}."


def main() -> None:
    assert WEIGHTS.exists()
    torch.manual_seed(1337)
    layer_input = torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float32)
    per_layer_input = torch.randn(1, SEQ_LEN, PLE_DIM, dtype=torch.float32)

    cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
    tc = getattr(cfg, "text_config", cfg)
    tc.torch_dtype = torch.float32

    layer = Gemma4TextDecoderLayer(tc, LAYER_IDX).to(torch.float32)
    layer.train(False)
    assert getattr(layer, "is_kv_shared_layer", False) is False, "layer 14 doit être producer"

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

    rot = Gemma4TextRotaryEmbedding(tc)
    pos = torch.arange(SEQ_LEN).unsqueeze(0)
    cos, sin = rot(layer_input, pos, layer_type="full_attention")   # [1,4,512]
    assert tuple(cos.shape) == (1, SEQ_LEN, FULL_HD)
    min_val = torch.finfo(torch.float32).min
    idx = torch.arange(SEQ_LEN)
    causal = (idx.view(SEQ_LEN, 1) >= idx.view(1, SEQ_LEN))
    attn_mask = torch.where(causal, torch.zeros(()), torch.full((), min_val)).view(1, 1, SEQ_LEN, SEQ_LEN)

    print("=" * 70)
    print(f"P5.7.4 — oracle couche FULL attention (layer {LAYER_IDX}, head_dim {FULL_HD})")
    print("=" * 70)
    print(f"layer_scalar = {raw['layer_scalar'].tolist()} | cos_full {tuple(cos.shape)}")

    with torch.no_grad():
        layer_out = layer(
            layer_input, per_layer_input=per_layer_input, shared_kv_states={},
            position_embeddings=(cos, sin), attention_mask=attn_mask,
            position_ids=pos, past_key_values=None,
        )
    if isinstance(layer_out, tuple):
        layer_out = layer_out[0]
    layer_out = layer_out.to(torch.float32).contiguous()
    assert tuple(layer_out.shape) == (1, SEQ_LEN, HIDDEN)
    assert not torch.isnan(layer_out).any()

    print("\nFixed points (layer_out):")
    for q in [0, 3]:
        vals = layer_out[0, q, :8].tolist()
        print(f"  layer_out[0,{q},:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print(f"\nStats layer_out: mean={layer_out.mean():.4e} std={layer_out.std():.4e} "
          f"min={layer_out.min():.4e} max={layer_out.max():.4e}")

    tensors = {
        "layer_input": layer_input.contiguous(),
        "per_layer_input": per_layer_input.contiguous(),
        "cos_full": cos.contiguous(),
        "sin_full": sin.contiguous(),
        "attn_mask": attn_mask.contiguous(),
        "layer_out": layer_out,
    }
    for sub, t in raw.items():
        tensors["w__" + sub.replace(".", "__")] = t.contiguous()
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    print(f"\nwrote {OUT_FIXTURE}  ({sum(t.numel()*t.element_size() for t in tensors.values())} bytes, {len(tensors)} tenseurs)")

    manifest = {
        "source": "P5.7.4 oracle couche FULL attention (layer 14 producer, head_dim 512, RoPE manuelle)",
        "spec_refs": ["Gemma4TextDecoderLayer.forward (module réel) ; full_attention rotary proportional partial"],
        "layer_idx": LAYER_IDX,
        "config": {"hidden": HIDDEN, "intermediate": 6144, "head_dim": FULL_HD, "n_heads": 8, "n_kv": 1,
                   "ple_dim": PLE_DIM, "seq_len": SEQ_LEN, "layer_type": "full_attention",
                   "rope": "proportional partial 0.25 theta1e6 (manuelle)"},
        "layer_scalar": raw["layer_scalar"].tolist(),
        "tensors": {n: {"shape": list(t.shape), "dtype": "float32"} for n, t in tensors.items()},
        "expected_zml_max_abs_le": 5.0e-4,
        "note": "généralise P5.3 (sliding) à full : head_dim 512 + RoPE manuelle partielle (P5.6/P5.6.K), reste identique.",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}\nP5.7.4 oracle PASS.")


if __name__ == "__main__":
    main()
