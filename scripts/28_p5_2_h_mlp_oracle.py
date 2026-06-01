"""P5.2.H — PyTorch oracle + fixture : sous-couche MLP (feed-forward).

Avec P5.2.G (sous-couche attention), ferme une COUCHE DÉCODEUR complète (sliding).
Reproduit la 2e moitié de `Gemma4TextDecoderLayer.forward` (modeling_gemma4.py 5.9.0, L1408-1427)
+ `Gemma4TextMLP.forward` (L~1062-1068) :

    residual = hidden_states                          # = sortie de la sous-couche attention (G)
    hidden_states = pre_feedforward_layernorm(hidden_states)
    hidden_states = mlp(hidden_states)                # gate/up/down + gelu
    hidden_states = post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

Gemma4TextMLP : gate=gate_proj(x) ; up=up_proj(x) ; h=act_fn(gate)*up ; out=down_proj(h)
  - act_fn = gelu_pytorch_tanh = F.gelu(x, approximate="tanh")
    = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))   ← EXACTEMENT zml Tensor.gelu.
  - gating : act(gate) * up (l'activation s'applique au gate SEUL), PAS act(gate*up).
  - gate_proj/up_proj : Linear 1536->6144 ; down_proj : Linear 6144->1536 ; bias=False.
  - pre/post_feedforward_layernorm : Gemma4RMSNorm(1536, with_scale=True) pattern Llama (*weight).

Chaînage : `residual` = `attn_sublayer_out` de P5.2.G (input MLP ET résiduel), comme dans la couche.

Oracle = source de vérité : modules réels Gemma4RMSNorm + activation HF ACT2FN['gelu_pytorch_tanh'].

Fixture (7 tenseurs) : residual, gate_proj_weight, up_proj_weight, down_proj_weight,
pre_ff_norm_weight, post_ff_norm_weight (inputs), mlp_sublayer_out (oracle). ~113 MB (poids 6144).

Interdits P5.2.H : attention (faite E/F/G), input_layernorm, per_layer_input (PLE), layer 14.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers.activations import ACT2FN
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm


ROOT = Path(__file__).resolve().parents[1]
G_FIXTURE = ROOT / "fixtures" / "p5_2_g_attn_residual_layer15.safetensors"   # P5.2.G (attn_sublayer_out)
WEIGHTS = ROOT / "weights" / "model.safetensors"
OUT_FIXTURE = ROOT / "fixtures" / "p5_2_h_mlp_layer15.safetensors"
OUT_MANIFEST = ROOT / "fixtures" / "p5_2_h_mlp_layer15_manifest.json"

LAYER_IDX = 15
HIDDEN = 1536
# Layer 15 est KV-shared (reader, i>=15) + config.use_double_wide_mlp=True
# -> intermediate DOUBLE = 12288 (cf modeling_gemma4 L1057-1060). Layers 0-14 = 6144.
INTERMEDIATE = 12288
SEQ_LEN = 4
RMS_EPS = 1e-6
PFX = f"model.language_model.layers.{LAYER_IDX}"
KEYS = {
    "gate": f"{PFX}.mlp.gate_proj.weight",
    "up": f"{PFX}.mlp.up_proj.weight",
    "down": f"{PFX}.mlp.down_proj.weight",
    "pre": f"{PFX}.pre_feedforward_layernorm.weight",
    "post": f"{PFX}.post_feedforward_layernorm.weight",
}
EXPECTED = {
    "gate": (INTERMEDIATE, HIDDEN), "up": (INTERMEDIATE, HIDDEN), "down": (HIDDEN, INTERMEDIATE),
    "pre": (HIDDEN,), "post": (HIDDEN,),
}
SHAPE = (1, SEQ_LEN, HIDDEN)


def main() -> None:
    assert G_FIXTURE.exists(), f"missing P5.2.G fixture {G_FIXTURE}"
    assert WEIGHTS.exists(), f"missing weights {WEIGHTS}"

    residual = load_file(str(G_FIXTURE))["attn_sublayer_out"].to(torch.float32).contiguous()
    assert tuple(residual.shape) == SHAPE, f"residual {tuple(residual.shape)} != {SHAPE}"

    w = {}
    with safe_open(str(WEIGHTS), framework="pt") as s:
        for name, key in KEYS.items():
            assert key in s.keys(), f"missing {key}"
            w[name] = s.get_tensor(key).to(torch.float32).contiguous()
            assert tuple(w[name].shape) == EXPECTED[name], f"{name} {tuple(w[name].shape)} != {EXPECTED[name]}"

    print("=" * 70)
    print(f"P5.2.H — PyTorch oracle MLP feed-forward (layer {LAYER_IDX})")
    print("=" * 70)
    print(f"act = gelu_pytorch_tanh ; intermediate={INTERMEDIATE} ; residual = attn_sublayer_out (G)")
    print()

    pre_ln = Gemma4RMSNorm(HIDDEN, eps=RMS_EPS)
    post_ln = Gemma4RMSNorm(HIDDEN, eps=RMS_EPS)
    with torch.no_grad():
        pre_ln.weight.copy_(w["pre"]); post_ln.weight.copy_(w["post"])

    # === MLP sublayer (miroir verbatim) ===
    x = pre_ln(residual)                                # pre_feedforward_layernorm
    gate = F.linear(x, w["gate"])                       # [1,4,6144]
    up = F.linear(x, w["up"])                           # [1,4,6144]
    act = ACT2FN["gelu_pytorch_tanh"](gate)             # activation sur gate SEUL
    gated = act * up                                    # gating
    mlp_out = F.linear(gated, w["down"])                # [1,4,1536]
    y = post_ln(mlp_out)                                # post_feedforward_layernorm
    mlp_sublayer_out = residual + y                     # résiduel
    assert tuple(mlp_sublayer_out.shape) == SHAPE
    assert not torch.isnan(mlp_sublayer_out).any() and not torch.isinf(mlp_sublayer_out).any()

    # Sanity 1 : gelu_pytorch_tanh == F.gelu(approximate='tanh') (= la formule de zml Tensor.gelu).
    act_check = F.gelu(gate, approximate="tanh")
    gelu_diff = (act_check - act).abs().max().item()
    print(f"Sanity gelu ACT2FN vs F.gelu(tanh) |diff|_max = {gelu_diff:.3e} (attendu 0 — confirme zml.gelu)")
    assert gelu_diff < 1e-6

    # Sanity 2 : gating = act(gate)*up, PAS act(gate*up).
    wrong = ACT2FN["gelu_pytorch_tanh"](gate * up)
    order_diff = (wrong - gated).abs().max().item()
    print(f"Sanity ordre gating : max|act(gate)*up - act(gate*up)| = {order_diff:.3f} (doit etre > 0)")
    assert order_diff > 1e-3

    # Effet résiduel non vide.
    add_shift = (mlp_sublayer_out - y).abs().max().item()
    print(f"Effet résiduel : max|out - y| = {add_shift:.4f} (> 0)")
    assert add_shift > 1e-3
    print()

    print("Fixed points (mlp_sublayer_out):")
    for q in [0, 3]:
        vals = mlp_sublayer_out[0, q, :8].tolist()
        print(f"  mlp_sublayer_out[0,{q},:8] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print()
    print("Stats:")
    for name, t in [("residual", residual), ("x(pre_ln)", x), ("gate", gate), ("act", act),
                    ("gated", gated), ("mlp_out", mlp_out), ("mlp_sublayer_out", mlp_sublayer_out)]:
        print(f"  {name:<18} shape={tuple(t.shape)!s:<16} mean={t.mean().item(): .4e} "
              f"std={t.std().item(): .4e} min={t.min().item(): .4e} max={t.max().item(): .4e}")

    tensors = {
        "residual": residual.contiguous(),
        "gate_proj_weight": w["gate"], "up_proj_weight": w["up"], "down_proj_weight": w["down"],
        "pre_ff_norm_weight": w["pre"], "post_ff_norm_weight": w["post"],
        "mlp_sublayer_out": mlp_sublayer_out.contiguous(),
    }
    OUT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_FIXTURE))
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print()
    print(f"wrote {OUT_FIXTURE}  ({total_bytes} bytes of tensor payload)")

    manifest = {
        "source": "P5.2.H PyTorch oracle MLP feed-forward (layer 15)",
        "spec_refs": [
            "modeling_gemma4.py L1408-1427 (pre_ff_norm -> mlp -> post_ff_norm -> +residual)",
            "Gemma4TextMLP.forward : down_proj(act_fn(gate_proj(x)) * up_proj(x))",
            "act_fn = gelu_pytorch_tanh ; zml Tensor.gelu = meme formule tanh-approx",
        ],
        "layer_idx": LAYER_IDX,
        "config": {"hidden": HIDDEN, "intermediate": INTERMEDIATE, "seq_len": SEQ_LEN,
                   "rms_eps": RMS_EPS, "activation": "gelu_pytorch_tanh", "bias": False},
        "residual_note": "residual = attn_sublayer_out (P5.2.G) = input MLP ET résiduel (chaîne de couche).",
        "pipeline": [
            "x = pre_feedforward_layernorm(residual)",
            "gate = gate_proj(x) ; up = up_proj(x)   [1,4,6144]",
            "gated = gelu_pytorch_tanh(gate) * up",
            "mlp_out = down_proj(gated)   [1,4,1536]",
            "y = post_feedforward_layernorm(mlp_out)",
            "out = residual + y",
        ],
        "tensors": {name: {"shape": list(t.shape), "dtype": "float32"} for name, t in tensors.items()},
        "zml_pipeline_hint": [
            "residual = load {.b,.q,.d=1536}",
            "gate/up_proj_weight = {.f=6144,.d=1536} ; down_proj_weight = {.d=1536,.f=6144}",
            "x = rmsNorm(residual,.d,1e-6).mul(pre_ff_w.broad)",
            "gate = x.dot(gate_proj_weight,.d) ; up = x.dot(up_proj_weight,.d)   {.b,.q,.f}",
            "gated = gate.gelu().mul(up)",
            "mlp_out = gated.dot(down_proj_weight,.f)   {.b,.q,.d}",
            "y = rmsNorm(mlp_out,.d,1e-6).mul(post_ff_w.broad)",
            "out = residual.add(y)",
            "compare vs mlp_sublayer_out, tol 1e-4",
        ],
        "expected_zml_max_abs_le": 1.0e-4,
        "checks": {"gelu_diff": gelu_diff, "gating_order_diff": order_diff, "residual_add_shift": add_shift},
        "interdits_p5_2_h": ["attention", "input_layernorm", "per_layer_input (PLE)", "layer 14"],
        "closes": "avec P5.2.G : COUCHE DÉCODEUR sliding complète (attention + MLP)",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    print()
    print("P5.2.H oracle + fixture export PASS.")


if __name__ == "__main__":
    main()
