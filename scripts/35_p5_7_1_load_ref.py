"""P5.7.1 — Référence PyTorch pour la vérification du chargement ZML (embeddings + 1 couche).

P5.7.1 prouve que ZML charge les poids depuis le checkpoint RÉEL (weights/model.safetensors, bf16,
2011 tenseurs) par clé sélective — pas un slim-fixture. Ce script lit les MÊMES clés et émet :
- les shapes (pour assertions ZML),
- des fixed points (fp32) de 4 petits tenseurs (layer_scalar, input_layernorm, q_norm, k_norm de
  layer 13) à hardcoder dans le runner pour la comparaison de valeurs.

bf16 → fp32 est exact (élargissement) : ZML doit retrouver ces valeurs au bit près.
Aucune fixture écrite (le runner lit le checkpoint réel directement).
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "model.safetensors"
LAYER = 13
PFX = f"model.language_model.layers.{LAYER}"
SMALL = {
    "layer_scalar": f"{PFX}.layer_scalar",
    "input_layernorm": f"{PFX}.input_layernorm.weight",
    "q_norm": f"{PFX}.self_attn.q_norm.weight",
    "k_norm": f"{PFX}.self_attn.k_norm.weight",
}
SHAPES = {
    "embed_tokens": "model.language_model.embed_tokens.weight",
    "final_norm": "model.language_model.norm.weight",
    "q_proj": f"{PFX}.self_attn.q_proj.weight",
    "k_proj": f"{PFX}.self_attn.k_proj.weight",
    "o_proj": f"{PFX}.self_attn.o_proj.weight",
    "gate_proj": f"{PFX}.mlp.gate_proj.weight",
    "down_proj": f"{PFX}.mlp.down_proj.weight",
}


def main() -> None:
    assert WEIGHTS.exists(), f"missing {WEIGHTS}"
    print("=" * 70)
    print(f"P5.7.1 — référence chargement (layer {LAYER}), checkpoint bf16 réel")
    print("=" * 70)
    with safe_open(str(WEIGHTS), framework="pt") as f:
        keys = set(f.keys())
        print("Shapes (pour assertions ZML) :")
        for name, key in SHAPES.items():
            t = f.get_slice(key)
            print(f"  {name:<16} {key}  shape={list(t.get_shape())} dtype={t.get_dtype()}")
        print("\nFixed points fp32 (à hardcoder dans le runner) :")
        for name, key in SMALL.items():
            assert key in keys, f"missing {key}"
            t = f.get_tensor(key).to(torch.float32).flatten()
            n = min(8, t.numel())
            vals = t[:n].tolist()
            print(f"  {name:<16} shape={list(f.get_slice(key).get_shape())} dtype_disk={f.get_slice(key).get_dtype()}")
            print(f"    [:{n}] = [{', '.join(f'{v:.10f}' for v in vals)}]")
    print("\nP5.7.1 référence émise.")


if __name__ == "__main__":
    main()
