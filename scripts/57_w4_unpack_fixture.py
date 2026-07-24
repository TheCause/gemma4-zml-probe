#!/usr/bin/env python3
"""W4 W1/W2 — fixtures d'unpack/dequant.

Étage 0 (W1) : 2 rangées x 32 colonnes de q int4 CHOISIS (couvre les 16 nibbles, le bit 31 du mot
i32, les deux extrêmes -8/7), packés par la RÉFÉRENCE compressed-tensors (pack_to_int32 importé,
jamais recodé — piège 12). Expected = q int8 + dequant bf16.
Étage 1 (W2) : expected dequant bf16 de 5 matrices COMPLÈTES du checkpoint weights_w4, une par
famille de shape (déviation déclarée vs spec §3.5 « entier » — plan §3), avec selfcheck
bit-identique contre la décompression de RÉFÉRENCE (weights_w4dq, script 55).

Contre-test de non-vacuité W1 : une copie CORROMPUE du packed (1 nibble flippé) est aussi écrite ;
le runner doit diverger dessus, sinon le gate est vide.

Venv : /data/venvs/w4quant. Lancement (VM) :
  /data/venvs/w4quant/bin/python3 scripts/57_w4_unpack_fixture.py
Sorties : fixtures/w4_unpack.safetensors (+ manifest), fixtures/w4_mats.safetensors (+ manifest)
"""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors.pack_quantized import pack_to_int32, unpack_from_int32

ROOT = "/data/gemma4-zml-probe"
CKPT = os.path.join(ROOT, "weights_w4", "model.safetensors")
FX = os.path.join(ROOT, "fixtures")
os.makedirs(FX, exist_ok=True)

# ---------- Étage 0 : groupes choisis ----------
row0 = torch.tensor(list(range(-8, 8)) + list(range(7, -9, -1)), dtype=torch.int8)  # 32 valeurs
torch.manual_seed(1337)
row1 = torch.randint(-8, 8, (32,), dtype=torch.int8)
row1[7] = 7   # nibble 7 du mot 0 = 15 -> bit 31 du i32 à 1 (piège du signe)
row1[15] = 7  # idem mot 1
q = torch.stack([row0, row1])                                   # [2, 32] int8
packed = pack_to_int32(q.clone(), num_bits=4)                    # [2, 4] int32 (réf : q+8, LE nibble)
scales = torch.tensor([[0.5], [0.033203125]], dtype=torch.bfloat16)  # [2, 1] (1 groupe de 32/rangée)
deq = q.to(torch.bfloat16) * scales                              # dequant réf : (nibble-8)*scale, bf16

corrupt = packed.clone()
corrupt[0, 0] ^= 0x000000F0  # flippe le nibble 1 (colonne 1) de la rangée 0

save_file(
    {
        "packed": packed.contiguous(), "scales": scales.contiguous(),
        "q_expected": q.contiguous(), "deq_expected": deq.contiguous(),
        "packed_corrupt": corrupt.contiguous(),
    },
    os.path.join(FX, "w4_unpack.safetensors"),
)
json.dump(
    {
        "source": "57_w4_unpack_fixture.py étage 0",
        "layout": "compressed-tensors pack-quantized : nibble j = bits [4j,4j+4), stocke q+8",
        "pass": "unpack ZML == q_expected (int, bit-exact) ; dequant ZML == deq_expected (bf16, bit-exact) ; "
                "packed_corrupt DOIT diverger (non-vacuité)",
        "tensors": {"packed": [2, 4], "scales": [2, 1], "q_expected": [2, 32], "deq_expected": [2, 32]},
    },
    open(os.path.join(FX, "w4_unpack_manifest.json"), "w"), indent=2,
)

# selfcheck étage 0 : la référence doit se relire elle-même
q_rt = unpack_from_int32(packed, num_bits=4, shape=torch.Size([2, 32]))
assert torch.equal(q_rt, q.to(q_rt.dtype)), "selfcheck : unpack(pack(q)) != q"

# ---------- Étage 1 : 5 matrices réelles, une par famille de shape (plan §3, W2) ----------
LM = "model.language_model."
MATS = {  # nom fixture -> (module, [out, in])
    "deq_q0": (LM + "layers.0.self_attn.q_proj", [2048, 1536]),      # sliding
    "deq_q4": (LM + "layers.4.self_attn.q_proj", [4096, 1536]),      # full (head_dim 512)
    "deq_dn20": (LM + "layers.20.mlp.down_proj", [1536, 12288]),     # double-wide (KV-shared)
    "deq_plp0": (LM + "layers.0.per_layer_projection", [1536, 256]), # petite in
    "deq_plmp": (LM + "per_layer_model_projection", [8960, 1536]),   # globale
}
out_t = {}
with safe_open(CKPT, framework="pt") as f:
    for name, (mod, shp) in MATS.items():
        wp = f.get_tensor(mod + ".weight_packed")
        ws = f.get_tensor(mod + ".weight_scale")
        assert list(wp.shape) == [shp[0], shp[1] // 8], f"{mod}: packed {list(wp.shape)}"
        q_full = unpack_from_int32(wp, num_bits=4, shape=torch.Size(shp))
        deq_m = (q_full.reshape(shp[0], shp[1] // 32, 32).to(torch.bfloat16) * ws.unsqueeze(-1)).reshape(*shp)
        out_t[name] = deq_m.contiguous()

# Selfcheck piège 12 : l'expected recalculé DOIT être bit-identique à la décompression de
# RÉFÉRENCE (weights_w4dq, produite par transformers/compressed-tensors au script 55) — on ne
# fait pas confiance à notre recalcul, on le prouve contre la lib de bout en bout.
DQREF = os.path.join(ROOT, "weights_w4dq", "model.safetensors")
with safe_open(DQREF, framework="pt") as f:
    for name, (mod, _) in MATS.items():
        ref = f.get_tensor(mod + ".weight")
        assert torch.equal(out_t[name], ref), f"selfcheck : {mod} recalcul != décompression référence"

save_file(out_t, os.path.join(FX, "w4_mats.safetensors"))
json.dump(
    {
        "source": "57_w4_unpack_fixture.py étage 1 (5 familles de shape de weights_w4)",
        "pass": "dequant ZML de weight_packed/weight_scale (lus du checkpoint) == deq_*, bit-exact bf16, 5/5",
        "matrices": {k: v[1] for k, v in MATS.items()},
        "modules": {k: v[0] for k, v in MATS.items()},
    },
    open(os.path.join(FX, "w4_mats_manifest.json"), "w"), indent=2,
)
print("PASS fixtures W1/W2 écrites")
