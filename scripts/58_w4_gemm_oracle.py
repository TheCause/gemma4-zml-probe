#!/usr/bin/env python3
"""W4 W3 — oracle du premier GEMM W4 (q_proj L0), flux du moteur : bf16 -> f32 GEMM.

x aléatoire seedé [1,1,1536] bf16 ; w = q_proj L0 lu dans weights_w4dq (la décompression de
RÉFÉRENCE produite au script 55 — piège 12 : jamais de recalcul maison) ;
out_expected = x.f32 @ w.f32.T  (même flux que dotPrec(fam=null) : les deux opérandes upcastés f32).
Sortie : fixtures/w4_gemm.safetensors + manifest (8 points fixes + stats).
Venv : /data/venvs/w4quant.
"""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = "/data/gemma4-zml-probe"
DQREF = os.path.join(ROOT, "weights_w4dq", "model.safetensors")
FX = os.path.join(ROOT, "fixtures")
os.makedirs(FX, exist_ok=True)
QPROJ = "model.language_model.layers.0.self_attn.q_proj"

torch.manual_seed(1337)
x = torch.randn(1, 1, 1536, dtype=torch.float32).to(torch.bfloat16)

with safe_open(DQREF, framework="pt") as f:
    w = f.get_tensor(QPROJ + ".weight")  # [2048, 1536] bf16, dequant par la référence

out = x.to(torch.float32) @ w.to(torch.float32).t()  # [1, 1, 2048]
assert not torch.isnan(out).any()

save_file({"x": x.contiguous(), "out_expected": out.contiguous()}, os.path.join(FX, "w4_gemm.safetensors"))
json.dump(
    {
        "source": "58_w4_gemm_oracle.py (q_proj L0, flux bf16->f32 GEMM = dotPrec fam=null)",
        "pass": "max_abs <= 1e-4, mean_abs <= 1e-6, 8 points fixes",
        "fixed_points_out_0_0_0_8": out[0, 0, :8].tolist(),
        "stats": {"max": float(out.max()), "min": float(out.min()), "mean": float(out.mean())},
    },
    open(os.path.join(FX, "w4_gemm_manifest.json"), "w"), indent=2,
)
print("PASS fixture W3 écrite")
