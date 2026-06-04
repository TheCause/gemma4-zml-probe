#!/usr/bin/env python3
"""Q3 oracle — MSE V-only quantizer chain reference fixture.

Pour d in {256, 512} :
  - tenseur V jouet v = randn(KPOS=8, d) * 0.13 (seed fixe)
  - mse = TurboQuantMSE(d, 4, device="cpu", rotation="hadamard")
  - v_hat = mse.dequant(mse.quant(v))   (chaine de reference, embarque l'arrondi fp16 de la norm)
  - export fixture safetensors f32 contigus : v, hadamard (Pi), codebook, v_hat_oracle

Lance sur la 3090 (CPU torch).
"""
import os
import sys

os.environ["HF_HOME"] = "/data/hf_cache"
sys.path.insert(0, "/data/gemma4-zml-probe")

import torch
from safetensors.torch import save_file

from turboquant import TurboQuantMSE

KPOS = 8
BITS = 4


def build_fixture(d: int):
    torch.manual_seed(1234 + d)  # seed fixe par d
    v = (torch.randn(KPOS, d) * 0.13).to(torch.float32)

    mse = TurboQuantMSE(d, BITS, device="cpu", rotation="hadamard")
    v_hat = mse.dequant(mse.quant(v))

    tensors = {
        "v": v.contiguous().to(torch.float32),
        "hadamard": mse.Pi.contiguous().to(torch.float32),          # [d, d]
        "codebook": mse.codebook.contiguous().to(torch.float32),    # [16]
        "v_hat_oracle": v_hat.contiguous().to(torch.float32),       # [KPOS, d]
    }

    out = f"/data/gemma4-zml-probe/spike_vquant_{d}.safetensors"
    save_file(tensors, out)
    print(f"[d={d}] wrote {out}")
    for name, t in tensors.items():
        print(f"  {name:14s} shape={tuple(t.shape)} dtype={t.dtype}")
    # quick sanity : max abs ecart oracle vs v (juste informatif)
    err = (v_hat - v).abs()
    print(f"  recon err vs v : max={err.max().item():.4e} mean={err.mean().item():.4e}")


if __name__ == "__main__":
    for d in (256, 512):
        build_fixture(d)
    print("OK")
