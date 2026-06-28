#!/usr/bin/env python3
"""
spike_hadq_oracle.py — oracle du mini-spike ZML "Hadamard + nearest-centroid".

Produit une fixture safetensors + valeurs attendues pour valider que le moteur
ZML peut calculer, DANS LE GRAPHE :
  yr   = y0 @ Pi.T            (Hadamard, dot par matrice constante)
  idx  = argmin_k (yr - cb_k)^2   (nearest-centroid : sub + carre + argMax(-.))
  yhat = cb[idx]             (gather du codebook)

C'est exactement la brique risquee du quantizer TurboQuant (B). Pas de norm,
pas d'inverse-Hadamard ici : on isole les 2 ops a derisquer.

Export : /data/gemma4-zml-probe/spike_hadq.safetensors
  y0           [S, D]   f32  input
  hadamard     [D, D]   f32  Pi (Sylvester/sqrt(D)), tag ZML [.e,.d]
  codebook     [K]      f32  K centroides scalaires (Lloyd-Max via TurboQuantMSE)
  y_hat_oracle [S, D]   f32  resultat attendu = cb[argmin_k (yr-cb_k)^2]
"""
import os
import sys

os.environ.setdefault("HF_HOME", "/data/hf_cache")
import torch
from safetensors.torch import save_file

sys.path.insert(0, "/data/gemma4-zml-probe")  # turboquant.py deploye ici
from turboquant import TurboQuantMSE, _make_hadamard  # noqa: E402

S, D, B = 8, 256, 4          # S tokens, head_dim 256, b=4 bits -> K=16
OUT = "/data/gemma4-zml-probe/spike_hadq.safetensors"

torch.manual_seed(1337)

# Input arbitraire mais realiste (echelle ~ activations post-norm)
y0 = torch.randn(S, D, dtype=torch.float32) * 0.13

# Pi Hadamard Sylvester / sqrt(D) (deterministe, orthogonale). [D,D].
Pi = _make_hadamard(D, "cpu").to(torch.float32)   # Pi[e,d]

# Codebook gaussian Lloyd-Max identique a TurboQuantMSE (K=2^b)
mse = TurboQuantMSE(D, B, device="cpu", rotation="hadamard")
codebook = mse.codebook.to(torch.float32).clone()  # [K]
K = codebook.numel()

# Oracle: yr = y0 @ Pi.T ; idx = argmin_k (yr - cb_k)^2 ; yhat = cb[idx]
yr = y0 @ Pi.T                                   # [S, D]
dist2 = (yr.unsqueeze(-1) - codebook.view(1, 1, K)) ** 2   # [S, D, K]
idx = dist2.argmin(dim=-1)                        # [S, D]
y_hat = codebook[idx]                             # [S, D]

# Sanity : distances aux frontieres (flips possibles sous bruit Hadamard ~1e-5)
sorted_cb, _ = torch.sort(codebook)
min_gap = (sorted_cb[1:] - sorted_cb[:-1]).min().item()
# marge = distance du 2e centroide le plus proche - du plus proche
top2 = torch.topk(-dist2, 2, dim=-1).values        # [S,D,2] (= -d1, -d2)
margin = (-top2[..., 1] - (-top2[..., 0]))         # d2 - d1 >= 0
near_frontier = int((margin < 1e-3).sum())

print(f"S={S} D={D} K={K}  y0~N(0,0.13^2)")
print(f"codebook min_gap={min_gap:.4f}  range=[{codebook.min():.3f},{codebook.max():.3f}]")
print(f"coords proches d'une frontiere (margin<1e-3): {near_frontier}/{S*D} "
      f"(flips possibles sous bruit Hadamard ~1e-5)")
print(f"y_hat[0,:6] = {y_hat[0,:6].tolist()}")
print(f"idx[0,:8]   = {idx[0,:8].tolist()}")

save_file({
    "y0": y0.contiguous(),
    "hadamard": Pi.contiguous(),
    "codebook": codebook.contiguous(),
    "y_hat_oracle": y_hat.contiguous(),
}, OUT)
print(f"Wrote {OUT}")
# 8 premieres valeurs pour blocks fixed-point cote Zig
print("FIXED_POINT y_hat[0,:8]:", [round(x, 8) for x in y_hat[0, :8].tolist()])
