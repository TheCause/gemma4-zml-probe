#!/usr/bin/env python3
"""W4 W4g — contre-test de non-vacuité : copie du checkpoint avec UN weight_scale corrompu (x100).

Cible : model.language_model.layers.17.mlp.gate_proj.weight_scale (couche médiane).
Le décode gemma4_w4auto --oracle sur cette copie DOIT diverger (error.GenMismatch) : un 48/48
sur checkpoint corrompu prouverait que le chemin scale n'est pas consommé.

Historique des formes (publié, pas caché — 2 découvertes) :
1. q_proj x4 : PAS de flip — les marges top1-top2 bougent à chaque step (le scale EST consommé)
   mais l'argmax greedy est trop robuste (re-confirmation leçon du 28 juin).
2. q_proj x100 : PAS de flip NON PLUS — découverte : une corruption UNIFORME de scale sur q_proj
   est ANNULÉE par q_norm (RMSNorm(c.x) == RMSNorm(x), engine.zig:408). Idem pour k/v (k_norm,
   v_norm) et o/down/up (sandwich norms) : 10 des 11 familles de linears sont scale-invariantes
   par construction. Seul gate_proj traverse une NON-LINÉARITÉ (gelu) avant toute norm ->
   l'invariance casse -> cible retenue.
Sortie : /data/gemma4-zml-probe/weights_w4_corrupt.safetensors
"""
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/data/gemma4-zml-probe/weights_w4/model.safetensors"
DST = "/data/gemma4-zml-probe/weights_w4_corrupt.safetensors"
KEY = "model.language_model.layers.17.mlp.gate_proj.weight_scale"

out = {}
with safe_open(SRC, framework="pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        if k == KEY:
            t = t * 100.0
        out[k] = t.contiguous()
save_file(out, DST)
print(f"PASS : {DST} écrit ({KEY} x100)")
