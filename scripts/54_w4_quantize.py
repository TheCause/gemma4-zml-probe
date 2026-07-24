#!/usr/bin/env python3
"""W4 W0a — production du checkpoint E2B w4a16 compressed-tensors (jalon J1).

Recette = recipe.yaml du checkpoint Google google/gemma-4-12B-it-qat-w4a16-ct,
rejouée sur google/gemma-4-E2B-it-qat-q4_0-unquantized (bf16 entraîné pour le 4-bit) :
QuantizationModifier, int4 symétrique, strategy group, group_size 32, observer memoryless_minmax,
ignore = [lm_head, re:.*embed.*, re:.*vision_tower.*, re:.*vision_embedder.*, re:.*audio_tower.*, re:.*router.*].
Weights-only statique => AUCUNE donnée de calibration (RTN).

E2B est YOCO (num_kv_shared_layers=20) : les couches 15-34 n'ont pas de k_proj/v_proj/k_norm
=> 276 linears quantifiés (15 producers x9 + 20 readers x7 + per_layer_model_projection).

Venv : /data/venvs/w4quant. Lancement (VM) :
  HF_HOME=/data/hf_cache /data/venvs/w4quant/bin/python3 scripts/54_w4_quantize.py
Sortie : /data/gemma4-zml-probe/weights_w4/ (config.json + recipe.yaml + model.safetensors mono-fichier)
"""
import os
os.environ.setdefault("HF_HOME", "/data/hf_cache")

import json
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

SRC = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
DST = "/data/gemma4-zml-probe/weights_w4"

IGNORE = [
    "lm_head",
    "re:.*embed.*",
    "re:.*vision_tower.*",
    "re:.*vision_embedder.*",
    "re:.*audio_tower.*",
    "re:.*router.*",
]

model = Gemma4ForConditionalGeneration.from_pretrained(SRC, dtype=torch.bfloat16)

recipe = QuantizationModifier(
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="int", strategy="group",
                group_size=32, symmetric=True, dynamic=False,
                observer="memoryless_minmax",
            ),
        )
    },
    ignore=IGNORE,
)
oneshot(model=model, recipe=recipe)

# max_shard_size volontairement énorme : le loader ZML (TensorRegistry.fromPath) lit UN fichier,
# comme le 12B Google (mono model.safetensors, pas d'index).
model.save_pretrained(DST, save_compressed=True, max_shard_size="20GB")
AutoTokenizer.from_pretrained(SRC).save_pretrained(DST)

# --- Auto-vérification du format produit (échoue bruyamment si le layout dévie) ---
from safetensors import safe_open

path = os.path.join(DST, "model.safetensors")
assert os.path.exists(path), f"mono-fichier attendu : {path} absent (sharding non désactivé ?)"
n_packed = n_scale = n_zp = n_gidx = 0
kproj_layers = []
with safe_open(path, framework="pt") as f:
    for k in f.keys():
        if k.endswith("weight_packed"):
            n_packed += 1
            t = f.get_slice(k)
            assert t.get_dtype() == "I32", f"{k}: dtype {t.get_dtype()} != I32"
            if ".self_attn.k_proj." in k:
                kproj_layers.append(int(k.split("layers.")[1].split(".")[0]))
        elif k.endswith("weight_scale"):
            n_scale += 1
            t = f.get_slice(k)
            assert t.get_dtype() == "BF16", f"{k}: dtype {t.get_dtype()} != BF16"
        elif k.endswith("weight_zero_point"):
            n_zp += 1
        elif k.endswith("weight_g_idx"):
            n_gidx += 1
    # sonde de shape sur q_proj L0 : [2048, 1536] => packed [2048, 192], scale [2048, 48]
    qp = f.get_slice("model.language_model.layers.0.self_attn.q_proj.weight_packed")
    qs = f.get_slice("model.language_model.layers.0.self_attn.q_proj.weight_scale")
    assert list(qp.get_shape()) == [2048, 192], f"q_proj packed shape {qp.get_shape()}"
    assert list(qs.get_shape()) == [2048, 48], f"q_proj scale shape {qs.get_shape()}"

# 276 = 15 producers x 9 + 20 readers (YOCO, sans k/v) x 7 + per_layer_model_projection
assert n_packed == 276, f"{n_packed} linears packés != 276 (15x9 + 20x7 + 1)"
assert n_scale == 276 and n_zp == 0 and n_gidx == 0, f"scale={n_scale} zp={n_zp} gidx={n_gidx}"
# structure YOCO : k_proj packé sur les producers 0-14 UNIQUEMENT
assert sorted(kproj_layers) == list(range(15)), f"k_proj packés sur couches {sorted(kproj_layers)}"

qc = json.load(open(os.path.join(DST, "config.json")))["quantization_config"]
g0w = qc["config_groups"]["group_0"]["weights"]
assert qc["format"] == "pack-quantized" and g0w["group_size"] == 32 and g0w["symmetric"] is True

print(f"PASS W0a : {n_packed} linears packés (15 prod x9 + 20 readers x7 + plmp), pack-quantized g32 sym, mono-fichier OK")
