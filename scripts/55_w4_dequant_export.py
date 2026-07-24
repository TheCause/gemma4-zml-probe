#!/usr/bin/env python3
"""W4 W0b — round-trip HF du checkpoint packé + export du checkpoint DÉCOMPRESSÉ bf16.

Charge weights_w4 via transformers + CompressedTensorsConfig(run_compressed=False) : la
décompression est celle de compressed-tensors (référence, piège 12 — jamais re-dérivée).
Exporte le state_dict dense bf16 avec les clés du checkpoint E2B d'origine
=> weights_w4dq/model.safetensors, consommable par les oracles existants (49/56) tel quel.

E2B est YOCO : le checkpoint d'origine porte des k/v/k_norm VESTIGIAUX pour les 35 couches
(droppés par HF au load) ; le modèle QAT kv-shared ne les instancie pas pour les readers
15-34 -> 60 clés absentes de l'export, ATTENDU.

Venv : /data/venvs/w4quant. Lancement (VM) :
  HF_HOME=/data/hf_cache /data/venvs/w4quant/bin/python3 scripts/55_w4_dequant_export.py
"""
import os
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Gemma4ForConditionalGeneration
from transformers.utils.quantization_config import CompressedTensorsConfig

SRC = "/data/gemma4-zml-probe/weights_w4"
REF = "/data/gemma4-zml-probe/weights/model.safetensors"  # E2B d'origine : liste des clés attendues
OUT_DIR = "/data/gemma4-zml-probe/weights_w4dq"
OUT = os.path.join(OUT_DIR, "model.safetensors")

model = Gemma4ForConditionalGeneration.from_pretrained(
    SRC, dtype=torch.bfloat16,
    quantization_config=CompressedTensorsConfig(run_compressed=False),
)
sd = model.state_dict()

with safe_open(REF, framework="pt") as f:
    ref_keys = set(f.keys())

# Le checkpoint E2B-it ORIGINE porte des k/v/k_norm VESTIGIAUX pour les 35 couches (droppés par
# HF au load) ; le modèle QAT kv-shared n'instancie PAS ces modules pour les readers 15-34
# (num_kv_shared_layers=20) -> ces 60 clés sont ABSENTES du state_dict, et c'est ATTENDU.
VESTIGIAL = {
    f"model.language_model.layers.{i}.self_attn.{m}.weight"
    for i in range(15, 35)
    for m in ("k_proj", "v_proj", "k_norm")
}

out, missing = {}, []
for k in sorted(ref_keys):
    if k in sd:
        t = sd[k]
    elif "lm_head" in k or k in VESTIGIAL:
        continue  # tied / vestigial YOCO — absents du modèle, c'est le comportement HF
    else:
        missing.append(k)
        continue
    assert t.dtype == torch.bfloat16, f"{k}: {t.dtype} != bf16 (décompression incomplète ?)"
    assert not torch.isnan(t.float()).any(), f"NaN dans {k}"
    out[k] = t.contiguous()

assert not missing, f"clés du checkpoint d'origine absentes du modèle décompressé : {missing[:5]}"
assert not (VESTIGIAL & set(sd)), "des clés vestigiales existent dans le state_dict — hypothèse YOCO fausse ?"

os.makedirs(OUT_DIR, exist_ok=True)
save_file(out, OUT)
print(f"PASS W0b-export : {len(out)} tenseurs bf16 -> {OUT}")
