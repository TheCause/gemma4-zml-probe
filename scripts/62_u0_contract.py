#!/usr/bin/env python3
"""U0 — contrat du checkpoint 12B w4a16-ct, vérifié sur pièce (jalon J2).

Assertions = les faits D1 du plan. Toute déviation = échec bruyant AVANT le moteur.
lm_head != embed_tokens => exit 2 (STOP D7) — PAS de PASS possible dans cet état.
Venv : /data/venvs/g12b. Lancement (VM) :
  /data/venvs/g12b/bin/python3 scripts/62_u0_contract.py
Sortie : fixtures/u0_contract_manifest.json (committé au repo)
"""
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open

ROOT = "/data/gemma4-zml-probe"
CKPT = os.path.join(ROOT, "weights_12b", "model.safetensors")
LM = "model.language_model."
FULL = [5, 11, 17, 23, 29, 35, 41, 47]          # (i+1)%6==0
N_LAYERS = 48
TOK_SHA_E2B = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"

with safe_open(CKPT, framework="pt") as f:
    keys = set(f.keys())

    packed = sorted(k for k in keys if k.endswith("weight_packed"))
    assert len(packed) == 328, f"{len(packed)} packed != 328 (48x6 + 40 v_proj)"
    vp = sorted(int(k.split("layers.")[1].split(".")[0]) for k in packed if ".v_proj." in k)
    assert vp == [i for i in range(N_LAYERS) if i not in FULL], \
        f"v_proj sur {len(vp)} couches — attendu les 40 sliding uniquement"
    assert not any(k.endswith(("weight_zero_point", "weight_g_idx")) for k in keys)

    probes = {
        LM + "layers.0.self_attn.q_proj.weight_packed": [4096, 480],   # 16x256, in 3840
        LM + "layers.0.self_attn.q_proj.weight_scale": [4096, 120],
        LM + "layers.5.self_attn.q_proj.weight_packed": [8192, 480],   # 16x512 (full)
        LM + "layers.5.self_attn.k_proj.weight_packed": [512, 480],    # 1x512 MQA
        LM + "layers.5.self_attn.o_proj.weight_packed": [3840, 1024],  # in 8192
        LM + "layers.0.mlp.down_proj.weight_packed": [3840, 1920],     # in 15360
        LM + "embed_tokens.weight": [262144, 3840],
        "lm_head.weight": [262144, 3840],
        LM + "layers.5.self_attn.q_norm.weight": [512],
        LM + "layers.0.self_attn.q_norm.weight": [256],
        LM + "layers.0.layer_scalar": [1],
    }
    for k, shp in probes.items():
        got = list(f.get_slice(k).get_shape())
        assert got == shp, f"{k}: {got} != {shp}"
    assert LM + "layers.5.self_attn.v_proj.weight_packed" not in keys
    assert LM + "layers.0.self_attn.v_proj.weight_packed" in keys
    assert not any(".v_norm." in k for k in keys), "v_norm ne doit pas avoir de poids"

    # --- lm_head vs embed_tokens : bit-égalité (D7) — par tranches, RAM bornée ---
    same = True
    for lo in range(0, 262144, 16384):
        a = f.get_slice("lm_head.weight")[lo:lo + 16384]
        b = f.get_slice(LM + "embed_tokens.weight")[lo:lo + 16384]
        if not torch.equal(a, b):
            same = False
            break

# --- config ---
# readlink UN SEUL hop : realpath traverserait le 2e niveau de symlinks HF-cache et
# atterrirait dans blobs/ (content-addressed, sans config.json) — bug attrapé à l'exécution.
snap = os.path.dirname(os.readlink(CKPT))
cfg = json.load(open(os.path.join(snap, "config.json")))
tc = cfg["text_config"]
assert tc["num_hidden_layers"] == N_LAYERS and tc["hidden_size"] == 3840
assert tc["num_attention_heads"] == 16 and tc["num_key_value_heads"] == 8
assert tc["head_dim"] == 256 and tc["global_head_dim"] == 512
assert tc["num_global_key_value_heads"] == 1 and tc["attention_k_eq_v"] is True
assert tc["sliding_window"] == 1024 and tc["final_logit_softcapping"] == 30.0
assert tc["hidden_size_per_layer_input"] == 0 and tc["num_kv_shared_layers"] == 0
lt = tc["layer_types"]
assert [i for i, t in enumerate(lt) if t == "full_attention"] == FULL
qc = cfg["quantization_config"]
assert qc["format"] == "pack-quantized"
g0w = qc["config_groups"]["group_0"]["weights"]
assert g0w["group_size"] == 32 and g0w["symmetric"] is True and g0w["num_bits"] == 4

# --- tokenizer / chat template vs E2B (comparaison AUTOMATIQUE) ---
tok_sha = hashlib.sha256(open(os.path.join(snap, "tokenizer.json"), "rb").read()).hexdigest()
assert tok_sha == TOK_SHA_E2B, tok_sha


def sha_or_none(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


ct12 = sha_or_none(os.path.join(snap, "chat_template.jinja"))
# référence E2B : le snapshot E2B-it en cache VM (celui des oracles historiques)
import glob
e2b_snaps = glob.glob("/data/hf_cache/hub/models--google--gemma-4-E2B-it/snapshots/*/chat_template.jinja")
ct_e2b = sha_or_none(e2b_snaps[0]) if e2b_snaps else None
templates_match = (ct12 == ct_e2b) if (ct12 and ct_e2b) else None
print(f"chat_template 12B == E2B : {templates_match} (12B {ct12}, E2B {ct_e2b})")
if templates_match is None:
    print("U0 WARN: comparaison template incomplète (fichier absent d'un côté) — vérif MANUELLE requise avant Task 8")

json.dump(
    {
        "source": "62_u0_contract.py — contrat 12B vérifié sur pièce",
        "snapshot": os.path.basename(snap),
        "packed": 328, "full_layers": FULL, "v_proj_layers": "les 40 sliding",
        "lm_head_eq_embed": same, "tokenizer_sha256": tok_sha,
        "chat_template_12b": ct12, "chat_template_e2b": ct_e2b,
        "templates_match": templates_match,
        "pass": "toutes assertions D1 tenues" if same else "BLOQUÉ D7",
    },
    open(os.path.join(ROOT, "fixtures", "u0_contract_manifest.json"), "w"), indent=2,
)

print(f"lm_head == embed_tokens (bit) : {same}")
if not same:
    print("U0 VERDICT: DECISION REQUISE (D7) — lm_head != embed_tokens, STOP")
    sys.exit(2)
print("PASS U0 : contrat 12B conforme à D1")
