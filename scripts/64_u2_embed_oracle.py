#!/usr/bin/env python3
"""U2 — oracle embeddings 12B : module RÉEL Gemma4TextScaledWordEmbedding (piège 12).

Chemin HF EXACT reproduit par le gate (D12, U_12B_CONTRACT §7) : le module construit un buffer
`embed_scale` f32 (torch.tensor(√3840) = 61.96773…) et son forward fait
`super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)` — cast bf16 AU FORWARD
→ 62.0 exactement (0x4278), produit bf16 × bf16. AUCUNE recomposition manuelle ici : le module
est instancié depuis transformers (signature réelle : num_embeddings, embedding_dim,
padding_idx, embed_scale) et seul son poids est remplacé par le tenseur bf16 NON quantifié
`model.language_model.embed_tokens.weight` lu du checkpoint PACKÉ (safe_open, D9) — le buffer
f32 n'est PAS casté à la main, c'est le forward HF qui le fait.

Ids sondes (points fixes du plan) : 0 (pad), 1, 106, 4 ids texte (tokenisation consignée au
manifest), 262143 (dernier). Sortie : fixtures/u_embed.safetensors (ids i32 + expected bf16
[n,3840]) + manifest (critère BIT-EXACT max_abs == 0, versions figées).

Venv : /data/venvs/g12b. Lancement (VM) :
  /data/venvs/g12b/bin/python3 scripts/64_u2_embed_oracle.py
"""
import json
import os

os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = "/data/gemma4-zml-probe"
CKPT = os.path.join(ROOT, "weights_12b", "model.safetensors")
FX = os.path.join(ROOT, "fixtures")
SNAPSHOT_SHA = "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee"
EMB_KEY = "model.language_model.embed_tokens.weight"
TEXT_PROBE = "La capitale de la France est Paris."
N_TEXT_IDS = 4


def snapshot_dir() -> str:
    # PAS realpath : il traverse les 2 hops du cache HF et atterrit dans blobs/ (bug déjà mordu).
    snap = os.path.dirname(os.readlink(CKPT))
    assert SNAPSHOT_SHA in snap, f"snapshot inattendu : {snap}"
    return snap


def main() -> None:
    snap = snapshot_dir()
    cfg = json.load(open(os.path.join(snap, "config.json")))["text_config"]
    voc, hid, pad = cfg["vocab_size"], cfg["hidden_size"], cfg["pad_token_id"]
    assert (voc, hid, pad) == (262144, 3840, 0), f"config inattendue : {voc}/{hid}/{pad}"

    # Module RÉEL (piège 12) — signature vérifiée sur le venv : (num_embeddings, embedding_dim,
    # padding_idx, embed_scale). Buffer embed_scale = torch.tensor(√3840) f32, cast bf16 au forward.
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
    mod = Gemma4TextScaledWordEmbedding(voc, hid, pad, embed_scale=hid**0.5)

    with safe_open(CKPT, framework="pt") as f:
        w = f.get_tensor(EMB_KEY)
    assert w.dtype == torch.bfloat16 and list(w.shape) == [voc, hid], f"{EMB_KEY}: {w.dtype} {w.shape}"
    with torch.no_grad():
        mod.weight = torch.nn.Parameter(w)  # SEUL le poids remplacé — buffer f32 intact (chemin HF)

    # D12 : le cast bf16 du buffer doit donner 62.0 EXACTEMENT (0x4278) — sinon le contrat ment.
    assert mod.embed_scale.dtype == torch.float32, f"buffer {mod.embed_scale.dtype} != f32"
    sc16 = mod.embed_scale.to(torch.bfloat16)
    assert sc16.item() == 62.0 and sc16.view(torch.uint16).item() == 0x4278, \
        f"scale bf16 {sc16.item()} (0x{sc16.view(torch.uint16).item():04x}) != 62.0 (0x4278)"

    # Ids sondes : points fixes + 4 ids texte consignés (tokenizer réel du snapshot).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
    text_ids = tok(TEXT_PROBE, add_special_tokens=False)["input_ids"][:N_TEXT_IDS]
    assert len(text_ids) == N_TEXT_IDS, f"{len(text_ids)} ids texte < {N_TEXT_IDS}"
    ids = [0, 1, 106] + list(map(int, text_ids)) + [voc - 1]
    assert len(set(ids)) == len(ids), f"ids sondes non distincts : {ids}"

    with torch.no_grad():
        expected = mod(torch.tensor(ids, dtype=torch.int64))
    assert expected.dtype == torch.bfloat16 and list(expected.shape) == [len(ids), hid], \
        f"expected : {expected.dtype} {expected.shape}"
    assert not torch.isnan(expected.float()).any(), "NaN dans expected"
    assert expected.abs().float().max().item() > 0, "expected identiquement nul (fixture vacante)"

    os.makedirs(FX, exist_ok=True)
    save_file(
        {"ids": torch.tensor(ids, dtype=torch.int32), "expected": expected.contiguous()},
        os.path.join(FX, "u_embed.safetensors"),
    )

    import transformers
    manifest = {
        "source": "64_u2_embed_oracle.py (Task 3, plan J2 amendé 2026-07-24)",
        "snapshot_sha": SNAPSHOT_SHA,
        "oracle": "Gemma4TextScaledWordEmbedding (module réel transformers, poids du PACKÉ "
                  + EMB_KEY + ", buffer embed_scale f32 casté bf16 par le forward HF)",
        "cast_order": "gather bf16 -> * bf16(√3840)=62.0 (0x4278) -> bf16 (produit bf16×bf16)",
        "ids": ids,
        "ids_provenance": {
            "fixed": {"0": "pad", "1": "sonde basse", "106": "sonde", "262143": "dernier du vocab"},
            "text": {int(i): tok.decode([int(i)]) for i in text_ids},
            "text_probe": TEXT_PROBE,
        },
        "pass": "gate u2 : gather + scale chemin 12B == expected, BIT-EXACT u16 (max_abs == 0)",
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    with open(os.path.join(FX, "u_embed_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"PASS fixture U2 écrite : {len(ids)} ids {ids} -> fixtures/u_embed.safetensors", flush=True)


if __name__ == "__main__":
    main()
