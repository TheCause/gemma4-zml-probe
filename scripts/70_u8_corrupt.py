#!/usr/bin/env python3
"""U8 contre-test câblé (D11) : copie du checkpoint PACKÉ 12B avec
`model.language_model.layers.24.mlp.gate_proj.weight_scale` ×100.

Pourquoi gate_proj (leçon J1, 59_w4_corrupt_ckpt.py, historique publié) : seule famille de
linears traversant une NON-LINÉARITÉ (gelu) avant toute norm — les 10 autres sont
scale-invariantes par construction (RMSNorm(c·x) == RMSNorm(x)). Cible L24 = position relative
0,50 (== L17/35 E2B, D11). Critère du gate (pré-enregistré, piège 13) : divergence des LOGITS
top-5 à un step ≤ 48 + sortie `error.A1Mismatch` — le flip argmax = bonus, PAS le critère.

Différence vs 59 (E2B : dict complet en RAM puis save_file) : le packé 12B fait ~10,3 Go pour
~21 Go de RAM VM — ici copie du FICHIER + patch IN-PLACE du seul tenseur weight_scale (offsets
lus du header safetensors), jamais les 328 modules en RAM. ⚠ La source est un SYMLINK vers le
blob du cache HF : copyfile SUIT le lien (le blob original n'est JAMAIS touché — le patch
s'applique à la copie seulement, vérifié par re-lecture des deux fichiers à la fin).

Sortie : /data/gemma4-zml-probe/weights_12b_corrupt.safetensors (~10,3 Go — ménage par
l'appelant après le verdict, plan Step 9.3).
"""
import json
import os
import shutil
import struct

import torch
from safetensors import safe_open

ROOT = "/data/gemma4-zml-probe"
SRC = os.path.join(ROOT, "weights_12b", "model.safetensors")
DST = os.path.join(ROOT, "weights_12b_corrupt.safetensors")
KEY = "model.language_model.layers.24.mlp.gate_proj.weight_scale"
# Témoin non-touché : le packé du MÊME module (voisin immédiat dans le fichier).
WITNESS = "model.language_model.layers.24.mlp.gate_proj.weight_packed"
FACTOR = 100.0
CONFIG_NAME = "generation_config.json"


def find_generation_config(src: str) -> str | None:
    """Localise le `generation_config.json` du checkpoint SOURCE — UN seul hop, `realpath`
    interdit (même règle que 63_u_dequant_export.py:70-71 et zml_runner/gencfg.zig).

    `weights_12b/` ne contient que des symlinks ; le fichier vit dans le SNAPSHOT pointé. Un
    `realpath` complet atterrirait dans `blobs/<sha256>`, qui ne le contient pas non plus.
    """
    direct = os.path.join(os.path.dirname(src), CONFIG_NAME)
    if os.path.exists(direct):
        return direct
    if os.path.islink(src):
        target = os.readlink(src)
        if not os.path.isabs(target):  # cache HF : cibles relatives (mesuré)
            target = os.path.join(os.path.dirname(src), target)
        # ⚠ `os.path.normpath`, PAS `os.path.realpath` : realpath résout TOUS les liens et
        # atterrit dans `blobs/<sha256>`, qui ne contient pas le fichier. Erreur commise ici même
        # à la première écriture, et attrapée par un test sur la VM (la découverte rendait None).
        hop = os.path.join(os.path.dirname(os.path.normpath(target)), CONFIG_NAME)
        if os.path.exists(hop):
            return hop
    return None


def main() -> None:
    assert os.path.exists(SRC), SRC
    src_size = os.path.getsize(SRC)  # suit le symlink

    # Garde df (plan Step 9.3 : ~130 Go libres attendus, la copie fait ~10,3 Go).
    st = os.statvfs(ROOT)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    need_gb = src_size / 1e9
    assert free_gb > need_gb + 5, f"df insuffisant : {free_gb:.1f} Go libres pour {need_gb:.1f} Go + marge"

    # --- header safetensors : offsets du tenseur cible (KeyError si absent = fail bruyant) ---
    with open(SRC, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    meta = header[KEY]
    dtype, shape, (o0, o1) = meta["dtype"], meta["shape"], meta["data_offsets"]
    assert dtype == "BF16", f"{KEY} : dtype {dtype} != BF16 (contrat ct w4a16)"
    n_elem = 1
    for d in shape:
        n_elem *= d
    assert o1 - o0 == n_elem * 2, f"offsets incohérents : {o1 - o0} != {n_elem}*2"
    data_start = 8 + hlen
    print(f"cible : {KEY} shape={shape} dtype={dtype} offsets=[{o0},{o1}]", flush=True)

    # --- copie (suit le symlink -> vraie copie plate) puis patch in-place ---
    print(f"copie {need_gb:.1f} Go -> {DST} ...", flush=True)
    shutil.copyfile(SRC, DST)
    assert os.path.getsize(DST) == src_size

    with open(DST, "r+b") as f:
        f.seek(data_start + o0)
        raw = f.read(o1 - o0)
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
        mean_before = float(t.float().abs().mean())
        t = t * FACTOR  # bf16 × scalaire -> bf16 (l'arrondi bf16 est le comportement voulu)
        mean_after = float(t.float().abs().mean())
        f.seek(data_start + o0)
        f.write(t.contiguous().view(torch.uint16).numpy().tobytes())

    # --- vérifications par re-lecture (les DEUX fichiers, voie safetensors officielle) ---
    with safe_open(SRC, framework="pt") as fs, safe_open(DST, framework="pt") as fd:
        s_scale, d_scale = fs.get_tensor(KEY), fd.get_tensor(KEY)
        assert torch.equal(d_scale, (s_scale * FACTOR)), "patch != source×100 (bf16)"
        assert torch.equal(fs.get_tensor(WITNESS), fd.get_tensor(WITNESS)), \
            f"témoin TOUCHÉ : {WITNESS} diffère — patch hors cible"
        assert not torch.equal(s_scale, d_scale), "source == copie : patch sans effet ?"
    ratio = mean_after / mean_before
    print(f"mean|scale| : {mean_before:.6f} -> {mean_after:.4f} (ratio {ratio:.2f}, attendu ~{FACTOR:g})", flush=True)
    assert 95.0 < ratio < 105.0, f"ratio {ratio} hors [95,105]"
    print(f"PASS : {DST} écrit ({KEY} x{FACTOR:g} in-place ; source intacte, témoin bit-égal)", flush=True)

    # --- Politique de décodage (chantier `generation_config`, 29 juil 2026) ---
    # ⚠ LE PROBLÈME QUE CECI RÈGLE : depuis ce chantier, le runner REFUSE de tourner s'il ne
    # trouve pas sa politique (pas de repli silencieux). Or `DST` est écrit À PLAT, sans snapshot
    # ni symlink : les 4 étapes de la découverte échouent, et D11 — un gate historique — devient
    # inexécutable. Il mourrait EN SILENCE au prochain usage, faute d'être lancé souvent.
    # On dépose donc la politique du checkpoint SOURCE à côté du checkpoint corrompu : la
    # découverte nominale (étape 2, `dirname(ckpt)/generation_config.json`) la trouve, et le
    # contre-test compare bien deux runs à politique IDENTIQUE — ce qu'il doit faire, puisqu'il
    # porte sur les POIDS et non sur la politique.
    src_cfg = find_generation_config(SRC)
    dst_cfg = os.path.join(os.path.dirname(DST), CONFIG_NAME)
    if src_cfg is None:
        print(f"⚠ {CONFIG_NAME} INTROUVABLE depuis {SRC} — le runner refusera de tourner sur "
              f"{DST}. Lancer D11 avec --gen-config <chemin explicite>.", flush=True)
    else:
        shutil.copyfile(src_cfg, dst_cfg)
        with open(dst_cfg) as fh:
            keys = sorted(json.load(fh).keys())
        print(f"politique copiée : {src_cfg} -> {dst_cfg} (clés : {', '.join(keys)})", flush=True)

    print("\nD11 — commande de contre-test (le --gen-config est EXPLICITE : il reste juste même "
          "si le fichier ci-dessus disparaît) :", flush=True)
    print(f"  gemma4_g12auto {DST} <tokenizer.json> \\\n"
          f"    --prompt \"<prompt du témoin>\" --oracle <fixture> \\\n"
          f"    --gen-config {dst_cfg if src_cfg else '<dq>/' + CONFIG_NAME}", flush=True)
    print("  ⚠ --gen-config prend un CHEMIN DE FICHIER, jamais un répertoire.", flush=True)


if __name__ == "__main__":
    main()
