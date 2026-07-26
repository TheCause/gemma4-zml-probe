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


if __name__ == "__main__":
    main()
