"""Partagé oracles 12B (scripts 65/66/67) : lecture de l'export dq 2-shards (D9).

Extrait au 3e usage (script 67, Task 6) — règle du plan J2 : « Si tu recopies
`load_dq_weights` une 3e fois : c'est le seuil — extrais `scripts/_oracle_common.py` ».
Les scripts 65/66 migrés dessus ; non-régression = re-run + diff JSON des manifests
(0 champ changé attendu, même graine, mêmes versions).
"""
import json
import os

ROOT = "/data/gemma4-zml-probe"
DQ = os.path.join(ROOT, "weights_12b_dq")  # export dq (D8/D9) — PAS le packé
FX = os.path.join(ROOT, "fixtures")


def load_dq_weights(keys, dq_dir=DQ):
    """Lit les clés depuis l'export dq 2-shards via l'index JSON (safe_open par shard)."""
    from safetensors import safe_open

    with open(os.path.join(dq_dir, "model.safetensors.index.json")) as fh:
        wmap = json.load(fh)["weight_map"]
    out, by_shard = {}, {}
    for k in keys:
        by_shard.setdefault(wmap[k], []).append(k)
    for shard, ks in by_shard.items():
        with safe_open(os.path.join(dq_dir, shard), framework="pt") as f:
            for k in ks:
                out[k] = f.get_tensor(k)
    return out
