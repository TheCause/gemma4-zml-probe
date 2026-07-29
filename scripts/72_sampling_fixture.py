#!/usr/bin/env python3
"""Producteur de la fixture du gate S2-U — warpers de sampling (phase 2).

Spec : docs/superpowers/specs/2026-07-29-sampling-penalty-design.md (rév. 3).
Plan : docs/superpowers/plans/2026-07-29-sampling-phase2.md (rév. 2), Task 2.

POURQUOI CE SCRIPT EXISTE. Retranscrire une formule de warper ferait comparer **deux
transcriptions du même auteur** : une inversion de branche commise des deux côtés passerait le
gate. Ici on instancie les VRAIS `TemperatureLogitsWarper`, `TopKLogitsWarper` et
`TopPLogitsWarper` et on publie LEUR sortie.

⚠ FORMAT 1-D CONCATÉNÉ, PAS `{N,V}`. Les cas ont des vocabulaires de tailles différentes (8, 5,
262 144) : un tenseur rectangulaire est impossible, et le padding serait **fatal et silencieux**
(à 0.0, le cas « 8 logits égaux » deviendrait 262 144 ex æquo ; à -inf, `k_eff` serait évalué sur
262 144 au lieu de 5).

⚠ `compare_mode` PAR CAS. Comparer TOUJOURS par classe d'équivalence serait **aveugle au cas de
bord le plus discriminant** : sur 8 logits égaux avec top_p=0,25, HF garde {6,7} et la formulation
naïve {0,1} — ensembles DISJOINTS, mais **même multiset et même masse**. On compare donc par
INDICES quand c'est licite (logits distincts, ou V <= 128 où `torch.sort` est stable), et par
classe d'équivalence sinon.

Usage : python3 scripts/72_sampling_fixture.py [--out-dir fixtures]
Requiert transformers >= 5.14.1 (venv `g12b`).
"""
import argparse
import json
import os

import numpy as np
import torch
import transformers
from safetensors.numpy import save_file
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

VOCAB_REEL = 262144  # contrat U0 — la taille de la PRODUCTION
# `torch.sort` est stable jusqu'à n = 128 et ne l'est plus à partir de 129 (spec F14, mesuré).
SEUIL_STABLE = 128
TRANSFORMERS_MIN = (5, 14, 1)


def cas_reels():
    """Les cas. Chacun dit ce qu'il DISCRIMINE — un cas qui ne discrimine rien est du décor."""
    cas = []

    # (a) LE cas phare : HF garde {6,7}, le naïf descendant garde {0,1} — DISJOINTS.
    cas.append(dict(name="topp_disjoint_8egaux", warper="top_p",
                    logits=[0.0] * 8, params=dict(top_p=0.25, min_tokens_to_keep=1),
                    discrimine="tri ascendant + `<=` : ensembles DISJOINTS du naïf descendant"))

    # (b) min_tokens_to_keep mordant : il repêche la QUEUE du tri ascendant.
    for mk in (1, 2, 3):
        cas.append(dict(name=f"topp_minkeep_{mk}", warper="top_p",
                        logits=[10.0, 0.0, 0.0, 0.0, 0.0],
                        params=dict(top_p=0.1, min_tokens_to_keep=mk),
                        discrimine="min_keep protège la queue du tri ASCENDANT, pas le début"))

    # (c) ex æquo du k-ième : 5 survivants pour k=3.
    cas.append(dict(name="topk_exaequo_kieme", warper="top_k",
                    logits=[3.0, 2.0, 1.0, 1.0, 1.0], params=dict(top_k=3, min_tokens_to_keep=1),
                    discrimine="élimination STRICTE : top_k ne borne pas le nb de survivants"))
    cas.append(dict(name="topk_64egaux_k8", warper="top_k",
                    logits=[0.0] * 64, params=dict(top_k=8, min_tokens_to_keep=1),
                    discrimine="64 survivants pour k=8 — le cas extrême de (c)"))

    # (d) ÉCHELLE RÉELLE — sans lui le gate passe à vide (F14).
    rng = np.random.default_rng(20260729)
    big = rng.normal(0, 5, VOCAB_REEL).astype(np.float32)
    cas.append(dict(name="topk_vocab_reel", warper="top_k", logits=big.tolist(),
                    params=dict(top_k=64, min_tokens_to_keep=1),
                    discrimine="V = 262 144 : la taille de la PRODUCTION (torch.sort n'y est pas stable)"))
    cas.append(dict(name="topp_vocab_reel", warper="top_p", logits=big.tolist(),
                    params=dict(top_p=0.95, min_tokens_to_keep=1),
                    discrimine="idem, chemin top_p"))

    # (e) la température CHANGE le nombre de survivants (F8a), et peut créer une égalité (F8b).
    cas.append(dict(name="temp_puis_topk", warper="temp_top_k",
                    logits=[100.0, 12.5308094, 12.5308104, 1.0],
                    params=dict(temperature=0.7, top_k=2, min_tokens_to_keep=1),
                    discrimine="÷T fusionne deux logits en f32 => 3 survivants au lieu de 2"))

    # (f) preuve de l'optimisation « ne trier que les non-filtrés » : top_k PUIS top_p.
    cas.append(dict(name="topk_puis_topp", warper="top_k_top_p",
                    logits=big.tolist(), params=dict(top_k=64, top_p=0.95, min_tokens_to_keep=1),
                    discrimine="chaîne réelle de HF : top_k(19) AVANT top_p(20)"))
    return cas


def applique(c):
    """Applique le ou les VRAIS warpers HF. Rend le vecteur de sortie."""
    x = torch.tensor([c["logits"]], dtype=torch.float32)
    ids = torch.zeros((1, 1), dtype=torch.long)
    p = c["params"]
    mk = p.get("min_tokens_to_keep", 1)
    w = c["warper"]
    if w in ("temp_top_k",):
        x = TemperatureLogitsWarper(float(p["temperature"]))(ids, x)
    if w in ("top_k", "temp_top_k", "top_k_top_p"):
        x = TopKLogitsWarper(int(p["top_k"]), min_tokens_to_keep=mk)(ids, x)
    if w in ("top_p", "top_k_top_p"):
        x = TopPLogitsWarper(float(p["top_p"]), min_tokens_to_keep=mk)(ids, x)
    return x[0]


def mode_comparaison(logits, out):
    """`indices` quand c'est licite, `equivalence` sinon (cf. en-tête)."""
    n = len(logits)
    distincts = len(set(logits)) == n
    return "indices" if (distincts or n <= SEUIL_STABLE) else "equivalence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="fixtures")
    args = ap.parse_args()

    ver = tuple(int(v) for v in transformers.__version__.split(".")[:3])
    assert ver >= TRANSFORMERS_MIN, f"transformers {transformers.__version__} < 5.14.1"

    cas = cas_reels()
    logits_all, mask_all, offsets, meta = [], [], [0], []
    n_indices = n_equiv = 0
    n_topk_deborde = 0   # antécédent : top_k laisse PLUS de k survivants
    n_vocab_reel = 0     # antécédent : au moins un cas à l'échelle de production

    for c in cas:
        out = applique(c).numpy()
        surv = np.isfinite(out).astype(np.uint8)
        mode = mode_comparaison(c["logits"], out)
        n_indices += mode == "indices"
        n_equiv += mode == "equivalence"

        n_surv = int(surv.sum())
        if "top_k" in c["params"] and n_surv > int(c["params"]["top_k"]):
            n_topk_deborde += 1
        if len(c["logits"]) == VOCAB_REEL:
            n_vocab_reel += 1

        logits_all.append(np.asarray(c["logits"], dtype=np.float32))
        mask_all.append(surv)
        offsets.append(offsets[-1] + len(c["logits"]))
        meta.append(dict(name=c["name"], warper=c["warper"], params=c["params"],
                         compare_mode=mode, n_survivants=n_surv,
                         discrimine=c["discrimine"]))

    # --- NON-VACUITÉ, assertée côté producteur. Une fixture qui n'exerce rien ferait rendre
    # --- 100 % au gate sans rien prouver.
    assert n_indices >= 1, "aucun cas comparable par INDICES : on retomberait sur la règle aveugle"
    assert n_topk_deborde >= 1, "aucun cas où top_k laisse plus de k survivants (F9 non exercé)"
    assert n_vocab_reel >= 1, "aucun cas à V=262144 : le gate passerait à vide (F14)"

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "s2_cases.safetensors")
    save_file({
        "logits_in": np.concatenate(logits_all),
        "mask_expected": np.concatenate(mask_all),
        "offsets": np.asarray(offsets, dtype=np.int64),
    }, out_path)

    manifest = dict(
        version=1, produced_by="scripts/72_sampling_fixture.py",
        versions=dict(transformers=transformers.__version__, torch=torch.__version__),
        vocab_reel=VOCAB_REEL, seuil_sort_stable=SEUIL_STABLE,
        n_cases=len(cas), cases=meta,
        antecedents=dict(n_indices=n_indices, n_equivalence=n_equiv,
                         n_topk_deborde=n_topk_deborde, n_vocab_reel=n_vocab_reel),
    )
    with open(out_path + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"transformers {transformers.__version__} — {len(cas)} cas")
    for m in meta:
        print(f"  {m['name']:26s} {m['compare_mode']:12s} survivants={m['n_survivants']:6d}  {m['discrimine']}")
    print(f"antécédents : indices={n_indices} équivalence={n_equiv} "
          f"topk_déborde={n_topk_deborde} vocab_réel={n_vocab_reel}")
    print(f"écrit : {out_path} (+ .manifest.json)")


if __name__ == "__main__":
    main()
