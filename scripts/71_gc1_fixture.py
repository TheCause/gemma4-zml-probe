#!/usr/bin/env python3
"""Producteur de la fixture du gate GC1 — politique de décodage `generation_config.json`.

Spec : docs/superpowers/specs/2026-07-28-generation-config-design.md §4.5bis.

POURQUOI CE SCRIPT EXISTE. GC1 valide la sélection host-side du runner (`gencfg.GenCfg.select`,
qui choisit dans un top-5 pré-trié) contre la sémantique de HF (qui écrit `-inf` sur le vecteur
COMPLET puis fait `argmax` sur les 262 144 logits). Ces deux choses coïncident par un argument de
rang (§4.2) — un argument est un raisonnement, pas une mesure. Ce producteur le transforme en
mesure : `expect_tok` est calculé par le VRAI `SuppressTokensLogitsProcessor` de transformers
appliqué à un vecteur complet, jamais par une réimplémentation de notre côté. C'est précisément
ce qui rend la claim C2 falsifiable au lieu d'être postulée.

⚠ ANGLE MORT QUE CE SCRIPT NE DOIT PAS REPRODUIRE : le finding du 27 juil
(docs/FINDING_GENERATION_CONFIG.md) a été invisible pendant tout le portage parce que l'oracle
`69_u8_gen_oracle.py` faisait le MÊME argmax nu que le runner — l'instrument était aveugle au même
endroit que son sujet. D'où la règle ici : on importe le processor de transformers, on n'écrit
jamais « suppression » à la main.

Sorties (fixtures/, régénérables — `fixtures/*.safetensors` est gitignored) :
  - gc1_cases.safetensors               : cas de SÉLECTION (top5_idx/top5_val/expect_tok/expect_rank)
  - gc1_cases.safetensors.manifest.json : cas de VALIDATION + de DÉCOUVERTE + la politique
  - gc1_discovery_setup.sh              : fabrique les topologies de symlinks du 3ᵉ véhicule
                                          (à lancer SUR LA MACHINE DU SELFTEST — les chemins du
                                          manifest y font référence)

Usage : python3 scripts/71_gc1_fixture.py [--out-dir fixtures] [--discovery-root /tmp/gc1_disco]
Requiert transformers ≥ 5.14.1 (venv `g12b`) — la version est consignée au manifest.
"""
import argparse
import json
import os

import numpy as np
import torch
import transformers
from safetensors.numpy import save_file
from transformers.generation.logits_process import SuppressTokensLogitsProcessor

# Contrat 12B (docs/U_12B_CONTRACT.md) — `tokenizer.json` du snapshot : 262 144 entrées.
VOCAB = 262144
# MESURÉ depuis le tokenizer par le runner (`<turn|>`), jamais hardcodé côté production. Ici c'est
# une DONNÉE de fixture : le selftest tourne sans tokenizer (early-return avant son chargement).
EOT_ID = 106
# `google/gemma-4-12B-it/generation_config.json`, tel que Google le publie.
SUPPRESS = [258883, 258882]
EOS = [1, 106, 50]
TOP_K = 5  # doit valoir gencfg.TOP_K — un désaccord ferait échouer GC1 sur les formes, bruyamment

# Cas de sélection. `ids`/`vals` sont les cinq premiers rangs de l'ordre décroissant, tels que le
# runner les rapatrie du device.
#
# Le premier cas est le cas RÉEL du finding : top-5 fp32 mesuré à la position 57 du témoin 200
# (docs/FINDING_GENERATION_CONFIG.md §5, logits post-softcap). Un selftest qui n'exercerait que
# des cas fabriqués ne dirait rien du seul cas dont on sait qu'il s'est produit en vrai.
SELECTION_CASES = [
    {
        "name": "reel_pos57_du_temoin",  # 258882 top-1, marge 0,2499 — le cas qui a ouvert le chantier
        "ids": [258882, 11814, 3495, 1548, 13186],
        "vals": [19.1645, 18.9147, 18.6035, 18.0667, 17.9584],
    },
    {
        "name": "aucun_supprime",  # cas nominal : la politique ne doit RIEN changer
        "ids": [11814, 3495, 1548, 13186, 220],
        "vals": [19.0, 18.5, 18.0, 17.5, 17.0],
    },
    {
        "name": "supprime_rang2_inoffensif",  # 258882 présent mais sous le top-1 : rang 0 conservé
        "ids": [11814, 258882, 3495, 1548, 13186],
        "vals": [19.0, 18.5, 18.0, 17.5, 17.0],
    },
    {
        "name": "supprime_rang5_inoffensif",
        "ids": [11814, 3495, 1548, 13186, 258883],
        "vals": [19.0, 18.5, 18.0, 17.5, 17.0],
    },
    {
        "name": "deux_supprimes_en_tete",  # rang brut 2 = le pire cas admis par la garde |S|+1 ≤ TOP_K
        "ids": [258882, 258883, 11814, 3495, 1548],
        "vals": [19.5, 19.2, 18.9, 18.5, 18.0],
    },
    {
        "name": "deux_supprimes_disjoints",  # 258882 top-1, 258883 rang 3 : le rang retenu reste 1
        "ids": [258882, 11814, 258883, 3495, 1548],
        "vals": [19.5, 19.2, 18.9, 18.5, 18.0],
    },
    {
        "name": "egalite_exacte_top1_supprime",
        # Réserve documentée §4.2 : sur égalité EXACTE, le tie-break de `sort` peut différer de
        # celui d'`argmax`. Ici le rang 0 est SUPPRIMÉ, donc l'argmax post-suppression est sans
        # ambiguïté — le cas exerce l'égalité sans faire dépendre le gate d'un tie-break non
        # spécifié. Ce qui reste non vérifié (tie NON supprimé au sommet) est écrit comme tel
        # dans la spec §5 « ce qui ne sera PAS prouvé ».
        "ids": [258882, 11814, 3495, 1548, 13186],
        "vals": [19.1645, 19.1645, 18.6035, 18.0667, 17.9584],
    },
    {
        "name": "egalite_exacte_en_queue",  # égalité rangs 3/4, loin du sommet : compteur de ties nourri
        "ids": [11814, 3495, 1548, 13186, 220],
        "vals": [19.0, 18.5, 18.0, 18.0, 17.0],
    },
]

# Cas de validation (§4.1). `content` est le JSON LITTÉRAL : le selftest doit exercer le parser sur
# du texte, y compris malformé — pas sur un objet re-sérialisé par nos soins.
VALIDATION_CASES = [
    {
        "name": "nominal_gemma4_12b",
        "eot_id": EOT_ID,
        "content": json.dumps(
            {
                "eos_token_id": EOS,
                "suppress_tokens": SUPPRESS,
                "do_sample": True,
                "top_k": 64,
                "top_p": 0.95,
                "temperature": 1.0,
                "bos_token_id": 2,
                "pad_token_id": 0,
            }
        ),
        "expect_error": "",
        "expect_suppress_len": 2,
    },
    {
        "name": "eot_present_dans_eos",  # contrôle croisé exercé dans le sens ACCEPTÉ (§4.1)
        "eot_id": EOT_ID,
        "content": json.dumps({"eos_token_id": EOS, "suppress_tokens": []}),
        "expect_error": "",
        "expect_suppress_len": 0,
    },
    {
        "name": "eot_absent_des_eos",  # ... et dans le sens REFUSÉ
        "eot_id": EOT_ID,
        "content": json.dumps({"eos_token_id": [1, 50], "suppress_tokens": SUPPRESS}),
        "expect_error": "EotNotInEosList",
    },
    {
        "name": "eos_scalaire_normalise",  # HF normalise int → [int] (stopping_criteria.py:544-549)
        "eot_id": EOT_ID,
        "content": json.dumps({"eos_token_id": EOT_ID, "suppress_tokens": SUPPRESS}),
        "expect_error": "",
        "expect_suppress_len": 2,
    },
    {
        "name": "eos_liste_vide",
        "eot_id": EOT_ID,
        "content": json.dumps({"eos_token_id": [], "suppress_tokens": []}),
        "expect_error": "EosListEmpty",
    },
    {
        "name": "eos_absent",
        "eot_id": EOT_ID,
        "content": json.dumps({"suppress_tokens": SUPPRESS}),
        "expect_error": "EosListEmpty",
    },
    {
        "name": "begin_suppress_present",
        "eot_id": EOT_ID,
        "content": json.dumps(
            {"eos_token_id": EOS, "suppress_tokens": SUPPRESS, "begin_suppress_tokens": [220]}
        ),
        "expect_error": "BeginSuppressUnsupported",
    },
    {
        "name": "suppress_id_hors_vocab",
        "eot_id": EOT_ID,
        "content": json.dumps({"eos_token_id": EOS, "suppress_tokens": [VOCAB]}),
        "expect_error": "SuppressIdOutOfRange",
    },
    {
        "name": "suppress_id_negatif",  # HF l'ignorerait en silence (isin sur arange(V)) ; on refuse
        "eot_id": EOT_ID,
        "content": json.dumps({"eos_token_id": EOS, "suppress_tokens": [-1]}),
        "expect_error": "SuppressIdOutOfRange",
    },
    {
        "name": "suppress_doublons",  # sans effet sur HF, mais fausserait la garde |S|+1 ≤ TOP_K
        "eot_id": EOT_ID,
        "content": json.dumps(
            {"eos_token_id": EOS, "suppress_tokens": [258882, 258883, 258882, 258883]}
        ),
        "expect_error": "",
        "expect_suppress_len": 2,
        "is_dedup_case": True,
    },
    {
        "name": "suppress_trop_long_pour_top_k",  # 5 ids ⇒ rang brut possible 6 > TOP_K
        "eot_id": EOT_ID,
        "content": json.dumps(
            {
                "eos_token_id": EOS,
                "suppress_tokens": [258883, 258882, 258880, 258881, 258884],
            }
        ),
        "expect_error": "SuppressListTooLongForTopK",
    },
    {
        "name": "json_malforme",
        "eot_id": EOT_ID,
        "content": '{"eos_token_id": [1, 106, 50], "suppress_tokens": [',
        "expect_error": "GenerationConfigInvalid",
    },
]


def build_selection_fixture():
    """Calcule `expect_tok` avec le VRAI processor HF, sur le vecteur COMPLET de 262 144 logits."""
    proc = SuppressTokensLogitsProcessor(SUPPRESS, device="cpu")
    n = len(SELECTION_CASES)
    top5_idx = np.zeros((n, TOP_K), dtype=np.int32)
    top5_val = np.zeros((n, TOP_K), dtype=np.float32)
    expect_tok = np.zeros((n,), dtype=np.int32)
    expect_rank = np.zeros((n,), dtype=np.int32)

    for c, case in enumerate(SELECTION_CASES):
        ids, vals = case["ids"], case["vals"]
        assert len(ids) == TOP_K and len(vals) == TOP_K, case["name"]
        assert all(vals[j] >= vals[j + 1] for j in range(TOP_K - 1)), (
            f"{case['name']} : le top-5 rapatrié est TRIÉ décroissant — une fixture non triée "
            "testerait un contrat que le device ne fournit pas"
        )
        assert len(set(ids)) == TOP_K, f"{case['name']} : ids dupliqués dans le top-5"

        # Fond très bas et STRICTEMENT décroissant en id : aucun ex æquo involontaire avec le
        # top-5, et un argmax de fond parfaitement déterminé si jamais tout le top-5 disparaissait.
        scores = torch.full((1, VOCAB), -1e4, dtype=torch.float32)
        scores[0, :] -= torch.arange(VOCAB, dtype=torch.float32) * 1e-3
        for j in range(TOP_K):
            scores[0, ids[j]] = vals[j]

        suppressed = proc(torch.zeros((1, 1), dtype=torch.long), scores.clone())
        tok = int(torch.argmax(suppressed[0]).item())

        # Le token retenu DOIT être dans le top-5 rapatrié : c'est l'argument de rang du §4.2. S'il
        # n'y est pas, le design host-side est faux et le gate doit mourir ici, pas au runtime.
        assert tok in ids, (
            f"{case['name']} : argmax post-suppression = {tok}, ABSENT du top-5 {ids} — "
            "l'argument de rang de la spec §4.2 est réfuté"
        )
        top5_idx[c] = ids
        top5_val[c] = vals
        expect_tok[c] = tok
        expect_rank[c] = ids.index(tok)

    # Non-vacuité CÔTÉ PRODUCTEUR : une fixture où la suppression ne mord jamais ferait passer GC1
    # sans rien prouver. Le selftest a le même contrôle ; les deux doivent tenir.
    n_bite = int((expect_rank != 0).sum())
    assert n_bite > 0, "aucun cas où la suppression mord : la fixture serait vacueuse"
    return top5_idx, top5_val, expect_tok, expect_rank, n_bite


def discovery_cases(root):
    """Les 4 topologies exercées. Les chemins sont ceux que `gc1_discovery_setup.sh` fabrique."""
    return [
        {
            "name": "direct_a_cote_du_ckpt",  # étape 2 de la règle §4.1
            "ckpt": f"{root}/plain/model.safetensors",
            "expect_path": f"{root}/plain/generation_config.json",
        },
        {
            "name": "symlink_absolu_1_hop",  # topologie RÉELLE de weights_12b/ (mesurée)
            "ckpt": f"{root}/weights_abs/model.safetensors",
            "expect_path": f"{root}/snap_abs/generation_config.json",
        },
        {
            "name": "symlink_relatif_1_hop",  # topologie du cache HF (`../../blobs/…`, mesurée)
            "ckpt": f"{root}/weights_rel/model.safetensors",
            "expect_path": f"{root}/snap_rel/generation_config.json",
        },
        {
            "name": "introuvable_fichier_regulier",  # pas de repli silencieux : le runner refuse
            "ckpt": f"{root}/orphan/model.safetensors",
            "expect_error": "GenerationConfigNotFound",
        },
    ]


SETUP_SH = """#!/usr/bin/env bash
# Fabrique les topologies du 3ᵉ véhicule de GC1 (découverte 1-hop) — spec §4.5bis.
# À lancer SUR LA MACHINE DU SELFTEST : le manifest référence ces chemins en absolu.
# Idempotent (rm -rf de la racine au début). Aucun selftest du repo ne crée de symlink : la
# topologie se fabrique ici, pas dans le Zig.
set -euo pipefail
ROOT="${1:-%ROOT%}"
rm -rf "$ROOT"
mkdir -p "$ROOT"/{plain,weights_abs,snap_abs,weights_rel,snap_rel,orphan}

CFG='{"eos_token_id": [1, 106, 50], "suppress_tokens": [258883, 258882]}'

# (2) fichier régulier, config à côté
: > "$ROOT/plain/model.safetensors"
echo "$CFG" > "$ROOT/plain/generation_config.json"

# (3a) symlink ABSOLU vers un snapshot — topologie de weights_12b/ (mesurée)
: > "$ROOT/snap_abs/model.safetensors"
echo "$CFG" > "$ROOT/snap_abs/generation_config.json"
ln -s "$ROOT/snap_abs/model.safetensors" "$ROOT/weights_abs/model.safetensors"

# (3b) symlink RELATIF — topologie du cache HF (`tokenizer.json -> ../../blobs/…`)
: > "$ROOT/snap_rel/model.safetensors"
echo "$CFG" > "$ROOT/snap_rel/generation_config.json"
ln -s "../snap_rel/model.safetensors" "$ROOT/weights_rel/model.safetensors"

# (4) fichier régulier SANS config nulle part : doit échouer, pas se replier
: > "$ROOT/orphan/model.safetensors"

echo "topologies GC1 prêtes sous $ROOT"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="fixtures")
    ap.add_argument("--discovery-root", default="/tmp/gc1_disco")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    top5_idx, top5_val, expect_tok, expect_rank, n_bite = build_selection_fixture()

    fixture = os.path.join(args.out_dir, "gc1_cases.safetensors")
    save_file(
        {
            "top5_idx": top5_idx,
            "top5_val": top5_val,
            "expect_tok": expect_tok,
            "expect_rank": expect_rank,
        },
        fixture,
    )

    manifest = {
        "version": 1,
        "produced_by": "scripts/71_gc1_fixture.py",
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
        "selection": {
            "vocab_size": VOCAB,
            "eot_id": EOT_ID,
            "suppress_tokens": SUPPRESS,
            "eos_token_id": EOS,
            "n_cases": len(SELECTION_CASES),
            "n_cases_ou_la_suppression_mord": n_bite,
            "case_names": [c["name"] for c in SELECTION_CASES],
        },
        "validation_cases": VALIDATION_CASES,
        "discovery_cases": discovery_cases(args.discovery_root),
    }
    with open(fixture + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    setup = os.path.join(args.out_dir, "gc1_discovery_setup.sh")
    with open(setup, "w") as f:
        f.write(SETUP_SH.replace("%ROOT%", args.discovery_root))
    os.chmod(setup, 0o755)

    print(f"transformers {transformers.__version__} — SuppressTokensLogitsProcessor({SUPPRESS})")
    print(f"{fixture} : {len(SELECTION_CASES)} cas de sélection, {n_bite} où la suppression mord")
    for c, case in enumerate(SELECTION_CASES):
        print(
            f"  {case['name']:32s} top1={case['ids'][0]:6d} -> tok={int(expect_tok[c]):6d} "
            f"rang={int(expect_rank[c])}"
        )
    print(f"{fixture}.manifest.json : {len(VALIDATION_CASES)} validations, "
          f"{len(manifest['discovery_cases'])} découvertes")
    print(f"{setup} : à lancer sur la machine du selftest (racine {args.discovery_root})")


if __name__ == "__main__":
    main()
