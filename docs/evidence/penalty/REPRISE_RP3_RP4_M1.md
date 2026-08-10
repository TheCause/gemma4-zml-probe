# Reprise — RP3, RP4 et M1 (les 3 gates restants de la phase 1 penalty)

**Pourquoi ce document.** Le 10 août 2026, à la clôture du chantier penalty, la 3090 était
occupée par un autre travail (ComfyUI, 15,4 GiB — le 12B en demande 20). Les trois gates
restants sont **prêts à exécuter** : fixtures produites, mordant mesuré, code en place. Ce
document donne les commandes exactes, pour qu'aucune décision ne soit à reprendre.

**Durée estimée :** ~20 min de GPU (2 runs RP3, 3 builds + 3 runs RP4, 2 runs M1).

## Préalable — vérifier que la carte est libre

```bash
ssh $ZML_REMOTE 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'
```

Vide = libre. Sinon, identifier le process avant toute action (`ps -o pid,etime,user,cmd -p <pid>`)
et **ne jamais tuer un process qui n'est pas le nôtre** — la garde `GpuBusy` du runner nomme déjà
l'occupant, c'est elle qui a rendu ce report propre.

Vérifier aussi que le binaire correspond bien à la branche :

```bash
ZML_REMOTE=user@gpu-host ZML_DST=/data/rqz_workspace/zml/examples/rqz ./zml_runner/deploy_to_3090.sh
ZML_REMOTE=user@gpu-host ZML_WS=/data/rqz_workspace/zml ./zml_runner/build_3090.sh
```

⚠ **Capturer le md5 du binaire** (`md5sum bazel-bin/examples/rqz/gemma4_g12auto`) et le consigner
avec les résultats — c'est la leçon du finding `FINDING_temoin_ids_non_reproductible.md`.

## RP3 — le gate : `ids == HF` sous penalty

Les fixtures sont sur la VM (`/data/gemma4-zml-probe/fixtures/oracle_rp{1.15,0.8}.safetensors`)
et leurs manifests sont committés. Elles portent `prompt_ids` : le runner vérifie désormais le
prompt **littéralement**, pas seulement sa longueur.

```bash
B1=./bazel-bin/examples/rqz/gemma4_g12auto ; W=/data/gemma4-zml-probe/weights_12b
P="Tell me the story of the number zero, from its invention to modern mathematics."
for RP in 1.15 0.8; do
  $B1 $W/model.safetensors $W/tokenizer.json --prompt "$P" \
    --oracle /data/gemma4-zml-probe/fixtures/oracle_rp${RP}.safetensors \
    --repetition-penalty $RP
done
```

**Attendu :** `A1 PASS — 48/48`, `PENALTY: … n_penalty_touched > 0`, et la ligne
`--oracle : prompt vérifié LITTÉRALEMENT (29 ids identiques à la fixture)`.

**Antécédent déjà mesuré** (le gate ne peut pas passer à vide) : hamming(`fed` 1.0, `fed` 1.15)
= **31/48**, hamming(1.0, 0.8) = **25/48**.

**Contre-test de non-vacuité à faire aussi** — la même fixture 1.15 **sans** armer la penalty
doit **FAIL** : sinon la fixture ne se distinguerait pas d'une trajectoire neutre.

**En cas de mismatch sans cause côté penalty**, appliquer la procédure §7-3 de la spec (3
conditions, ε = 2e-3) **et** relire `FINDING_temoin_ids_non_reproductible.md` : le manifest
`oracle_rp1.0` porte `min_margin = 0,004589 @ gen=47`, donc cette trajectoire passe par une
zone où la sélection ne tient qu'à 0,0046. Un mismatch tardif est à instruire avec cette
grandeur en main, pas à attribuer d'emblée au code.

## RP4 — les trois corruptions, chacune vue FAIL

Dans un **worktree jetable** (jamais committé), rebuild et re-run de RP3 (fixture 1.15) après
chaque corruption. Publier le **mordant** de chacune (nombre d'ids qui divergent ; plancher 1).

| # | Corruption dans `zml_runner/sampling.zig` / le câblage | Ce qu'elle doit tuer |
|---|---|---|
| a | branches de signe échangées : `if (v < 0) v / penalty else v * penalty` | la formule HF elle-même |
| b | déduplication retirée (ignorer `seen`, pénaliser chaque occurrence) | l'application « au plus une fois par token distinct » |
| c | `ignore_prompt` forcé à l'inverse au point d'insertion | la définition de l'historique (prompt inclus) |

⚠ Un mordant **faible** face à la cascade attendue (~30 ids sur 48) est à **instruire**, pas à
accepter : cela signifierait que la corruption ne mord presque pas, donc que le gate discrimine
mal. Les corruptions restent dans le worktree et ne sont **jamais** committées.

## M1 — coût, mesure publiée (pas un gate)

⚠ Le chemin B doit être **armé dans les deux runs**, sinon `M-COUT` n'est pas publié côté OFF et
il n'y a rien à comparer : `--repetition-penalty 1.0` n'arme pas. On arme les deux par
`--top-k 1` (le régime neutre du gate-pont), ce qui fait de la penalty la **seule** variable.

⚠ **Ne pas armer `--gate-d1d2`** : il travaille dans la fenêtre chronométrée et l'invalide — le
binaire le dit lui-même par un `log.warn`.

```bash
$B1 $W/model.safetensors $W/tokenizer.json --prompt "$P" --max-tokens 200 --top-k 1 \
  --out-ids /data/gemma4-zml-probe/t7_off.safetensors
$B1 $W/model.safetensors $W/tokenizer.json --prompt "$P" --max-tokens 200 --top-k 1 \
  --repetition-penalty 1.15 --out-ids /data/gemma4-zml-probe/t7_on.safetensors
```

Publier les deux lignes `M-COUT` (moyenne µs/step, D2H seul, warpers) et vérifier
`BUILD: mode=ReleaseFast` aux deux.

## RP7 — complément à longueur égale

La mesure RP7 est **déjà publiée** sur les trajectoires HF de 48 tokens (`SAMPLING_RESULTS.md`
§8.1). Les deux fichiers d'ids produits par M1 permettent de la refaire **à 200 tokens et à
longueur égale**, ce qui est plus comparable :

```bash
python3 scripts/77_ngram_repetition.py logs/t7_off.safetensors logs/t7_on.safetensors
```

## À faire après

1. Compléter la table §8.1 de `docs/SAMPLING_RESULTS.md` (les trois lignes `EN ATTENTE`).
2. Tags : `gate/rp3-pass`, `gate/rp4-pass`, et `gate/rp7-pass` **seulement si** D4 avait retenu
   un gate — ce n'est pas le cas (RP7 est une mesure publiée, décision du 10 août).
3. Retirer la mention « still pending a free GPU window » du `README.md`.
