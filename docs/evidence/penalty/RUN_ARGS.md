# RUN_ARGS de référence — chantier repetition penalty (phase 1)

Figés le 10 août 2026, Task 1 Step 3 du plan
`docs/superpowers/plans/2026-08-10-dettes-restantes-penalty.md`.
**À réutiliser à l'identique en RP2** (non-régression penalty neutre) et en RP3/RP7.

## Binaire et poids

| Élément | Valeur |
|---|---|
| Cible | `//examples/rqz:gemma4_g12auto` (variante 1280) |
| Build | `zml_runner/build_3090.sh` — bannière vérifiée `BUILD: mode=ReleaseFast` |
| Poids | `weights_12b/model.safetensors` (`gemma-4-12B-it-qat-w4a16-ct`, snapshot `1d2c2d7f…`) |

## Témoin HLO (RP0)

```bash
XLA_FLAGS="--xla_dump_to=<dir>" $B1 $W/model.safetensors $W/tokenizer.json \
  --prompt "witness" --max-tokens 4
md5sum <dir>/*before_optimizations.txt
```

md5 `before_optimizations` mesuré : **`297679847aa04b719942d75d093adf2b`** — identique au
témoin stable des 5 chantiers précédents (GC0 27 juil, S2-G, DC0 9 août, G-D0 10 août).
512 fichiers dumpés. `ALLOC-LOOP: alloc=0` sur 17 steps.

Le témoin 4k (`gemma4_g12a4k`) n'est PAS re-capturé ici : la penalty n'est câblée que dans
`gemma4_g12auto`. Sa valeur de référence reste `704de4bc1999f5956724b184bd097ce6`
(`docs/evidence/kvdump/hlo_witness.md5`).

## Témoin d'ids long (RP2)

```bash
$B1 $W/model.safetensors $W/tokenizer.json \
  --prompt "Tell me the story of the number zero, from its invention to modern mathematics." \
  --max-tokens 200 --out-ids <f>.safetensors
```

Mesuré le 10 août 2026 : **200 ids générés** (pas d'EOT prématuré — le seuil de vacuité du
plan, `n < 50` → changer de prompt, n'est pas approché). `ALLOC-LOOP: alloc=0` sur 228 steps
(28 de prefill + 200 de génération). Témoin conservé hors arbre : `logs/rp_witness_long.safetensors`
sur M1, `/data/gemma4-zml-probe/rp_witness_long.safetensors` sur la VM (868 octets).

> ⚠ **Ce témoin n'est PAS une référence inter-fenêtre.** Mesuré le même jour : trois binaires
> distincts — dont `main` recompilé sans une ligne du chantier — s'accordent sur des ids
> DIFFÉRENTS de ce témoin, à graphe (md5 HLO identique), poids et prompt identiques.
> Ce qui reste valable ici, ce sont les **RUN_ARGS** (prompt + `--max-tokens 200`) ;
> ce qui est retiré, c'est la valeur du témoin comme référence d'un run ultérieur.
> RP2 se juge donc contre un run de `main` recompilé **dans la même fenêtre**.
> Détail et mesures : `FINDING_temoin_ids_non_reproductible.md`.

Pourquoi ce prompt et pas le prompt canonique du repo : ce dernier fait EOT au 2ᵉ token —
deux ids n'exercent aucune répétition, RP2 y passerait à vide (leçon « vacuité de l'antécédent »).
