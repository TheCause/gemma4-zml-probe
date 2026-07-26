# Mode résident `--repl` — Résultats (chantier du 26 juillet 2026, soir)

> Spec (spec+plan combinés, niveau léger) : `docs/superpowers/specs/2026-07-26-repl-mode-design.md`
> Base : PR #16 mergée. Demande : « fais le mode résident ». Branche `repl-mode`.

## Verdict en une ligne

**Le 12B se prompte en boucle sur la 3090** : load + compile payés une fois (~1 min 40),
puis chaque prompt coûte ~3 s de prefill + la génération (~9 tok/s) — usage :

```bash
./bazel.sh run --@zml//platforms:cuda=true //examples/rqz:gemma4_g12auto -- \
  <weights_12b>/model.safetensors <weights_12b>/tokenizer.json --repl [--max-tokens N]
# puis : un prompt par ligne sur stdin ; ligne vide ou EOF pour quitter.
```

## Design livré

- `generateOnce` : la génération complète extraite de `run()` — gardes de longueur PAR
  prompt, **cache zéros par prompt** (V1 : prompts indépendants, position 0 — pas de
  multi-tour), boucle steps, verdicts. **Un seul chemin de code** one-shot/résident (les
  modes `--oracle`/`--out-ids` passent par la même fonction ; `--repl` en est exclusif).
- `promptToIds` : tokenisation par prompt (`rendered` sur allocator + free — fix revue :
  l'ancien passage par l'arena aurait crû à chaque prompt résident).
- `engine.zig` : 0 octet.

## Gates

| Gate | Critère | Mesure | Verdict |
|---|---|---|---|
| **R0** non-régression one-shot | 48/48 ids == témoin | `cmp` vide (le refactor generateOnce ne change pas la génération) | **PASS** |
| **R1** équivalence résident | session 3 prompts [A, Canberra, A] | 3 générations ; **prompt 1 == prompt 3 == témoin** (aucun état caché entre prompts : encoder reset, cache zéros, step 0) ; « Canberra » au prompt 2 | **PASS** |
| **R2** stabilité mémoire | 20 prompts pipés, polls 5 s | **20/20 réponses** (stdout), toutes identiques ; **VRAM plate 22 234 MiB** ; **RSS en plateau** post-compile : +1 Mo TOTAL sur 20 prompts (~50 Ko/prompt, bruit allocateur — pas de fuite structurelle) | **PASS** |

## Les 2 bugs attrapés par R1 (1ʳᵉ passe) — pièges std.Io 0.16 à retenir

1. **`takeDelimiterExclusive` ne consomme PAS le délimiteur** (Reader.zig:872 :
   `toss(result.len)` — le `\n` reste en tête) → le tour suivant lit une ligne vide →
   sortie prématurée après UN prompt, **exit 0, « sortie propre »** : un faux succès que
   seul le COMPTAGE des générations (3 attendues, 1 trouvée) a démasqué. Fix :
   `takeDelimiter` (« advancing past the delimiter », null = EOF).
2. **Deux writers sur le même fd entrelacent leurs octets** (std.Io async — l'ordre
   d'écriture inter-writers n'est pas garanti même avec flush) → writer stdout UNIQUE
   partagé, passé à `generateOnce`.

Nota : dans un fichier de log capturant stdout+stderr ensemble, des lignes stderr peuvent
fusionner (compteurs de logs < réalité) — compter sur le flux stdout (`réponse`) fait foi.

## Incident bénin (1ʳᵉ relance des gates)

`R0-CRASH` = la **garde VRAM** faisant son travail : un `ollama llama-server` (gemma4:31b,
21,8 GiB) s'était installé sur la carte. `ollama stop` (réversible, geste documenté) et la
séquence est passée. Le garde-fou du PLANNING reste d'actualité en mode résident : la
session REPL occupe la carte pour toute sa durée.
