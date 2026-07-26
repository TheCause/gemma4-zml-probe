# Spec — Mode résident (`--repl`) : compile une fois, prompts en boucle

> **Date** : 2026-07-26 · **Niveau** : léger (spec+plan combinés, pattern cache-donation) ·
> **Base** : PR #16 mergée (`a4529f4`). Demande Régis : « fais le mode résident » — payer
> load (~1 min) + compile (~38 s) UNE fois, puis répondre à des prompts successifs.

## 1. Design

Nouveau flag `--repl` sur `gemma4_g12auto` (hérité par g12a4k/g12a8k — corps `G12Auto`).

**Boucle** : après load + compile + `eng_buf`/`pk_buf` (inchangés), au lieu d'une génération
unique :

```
loop:
  afficher "prompt> " ; lire UNE LIGNE sur stdin (EOF ou ligne vide → sortie propre)
  renderChatTemplate(ligne) → encoder.reset() + encodeAlloc → ids (BOS préfixé, comme aujourd'hui)
  garde : ids.len + max_tokens ≤ L_MAX sinon message d'erreur et prompt suivant (pas de crash)
  cache_buf = fromBytes(zéros) depuis host.cache_* (toujours vivants — RÉINIT par prompt)
  boucle steps existante (prefill-par-decode + génération, early-stop EOT/max-tokens/L_MAX)
  détok + print réponse + PERF ; libérer generated/gen_top5/cache_buf de l'itération
```

**Sémantique V1 : prompts INDÉPENDANTS** — chaque prompt repart d'un cache zéro et de la
position 0. Pas de multi-tour (conversation avec contexte accumulé) : hors scope, chantier
futur si demandé (exigerait la gestion des positions de reprise et la validation du template
multi-tour vs HF).

**Refactor** : extraire la génération unique (aujourd'hui inline dans `main`, ~:1231-1430 :
boucle steps + détok + PERF) en `fn generateOnce(ctx: *ReplCtx, prompt_text: []const u8) !void`
où `ReplCtx` porte {allocator, io, platform, sharding, exe, eng_buf, pk_buf, tok_sym,
cache_sym, host, tokenizer/encoder, eot_id, max_tokens}. Le mode one-shot appelle
`generateOnce` UNE fois — même chemin de code (pas deux implémentations).

**Gardes de compatibilité** : `--repl` est EXCLUSIF des modes fixtures/probe
(`--oracle`, `--window-vacuity`, `--out-ids`, `--ids-only`, `--selftest-*`) → erreur au
parsing. `--prompt` devient optionnel avec `--repl` (s'il est fourni : premier prompt de la
boucle). `--dump-top5` reste permis (par prompt).

**Point d'API à trancher au build** (seul risque technique) : lecture stdin ligne par ligne
en Zig 0.16 — voie 1 : `std.Io.File.stdin().reader(io, …)` (symétrique du
`std.Io.File.stdout().writer` déjà utilisé :1427) ; repli : `std.Io.Dir.cwd().openFile(io,
"/dev/stdin", …)` + lecture bufferisée (pattern openFile déjà partout dans le runner).

## 2. Non-objectifs

- Multi-tour / contexte accumulé (V1 = indépendant).
- Sampling (greedy inchangé), serveur réseau, batching B>1.
- `engine.zig` : 0 octet (chantier runner-only).

## 3. Gates pré-enregistrés

| Gate | Contenu | Critère PASS |
|---|---|---|
| **R0** moteur intact + non-régression one-shot | `git diff` engine.zig vide ; run one-shot (sans `--repl`), protocole témoin M1 | diff **vide** ; **48/48 ids == `logs/mi_witness_1280_ids.safetensors`** (le refactor generateOnce ne change pas la génération) |
| **R1** équivalence résident | session `--repl` avec 3 prompts : [fenêtre glissante, capitale Australie, fenêtre glissante] | prompt 1 et 3 : **ids identiques entre eux ET == témoin 48** (la résidence n'a pas d'état caché entre prompts) ; prompt 2 : early-stop EOT, réponse contient « Canberra » |
| **R2** stabilité mémoire | pendant R1 : VRAM (`nvidia-smi` par prompt) et RSS | VRAM stable entre prompts (± bruit, pas de croissance monotone) ; pas de fuite host visible sur 3 prompts |

Règle d'arrêt : R0-R2 FAIL → STOP, diagnostiquer (jamais requalifier sans décision Régis).

## 4. Risques

| # | Risque | Réponse |
|---|---|---|
| A | API stdin 0.16 introuvable telle qu'imaginée | Repli `/dev/stdin` pré-enregistré (§1) — Linux only, la cible EST la 3090 |
| B | État caché entre prompts (encoder iree, buffers, step) | `encoder.reset()` déjà systématique (:872) ; cache/step/fed réinitialisés par construction dans generateOnce ; R1 le VÉRIFIE (prompt 1 == prompt 3) |
| C | Fuite mémoire par itération (buffers non libérés dans le refactor) | R2 le mesure ; la boucle steps libère déjà tout par step (motif existant) |

## 5. Livrables

`gemma4_g12auto.zig` (flag, ReplCtx/generateOnce, boucle REPL, gardes), usage/CLI docs de
tête, `docs/REPL_RESULTS.md` (gates R0-R2), PLANNING, grep §5.4, PR.
