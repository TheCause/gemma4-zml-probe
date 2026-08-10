# Cadrage des dettes restantes — K3, K4, K5, Triton paged attention

**Date :** 10 août 2026 · **Statut :** fiches de CADRAGE, aucun code écrit.
**Produit par :** Task 8 du plan `2026-08-10-dettes-restantes-penalty.md`, à l'issue du
chantier phase 1 (repetition penalty).

Ces quatre chantiers ont été explicitement écartés de la phase 1 : chacun exige une spec
dédiée avant la moindre ligne. Cette page dit, pour chacun, **ce qu'il faut décider avant de
commencer** — pas comment le faire. L'ordre recommandé est en fin de page.

---

## Fiche 1 — K3 : conformité de la claim pour E2B

**L'état.** La claim de conformité « politique de décodage appliquée » est prouvée pour le 12B
et **fausse pour E2B**, où elle n'est simplement pas applicable : le graphe E2B rend **six**
sorties (`gemma4_gen_auto.zig:753` — `t5.values, t5.indices, slk, slv, flk, flv`) et les
logits complets **ne sortent pas du graphe**. Sans logits host, il n'y a ni `suppress_tokens`,
ni penalty, ni aucun warper : le chemin B n'existe pas pour E2B.

**Ce qu'il faut décider AVANT de coder** — deux issues, et il faut choisir :
1. **Sortir les logits du graphe E2B** (6 sorties → 7). Techniquement symétrique du 12B, mais
   cela **change le graphe** : nouveaux témoins HLO E2B, et tous les gates E2B adossés au md5
   actuel sont à re-passer. Coût du transfert host à chaque step à mesurer, pas à supposer.
2. **Documenter la claim comme non applicable à E2B**, définitivement, et le dire dans le
   README plutôt que de laisser croire à une couverture uniforme.

**Le piège spécifique.** Google **ne publie pas** de `suppress_tokens` pour E2B. En coder un
« par analogie » avec le 12B serait fabriquer une politique — précisément la faute que le
finding `FINDING_GENERATION_CONFIG.md` a coûté cher à corriger. Si l'option 1 est retenue,
la politique E2B doit venir de **son** `generation_config.json` ou rester vide.

**Prérequis :** re-témoins HLO E2B. **Taille :** ~1 session. **Bloque :** rien.

---

## Fiche 2 — K4 : le résident à reprise (`--repl` + dump/restore)

**L'état.** `--repl` et `--load-cache`/`--dump-cache` sont mutuellement exclusifs, par refus
bruyant et assumé (`LoadCacheReplUnsupported`, `DumpCacheReplUnsupported`). La raison est
qu'il n'existe **aucune sémantique multi-tour** : chaque prompt du repl repart d'un cache à
zéros, en position 0.

**Ce qu'il faut décider AVANT de coder.** Le vrai sujet n'est pas « ajouter `:dump <f>` et
`:load <f>` » — c'est **ce que signifie un tour de conversation**. Trois questions, dans cet
ordre :
1. Un `:load` en cours de session **remplace-t-il** le contexte ou s'y **ajoute-t-il** ?
2. Que devient l'historique de la penalty au `:load` ? (Le chantier phase 1 a tranché pour le
   one-shot : `hist` est seedé depuis `ids_fed`, et `--ignore-prompt` + `--load-cache` est
   refusé parce que « le prompt » n'y est plus une notion définie. Le multi-tour rouvre
   exactement cette question, avec en plus la frontière entre tours.)
3. Quelle est la politique d'arrêt entre deux tours ?

**Prérequis dur : K5.** Un second prompt après un restore **EST** un prefill partiel. Ouvrir
K4 avant K5, c'est réinventer K5 dans un coin.

**Taille :** 1-2 sessions. **Spec obligatoire.**

---

## Fiche 3 — K5 : le prefill partiel (reprendre un cache et feeder un prompt NEUF)

**L'état.** `--load-cache` reprend un cache et **continue** la génération. Il ne sait pas
absorber un prompt neuf : la boucle de reprise entre directement en phase de génération.

**Ce qu'il faut dériver et PROUVER, pas supposer.** Tout ce qui dépend de la position :
- les **positions** au-delà de `step_next` ;
- les **masques** sliding (la fenêtre glissante ne commence plus à 0) ;
- le **RoPE** au-delà de `step_next`.

Chacun doit être vérifié **teacher-forcé contre HF**, pas en décodage libre : en libre, une
erreur de position se manifeste par une divergence d'ids qu'on peut confondre avec un quasi
ex æquo — le finding `FINDING_temoin_ids_non_reproductible.md` montre que ce genre de
confusion coûte cher. Teacher-forcé, l'erreur apparaît sur les logits eux-mêmes.

**Pourquoi ce chantier compte.** C'est le chemin naturel vers un vrai serving : sans prefill
partiel, pas de conversation, pas de préfixe partagé, pas de cache réutilisable.

**Taille :** 1-2 sessions. **Spec obligatoire.** **Débloque :** K4.

---

## Fiche 4 — Triton paged attention

**L'état.** C'est **le** gros chantier restant, et le seul chemin crédible vers un flash
attention **B>1**. L'audit `ZML_UPSTREAM_AUDIT_2026-07-12.md` §2 relève, côté upstream depuis
`57d191d0` : B>1 natif (block_table / seq_lens / query_start_len, style vLLM varlen), f32
accepté au niveau du DSL, scale custom (le 1.0 de Gemma est exprimable, `6ad3e267`), sliding
window et `is_causal` supportés, heuristique dédiée pour head_dim ≥ 256.

**Les deux inconnues à lever AVANT de s'engager :**
1. **`hd=512` n'est pas testé upstream** — et l'audit note que le lane full hd=512 fp32 du
   moteur est « très probablement incompatible, bump ou pas ». C'est la question qui décide de
   la faisabilité : à instruire en premier, sur un cas minimal, avant tout portage.
2. **Le cache YOCO doit passer à un layout PAGINÉ.** Ce n'est pas un ajustement : c'est une
   refonte de la structure qui porte tout l'état.

**Le coût caché, à énoncer d'emblée :** le bump ZML **invalide les témoins HLO**. Le md5
`297679847aa04b719942d75d093adf2b`, stable sur six chantiers, ne le sera plus — et avec lui
toute la batterie de gates qui s'y adosse. Il faut donc budgéter **des re-gates complets**,
pas seulement le portage.

**Taille :** 3+ sessions. **Indépendant** de K3/K4/K5.

---

## Ordre recommandé

```
K5 (prefill partiel) ──→ K4 (résident à reprise)
K3 (E2B)          indépendant
Triton            indépendant, le plus lourd, à ouvrir seul
```

**K5 avant K4** : dépendance dure, énoncée en fiche 2. **K3** est le moins cher et solde une
claim aujourd'hui inexacte — bon candidat si l'objectif est la justesse de la documentation
plutôt que la performance. **Triton** ne devrait pas être ouvert en parallèle d'un autre
chantier : il déplace le socle sur lequel les autres se vérifient.
