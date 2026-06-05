# Génération longue — Design

> Statut : **design validé** (brainstorming 5 juin 2026). Implémentation non commencée.
> Branche : `generation-longue` (depuis `turboquant-zml-vonly`).
> Spec sœur du socle modulaire : voir `ZML_MODULAR_ENGINE_DESIGN.md`.

## 1. Contexte et problème

Le decode ZML actuel (`engine.zig`, gates E1/E2) est un **replay validé**, pas une génération autonome :

- La fixture (`p5_7_8_gen` / `decode_vq_gen`) contient déjà les **embeds pré-calculés** des 4 tokens greedy HF, ainsi que `embptls`, `cos/sin`, `masks` et `positions` par step. Le host (`gemma4_engine_e1.zig`) fait l'argmax du forward et le **compare** à une séquence `expected` connue.
- Le cache est dimensionné au plus juste : `KMAX = SEQ_LEN + N_DECODE = 8`. Le scatter écrit à `.k = pos` (jamais ≥ 8 → **aucun wrap**).
- Les masques sont **pré-calculés par step** côté host (`masks: {step,b,h,q,k}`), jamais une logique de fenêtre glissante.
- Le `sliding_window = 512` réel de Gemma4 n'est donc **jamais atteint** (8 positions seulement).

**Objectif** (décision Régis, 5 juin 2026) : faire du banc d'essai ZML un moteur de **génération longue** (>> 4 tokens, fenêtre glissante 512, cache borné), afin notamment de stresser les **briques** de compression KV (TurboQuant et suivantes) sur de vraies longues séquences.

La table `embed_tokens_per_layer.weight` (`[vocab, 8960]`, 8960 = 35×256) existe dans le checkpoint → une **vraie génération autonome est faisable** (on peut *gather* le token produit en ZML/host).

## 2. Décisions de cadrage (brainstorming)

| # | Décision | Choix |
|---|----------|-------|
| D1 | Objectif | **Les deux, en séquence** : preuve de fidélité longue distance (oracle HF) PUIS inférence autonome. |
| D2 | Architecture boucle autonome | **Hybride** : host-orchestrée d'abord, internalisation in-graph en gate ultérieur optionnel. |
| D3 | Longueur cible `L_max` | **Paramétrable comptime**, défaut **2048**. |
| D4 | Découpage | **Gate-par-gate maximal** : palier linéaire borné avant le vrai ring. |
| D5 | Prefill en L2 | **Conservé via fixture `cache0`** ; seule la phase decode devient autonome. |

**Invariant oracle** : tant qu'on reste en **greedy**, la séquence HF est déterministe → l'oracle reste **bit-exact (argmax == HF)** aux deux étages. La différence L1→L2 n'est pas l'oracle mais la *source des embeds* (fixture pré-calculée → gather).

## 3. Principe directeur

La génération longue **fait évoluer `engine.zig` lui-même** (le socle publié), **pas une copie** — sinon on retombe dans le travers que le socle modulaire visait à éliminer (copier le moteur par hypothèse).

**Garantie de non-régression** : **E1 et E2 restent verts à chaque commit**. C'est possible parce que les changements sont rétro-compatibles : `KMAX_SLIDING` / `KMAX_FULL` deviennent des paramètres **comptime** dont la valeur par défaut reproduit le comportement actuel, et le modulo `pos % KMAX_SLIDING` est l'identité tant que `pos < KMAX_SLIDING` (cas E1/E2 : positions 0..7, KMAX=8). La technique de preuve HLO (`diff -rq` des dumps `--xla_dump_to`) reste l'arme de non-régression.

## 4. Découpage en gates

| Gate | Contenu | Oracle / critère |
|------|---------|------------------|
| **L0** | Script Python : génère N ≈ 2048 tokens greedy HF (sliding window actif côté HF) + fixture `gen_long.safetensors` (`embeds`, `embptls`, `cos/sin`, **masques bande**, `positions`, `expected`, `cache0` du prompt). Dérivé de `scripts/45_gen_vq_oracle.py`. | produit l'oracle (pas de gate) |
| **L1a** | Cache sliding **linéaire borné** (`.k = L_max`), scatter à `pos`, masque **bande causale** `[pos-511, pos]`. Replay (embeds depuis fixture). | argmax == HF sur les N tokens |
| **L1b** | Conversion en **vrai ring-buffer 512** : scatter `pos % 512`, masque **circulaire** (host). Replay. | argmax == HF sur les N tokens |
| **L2** | Inférence **autonome host-orchestrée** : host fait argmax → gather `embed_tokens[tok]` + `embed_tokens_per_layer[tok]` → calcule `cos/sin` + masque pour `pos+1` → réinjecte. **Prefill du prompt conservé via `cache0`** (D5), seule la phase decode est autonome. | séquence générée == HF greedy |
| **L3** *(optionnel)* | Internalise gather + RoPE + masque + argmax dans le graphe ZML (le forward devient `token_in → token_out`, le host ne thread qu'un scalaire). | séquence == HF greedy |

Chaque gate est validé **par perturbation** (corrompre l'oracle → le gate doit FAIL) pour réfuter l'aliasing/la vacuité, comme les gates précédents du projet.

## 5. Architecture et flux de données

### 5.1 Cache
`Cache` gagne deux tailles **comptime** :
- `KMAX_SLIDING` — `= L_max` en L1a, **= 512** en L1b (et au-delà).
- `KMAX_FULL` — **= `L_max`** (les couches *full* / global attention ne sont **jamais** fenêtrées ; les tronquer divergerait de HF).

Rappel topologie (socle actuel) : couches 0..14 = producteurs de KV ; couches ≥ 15 = lecteurs (partagent le KV des writers 13 sliding / 14 full, archi YOCO/shared-KV). Le ring ne concerne que les **slots sliding** ; les **slots full** restent linéaires `L_max`.

### 5.2 Indexation (le seul vrai changement de logique dans `runLayerGen`)
- **Sliding** : `slot_k = pos % KMAX_SLIDING` pour le scatter **et** la lecture (vs `pos` aujourd'hui).
- **Full** : inchangé, `pos`.

### 5.3 Masque
Le masque **reste une entrée** (`Packed.masks`), calculé **côté host** :
- L1a : bande causale `[max(0, pos-511), pos]`.
- L1b : même fenêtre sémantique, mais ré-indexée pour le wrap circulaire du ring.

→ `engine.zig` ne touche **que l'indexation du cache**, pas la sémantique du masque. C'est ce qui rend le diff minimal.

### 5.4 Positions
Déjà threadées (`positions[step]`), inchangées sur le principe ; étendues à N ≈ 2048.

### 5.5 Boucle L2 (host-orchestrée)
À chaque step de decode, le host :
1. `argmax(logits)` → `tok`.
2. gather `embed_tokens[tok] * √1536` → `embeds` ; gather `embed_tokens_per_layer[tok]` → `embptls`.
3. calcule `cos/sin` (RoPE full, position `pos+1`) et le masque fenêtre.
4. réinjecte ces tensors comme entrée du forward (graphe = forward 1-step, quasi inchangé).

Le prefill du prompt reste fourni par `cache0` (fixture), comme aujourd'hui.

## 6. Fichiers

- **Évolue** : `engine.zig` (cache paramétrable comptime + modulo sliding).
- **Inchangés, servent de régression** : `gemma4_engine_e1.zig`, `gemma4_engine_e2.zig` (doivent rester PASS).
- **Nouveaux** :
  - `scripts/46_gen_long_oracle.py` (L0).
  - `gemma4_gen_long.zig` (runner L1a/L1b, replay long).
  - `gemma4_gen_auto.zig` (runner L2, decode autonome host-orchestré).
- **Intacts (oracles historiques)** : `gemma4_decode4.zig`, `gemma4_gen_vq.zig`, `brick_turboquant.zig`.

## 7. Validation et risques

- **Oracle L0 / EOS** : greedy peut atteindre EOS avant N tokens → forcer `min_new_tokens` ou ignorer EOS pour produire une séquence de longueur fixe testable.
- **Ring circulaire (L1b)** : neutralisé par test de perturbation + contrôle de non-vacuité (corrompre l'oracle doit faire FAIL le scan global).
- **Mémoire** : `L_max = 2048` → cache full ≈ 30 MB ; négligeable sur la 3090 (24 GB).
- **Non-régression** : à chaque commit, re-run E1 (preuve HLO `diff -rq`) + E2. Tout commit qui les casse est rejeté.
- **Compute** : exécution sur la 3090 (`ssh ia@192.168.1.163`, workspace `/data/rqz_workspace/zml/examples/rqz/`, checkpoint `/data/gemma4-zml-probe/weights/model.safetensors`), via `deploy_to_3090.sh`. Fixtures volumineuses laissées sur la 3090 (gitignorées). Aucune commande à coller pour Régis.

## 8. Hors-scope (YAGNI)

- Sampling non-greedy (température, top-p) — l'oracle greedy suffit à valider la mécanique.
- Prefill autonome (multi-token) — conservé via fixture (D5).
- Internalisation in-graph — déférée en L3 optionnel (D2).
- Briques de compression sur génération longue — chantier suivant, **une fois** le socle long validé (le banc d'essai sera alors prêt à les recevoir).
