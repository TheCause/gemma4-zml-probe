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
| D3 | Longueur cible `L_max` | **Paramétrable comptime**, défaut **2048** (cible). **Implémenté à 1024** : le compile XLA-CPU à `.k=2048` pic ~34 Go > 32 Go hôte → abaissement à 1024 (la fenêtre 512 reste franchie ~2×). Remonter quand RAM VM augmentera. |
| D4 | Découpage | **Gate-par-gate maximal** : palier linéaire borné avant le vrai ring. |
| D5 | Prefill en L2 | **Conservé via fixture `cache0`** ; seule la phase decode devient autonome. |

**Invariant oracle** : tant qu'on reste en **greedy**, la séquence HF est déterministe → l'oracle reste **bit-exact (argmax == HF)** aux deux étages. La différence L1→L2 n'est pas l'oracle mais la *source des embeds* (fixture pré-calculée → gather).

## 3. Principe directeur

La génération longue **fait évoluer `engine.zig` lui-même** (le socle publié), **pas une copie** — sinon on retombe dans le travers que le socle modulaire visait à éliminer (copier le moteur par hypothèse).

**Garantie de non-régression** : **E1 et E2 restent verts à chaque commit**. C'est possible parce que les nouveautés sont gardées derrière des **paramètres comptime** dont la valeur par défaut **n'émet aucune nouvelle op** :

- `KMAX_SLIDING` / `KMAX_FULL` : tailles de cache comptime, défaut **8** (= `SEQ_LEN + N_DECODE` actuel) → fixture E1/E2 (KMAX=8) chargée inchangée.
- `ring: bool` comptime, défaut **`false`** : en `false`, le scatter/lecture sliding reste à `.k = pos` (chemin decode4 actuel, **aucune op modulo émise**) ; en `true` (L1b), `.k = pos % KMAX_SLIDING`.

⚠️ **La preuve HLO exige l'élision comptime, pas l'identité numérique.** Émettre `pos % 8` produirait une op modulo absente de decode4 → le `diff -rq` des dumps `--xla_dump_to` divergerait **même si les tokens coïncident**. Donc en config E1/E2 (`ring=false`), le modulo doit être **comptime-supprimé** (`if (ring) … else …`), pas calculé. Idem pour la sélection de masque (§5.3) : en config par défaut, un seul masque est émis, comme aujourd'hui. La preuve HLO (`diff -rq`) reste l'arme de non-régression à chaque commit.

## 4. Découpage en gates

| Gate | Contenu | Oracle / critère |
|------|---------|------------------|
| **L0** | Script Python : fixture `gen_long.safetensors`. **Réécriture ciblée** de `scripts/45_gen_vq_oracle.py` (≠ simple dérivation) — delta concret : (a) générer N ≈ 2048 tokens greedy HF avec le **sliding window 512 réellement actif** + `min_new_tokens` / EOS ignoré pour garantir la longueur ; (b) émettre **deux masques** `masks_sliding` (bande, `.k=KMAX_SLIDING`) et `masks_full` (causal plein, `.k=KMAX_FULL`) au lieu du masque causal unique `.k=8` ; (c) dimensionner les caches sliding/full à `.k = L_max` ; (d) **retirer les hooks V-quant** (compression hors-scope, §8). Tenseurs produits : `embeds`, `embptls`, `cos/sin`, `masks_sliding`, `masks_full`, `positions`, `expected`, `cache0` du prompt. | produit l'oracle (pas de gate) |
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

### 5.2 Indexation du cache
Sous garde `comptime ring` (§3) :
- **Sliding**, `ring = true` (L1b) : le **scatter** (écriture) se fait à `slot_k = pos % KMAX_SLIDING`. `ring = false` (E1/E2, L1a) : `slot_k = pos` (chemin actuel, modulo comptime-élidé).
- **Full** : scatter toujours à `pos` (jamais de ring).

> **Côté lecture** : les readers ne s'indexent **pas** par `.k` (ils lisent tout le tenseur cache via `choose1d(.slot, …)` et c'est le **masque** qui sélectionne les positions valides — `engine.zig:226-272`). Le comportement circulaire de L1b est donc porté par (a) l'index de **scatter** `pos % KMAX_SLIDING` et (b) le **masque circulaire** calculé host (§5.3), pas par un modulo côté lecture.

### 5.3 Masque — **deux entrées par type de couche** (correction post-revue)
Point de design porteur : le masque est appliqué **identiquement à toutes les couches** (`engine.zig:268`, `scores.add(mask.broad(scores.shape()))`), et sa dimension `.k` **doit égaler `cache_k.dim(.k)`** pour broadcaster. Dès que le cache sliding a `.k = 512` et le cache full `.k = L_max`, **un masque unique ne peut pas servir les deux** (le code actuel ne marche que parce que sliding et full partagent KMAX=8).

→ Le masque devient **deux entrées** dans `Packed`, calculées côté host :
- `masks_sliding` (`.k = KMAX_SLIDING`) : bande causale `[max(0, pos-511), pos]` (L1a), ré-indexée pour le wrap circulaire (L1b).
- `masks_full` (`.k = KMAX_FULL`) : **causale pleine** (les couches full ne sont **jamais** fenêtrées — cf. §1/§5.1 ; leur appliquer la bande corromprait silencieusement les couches dont la correction est la plus critique).

`runLayerGen` reçoit les deux masques (issus de `Packed` via `pickStep`) et sélectionne par **`comptime isFull(i)`** : `const mask = if (isFull(i)) mask_full else mask_sliding;`. Ça couvre producers **et** readers (un reader hérite du type de son writer, déjà porté par `isFull(i)`).

**Compatibilité E1/E2** : en config par défaut (`KMAX_SLIDING == KMAX_FULL == 8`), les deux entrées ont la même forme et le même contenu causal → pour préserver le HLO de decode4, le défaut **émet un seul masque** (le dédoublement est lui aussi gardé comptime). Le diff `engine.zig` reste localisé (indexation + sélection de masque par type, toutes deux comptime), mais l'affirmation initiale « masque unique intact » est **abandonnée**.

> **Implémentation — `Packed`/`Cache` paramétrés comptime.** Comme `zml.io.load` réfléchit sur les champs du struct, le dédoublement ne peut pas être un simple `if` dans `forward` : `Packed` (et `Cache`) doivent devenir des **types paramétrés comptime** (à la `EngineModel(Brick)`) — la config par défaut expose `masks` + `cache_*` en KMAX=8 (fixture E1/E2 chargée inchangée), la config longue expose `masks_sliding`/`masks_full` + caches `L_max`. Le plan doit traiter cette paramétrisation comme une tâche à part entière.

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
- **Mémoire** *(corrigé après exécution, cf ENGINE_LOG 5-7 juin)* : le pic **NE VIENT PAS du cache**
  (sliding/full ≈ dizaines de Mo) mais des **conversions f32 des poids** (`c(layer.*)` ×35 couches
  matérialisées dans un seul graphe). À `.k≥512` le forward 35-couches mono-graphe monte à ~33 Go (RAM
  23 Go VM + swap) → swap thrash. **Contourné par le chunking du decode** (mode `chain`, pic ~23,6 Go +
  ~4 Go swap résiduel à traquer) + swapfile temporaire 16 Go. La note initiale « cache 30 MB négligeable »
  est **inexacte** : elle oubliait les poids f32, dominateur réel (cf `GENERATION_LONGUE_CHUNKING_DESIGN.md` §1).
- **Non-régression** : à chaque commit, re-run E1 (preuve HLO `diff -rq`) + E2. Tout commit qui les casse est rejeté. E1/E2 tournent en **config par défaut** (`ring=false`, `KMAX_SLIDING=KMAX_FULL=8`, masque unique) : la fixture KMAX=8 et la brique TurboQuant chargent inchangées, et le graphe émis est byte-identique à decode4/gen_vq.
- **Compute** : exécution sur la 3090 (`ssh ia@192.168.1.163`, workspace `/data/rqz_workspace/zml/examples/rqz/`, checkpoint `/data/gemma4-zml-probe/weights/model.safetensors`), via `deploy_to_3090.sh`. Fixtures volumineuses laissées sur la 3090 (gitignorées). Aucune commande à coller pour Régis.

## 8. Hors-scope (YAGNI)

- Sampling non-greedy (température, top-p) — l'oracle greedy suffit à valider la mécanique.
- Prefill autonome (multi-token) — conservé via fixture (D5).
- Internalisation in-graph — déférée en L3 optionnel (D2).
- Briques de compression sur génération longue — chantier suivant, **une fois** le socle long validé (le banc d'essai sera alors prêt à les recevoir).
