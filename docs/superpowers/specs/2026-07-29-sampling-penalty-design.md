# Spec — repetition penalty + sampling : faire du 12B un moteur d'inférence conforme

> **Date** : 2026-07-29 · **Statut** : rév. 1, à relire · **Machine cible** : 3090 (12B) + M4 (oracle)
>
> **Rapport à la spec du 27 juil** (`2026-07-27-sampling-repetition-penalty-design.md`, rév. 4,
> deux tours de revue) : elle reste la **source de vérité de la phase 1** (repetition penalty).
> Ce document la **prolonge** — il ne la remplace pas et ne la réécrit pas. Il intègre ce que le
> chantier `generation_config` (29 juil, 12 gates) a changé, et il **développe la phase 2**
> (sampling), que la rév. 4 ne faisait qu'esquisser en cinq lignes de gates.
>
> Ce qui reste valide sans modification : C1 (deux phases), C2 (host-side pur), C3 (12B seul),
> C4 (directives `:param` à chaud), C5 (l'oracle appelle le vrai processor), F1→F7, et le design
> de `applyRepetitionPenalty`.

---

## 1. Objectifs

Formulés par Régis, et traduits ici en critères vérifiables.

**O1 — pouvoir valider chaque brique en régime déterministe avant d'activer l'aléatoire.**
Une fonction de sampling ne se valide pas en regardant si « ça a l'air bien ». Chaque brique doit
avoir un **régime neutre** où elle n'a pas le droit d'être aléatoire, et où elle doit rendre
**bit-à-bit** ce que le moteur rendait avant elle.

**O2 — que le 12B devienne un moteur d'inférence conforme.** Aujourd'hui il décode en greedy, un
régime que Google **ne recommande pas** : `generation_config.json` déclare
`do_sample: true, top_k: 64, top_p: 0.95, temperature: 1.0`. À la fin de ce chantier, le runner
applique **8 clés sur 8** (les 2 du chantier précédent + les 4 de sampling + la penalty), et ses
sorties sont **reproductibles à seed fixée**.

**Critère de succès global, énonçable par un tiers** : `--seed 42` deux fois de suite donne le
même texte à l'octet près ; `--repetition-penalty 1.0 --top-k 1` donne **exactement** la sortie
greedy d'avant le chantier ; et l'oracle HF, à politique identique, produit la même séquence.

---

## 2. Faits établis par lecture de la source (transformers 5.14.1, venv `g12b`)

**F8 — l'ordre des processors, mesuré et non supposé.** `_get_logits_processor` les instancie
dans cet ordre (rang réel dans la liste complète de 23) :

```
 4. RepetitionPenaltyLogitsProcessor
14. SuppressTokensLogitsProcessor
16. TemperatureLogitsWarper
18. TopKLogitsWarper
19. TopPLogitsWarper
```

⇒ **Penalty → Suppress → Temperature → TopK → TopP**. Deux conséquences non triviales :

1. **La température s'applique AVANT top-p.** Beaucoup d'implémentations font l'inverse. Comme
   `top_p` raisonne sur les probabilités **après softmax**, diviser par `T` change la masse
   cumulée, donc **le nombre de tokens retenus**. L'ordre n'est pas cosmétique.
2. La température est en revanche **sans effet sur top-k** : la division par `T > 0` est monotone
   croissante, donc elle préserve l'ordre des logits. Écrit ici pour qu'on ne « corrige » pas un
   jour cet ordre en croyant bien faire.

**F9 — `TopKLogitsWarper` garde les ex æquo du k-ième.**

```python
indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
```

L'inégalité est **stricte** : tout token dont le logit **égale** celui du k-ième est **conservé**.
Une implémentation naïve « garder les k premiers indices du tri » diverge donc dès qu'il y a une
égalité au rang k. Sur un vocabulaire de 262 144 logits f32, ce n'est pas une hypothèse d'école.

**F10 — `TopPLogitsWarper` trie en ASCENDANT et retranche par le bas.**

```python
sorted_logits, sorted_indices = torch.sort(scores, descending=False)
cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0
```

Ce **n'est pas** la formulation habituelle (tri descendant, garder tant que le cumul reste sous
`p`). Les deux coïncident au centre et **diffèrent aux bords** — égalités, `min_tokens_to_keep`,
et le fait que le test soit `<=` et non `<`. On reproduit **cette** formulation, pas celle qu'on
croit connaître. `min_tokens_to_keep` protège les **derniers** éléments du tri ascendant,
c'est-à-dire les **plus probables**.

**F11 — `filter_value` vaut `-inf` par défaut**, sur les deux warpers. Cohérent avec le choix déjà
fait au chantier précédent (`-inf` plutôt que `finfo.min`, qui déborderait sous multiplication).

**F12 — le prérequis RP-1 est en grande partie levé.** La spec rév. 4 exigeait de refonder
l'oracle de décode 12B en fp32 (il était bf16 par construction, `69:371` refusant `--compute-fp32`
hors teacher-force). Cette garde a été **levée le 29 juil** (Task 6 du chantier `generation_config`),
et le mode décode fp32 a **tourné** : coût mesuré **69,75 s/token** (médiane tokens 2-4) contre
28,57 s/token en bf16. RP-1 devient une **vérification**, plus un chantier.

**F13 — deux mécanismes de sélection cohabiteront.** `gencfg.select()` (chantier précédent)
choisit dans le **top-5 pré-trié** en *sautant* les ids supprimés — il n'écrit **pas** `-inf`.
L'argument d'exactitude qui le fonde (« rang brut ≤ 3 ») **ne vaut que là**. Dès qu'une brique de
ce chantier s'arme, on travaille sur le **vecteur complet** et la suppression doit y être appliquée
comme HF la fait. Le raisonnement « inverser penalty et suppression est inoffensif car
`-inf × p = -inf` » (spec `generation_config` §4.6) **ne s'applique donc pas** au chemin top-5.

---

## 3. Claims falsifiables — prédictions PRÉ-ENREGISTRÉES

> Exigence Régis (`feedback_falsifiabilite_aussi_en_ingenierie`, 28 juil) : chaque claim porte une
> prédiction chiffrée et **ce qui la tuerait**, committées **avant** la première mesure.

**S1 — « le chemin complet, en régime neutre, est indiscernable du chemin greedy ».**
Prédiction : avec `--repetition-penalty 1.0 --top-k 1` (sampling armé, aléatoire neutralisé), les
ids générés sont **bit-identiques** à ceux du chemin greedy actuel, sur le témoin 48 **et** le
témoin 124. **Ce qui la tue** : un seul id différent. Pas de tolérance — les deux chemins calculent
le même argmax sur les mêmes logits.

**S2 — « la penalty reproduit HF ».** Prédiction : sur le témoin 200 en teacher-forcing fp32 avec
`penalty = 1.15`, `n_match == n_total − 1`, l'unique mismatch autorisé étant **@47** (le tie
bistable déjà instruit). **Ce qui la tue** : un mismatch ailleurs qu'à 47.

**S3 — « les warpers reproduisent HF, y compris aux bords ».** Prédiction : le selftest host-only
rend **100 %** sur des cas incluant explicitement (a) une **égalité exacte au rang k**, (b) un
`top_p` qui tombe **pile** sur une frontière de cumul, (c) `min_tokens_to_keep` mordant. **Ce qui
la tue** : un seul cas de bord divergent — et ce sont précisément les cas que F9/F10 rendent
probables.

**S4 — « le tirage suit la distribution ».** Prédiction : χ² sous α = 0,01 sur 10 000 tirages,
distribution théorique calculée par une implémentation **indépendante** (torch), binning à ~10
catégories (effectif attendu ~1000, σ ≈ 31). **Ce qui la tue** : χ² au-delà du seuil. **Non-vacuité
exigée** : un biais injecté de 5 % doit **FAIL** le test — sinon le test ne discrimine rien.

**S5 — « le graphe ne bouge toujours pas ».** Prédiction : HLO pré-optimisation **byte-identique**
au témoin, comme au chantier précédent. **Ce qui la tue** : un md5 différent — signe qu'un
`Tensor.Rng` ou un warper a fui dans le graphe.

**S6 — « le surcoût du D2H complet est borné ».** Le chemin complet rapatrie 262 144 f32 (~1 Mo)
par step au lieu de ~48 octets. Prédiction : perte de débit **< 15 %** sur un run de 200 tokens
(référence 9,0 tok/s). **Ce qui la tue** : une perte supérieure — auquel cas il faudra arbitrer
(topK in-graph élargi, au prix de S5).

> ⚠ **Aucune de ces prédictions n'est un vœu** : S6 en particulier est un chiffre que je n'ai pas
> mesuré. S'il tombe, c'est le design qui est à revoir, pas le seuil.

---

## 4. Design — les deux chemins, et le pont entre eux

### 4.1 Chemin A (inchangé) — greedy nu

`repetition_penalty == 1.0` **et** sampling désactivé. Le graphe rapatrie le top-5 (~48 octets),
`gencfg.select()` choisit. **Strictement le code d'aujourd'hui**, prouvé par les 12 gates du
chantier précédent. Aucune ligne n'y est touchée.

### 4.2 Chemin B — dès qu'une brique s'arme

Le vecteur complet de logits est rapatrié (il **sort déjà** du graphe, F1 : 7 sorties, logits en
3ᵉ, aujourd'hui non lus), puis traité **dans l'ordre de F8** :

```
logits[V] ──penalty(hist)──▶ ──suppress(-inf)──▶ ──÷T──▶ ──topK──▶ ──topP──▶ ──argmax|tirage──▶ token
```

**Ce que le chemin B doit garantir** : quand toutes les briques sont en régime neutre, il produit
le **même token** que le chemin A. C'est S1, et c'est le cœur de l'objectif O1.

### 4.3 Le gate-pont (RP-PONT) — la pièce maîtresse

Deux chemins qui choisissent un token, c'est deux occasions de diverger en silence. Le pont est
donc un gate **à règle d'arrêt dure**, exécuté **avant** toute mesure de qualité :

- `--repetition-penalty 1.0 --top-k 1` sur les témoins 48 et 124 → ids **bit-identiques**.
- Le compteur `n_full_path` (nombre de steps passés par le chemin B) doit être **non nul** :
  sinon le gate a validé le chemin A contre lui-même. **C'est le détecteur de vacuité intégré**,
  et il est obligatoire — le chantier précédent a produit un contrôle qui ne pouvait pas échouer,
  on ne recommence pas.

### 4.4 Ce qui doit être dit dans le code

- La garde `suppress.len + 1 > TOP_K` (chantier précédent) protège **le chemin A uniquement**.
  Sur le chemin B elle est sans objet. Le commentaire doit nommer lequel des deux elle protège,
  sinon un futur lecteur la croira universelle.
- `gencfg.isSuppressed()` reste la **fonction unique** de suppression (exigence §4.7 de la spec
  précédente) : le chemin B l'appelle pour écrire `-inf`, il n'en réimplémente pas la logique.

---

## 5. Design — `zml_runner/sampling.zig`

**Responsabilité unique** : transformer un vecteur de logits en un token. **Aucune dépendance
ZML** (des `f32` nus) ⇒ testable par `zig test`, sans GPU, sans PJRT, sans poids. C'est ce qui rend
O1 praticable : on itère en secondes, pas en minutes de compile.

```zig
pub const Params = struct {
    repetition_penalty: f32 = 1.0,  // 1.0 = neutre
    temperature: f32 = 1.0,         // 1.0 = neutre (HF n'instancie PAS le warper à 1.0, F8)
    top_k: u32 = 0,                 // 0 = désactivé
    top_p: f32 = 1.0,               // 1.0 = neutre
    min_tokens_to_keep: u32 = 1,    // défaut HF
    seed: ?u64 = null,              // null = argmax (pas de tirage)
};

/// Penalty IN-PLACE, au plus une fois par token distinct (F3). Signe commande l'opération.
pub fn applyRepetitionPenalty(logits: []f32, hist: []const u32, penalty: f32, seen: *Bitset) void;

/// -inf sur les ids supprimés. Délègue à gencfg.isSuppressed — pas de seconde implémentation.
pub fn applySuppression(logits: []f32, policy: *const gencfg.GenCfg) void;

/// ÷ temperature. JAMAIS × (1/T) : l'équivalence mathématique casse la bit-exactitude f32.
pub fn applyTemperature(logits: []f32, t: f32) void;

/// F9 : élimine STRICTEMENT sous le k-ième ⇒ les ex æquo du k-ième SURVIVENT.
pub fn applyTopK(logits: []f32, k: u32) void;

/// F10 : tri ASCENDANT, cumsum des softmax, retrait par le bas (`<= 1 - p`), min_tokens_to_keep
/// protégeant la QUEUE du tri ascendant (= les plus probables).
pub fn applyTopP(logits: []f32, p: f32, min_keep: u32) void;

/// Tirage multinomial sur les logits filtrés. RNG **host** seedé par (seed, step) — jamais le
/// RNG device (F4 : non garanti déterministe entre backends/versions).
pub fn sample(logits: []const f32, rng: *Rng) u32;

/// argmax host, tie-break explicite (premier indice gagnant).
pub fn argmax(logits: []const f32) u32;
```

**Trois interdits, chacun adossé à un gate** :

1. **`× (1/penalty)` et `× (1/T)`** — mathématiquement équivalents, ils cassent la bit-exactitude
   attendue. La division reste une division.
2. **Retranscrire une formule de warper depuis la doc ou la mémoire.** Les fixtures sont produites
   par les **vrais** warpers HF (C5) : retranscrire ferait comparer deux transcriptions du même
   auteur, et une inversion de branche commise des deux côtés passerait le gate.
3. **Allouer par step.** Bitset et buffer de logits alloués **une fois**, dimensionnés au vocab
   runtime. Sinon 200 tokens × 20 prompts = plusieurs Gio brassés, et le critère RSS saute de
   bonne foi.

---

## 6. Design — oracle et fixtures

**Fixtures des warpers** (`scripts/72_sampling_fixture.py`, 72 = premier numéro libre) — patron
**GC1**, qui a rendu 8/8 hier et attrapé un double-free : cas de logits + résultat attendu calculé
par les **vrais** `TemperatureLogitsWarper` / `TopKLogitsWarper` / `TopPLogitsWarper`, plus le
sidecar des cas de bord. **Les cas de bord sont la raison d'être de la fixture** — S3 nomme les
trois qui doivent y figurer.

**Oracle de bout en bout** : `scripts/69_u8_gen_oracle.py`, déjà porteur de la politique de
décodage. On y ajoute `--repetition-penalty` et les paramètres de sampling, **pris au vrai
processor HF**, avec `script_md5` et `gen_policy` déjà au manifest.

---

## 7. Gates

### Phase 1 — déterministe (`RP`)

| Gate | Contenu | PASS (pré-enregistré) |
|---|---|---|
| **RP-0** | Témoins figés + HLO avant toute édition | dumps non vides, md5 consignés |
| **RP-1** | L'instrument est fp32 (F12) | un run décode `--compute-fp32` aboutit ; coût consigné |
| **RP-PONT** | **Le pont** : `penalty 1.0`, `top-k 1` | ids **bit-identiques** aux témoins 48 et 124 **et** `n_full_path > 0` — **FAIL ⇒ STOP** |
| **RP-2** | Selftest `zig test` de `applyRepetitionPenalty` | 100 % des cas, dont dédup et signe |
| **RP-3** | Penalty vs HF, teacher-forcé fp32 | `n_match == n_total − 1`, unique mismatch **@47** (S2) |
| **RP-4** | Non-vacuité de la penalty | (a) sans dédup ⇒ FAIL ; (b) signe inversé ⇒ FAIL ; (c) `penalty=1.0` ⇒ ids inchangés |
| **RP-5** | Graphe et mémoire | HLO **byte-identique** (S5) ; RSS stable sur 200 tokens |
| **RP-6** | **Coût du chemin complet** | débit mesuré vs 9,0 tok/s ; **< 15 %** de perte (S6) |

### Phase 2 — stochastique (`SM`), livrée après la phase 1

| Gate | Contenu | PASS (pré-enregistré) |
|---|---|---|
| **SM-0** | Régimes **dégénérés**, 10 seeds | `top-k 1` avec `T ≠ 0` ⇒ ids identiques aux 10 seeds **et** == greedy |
| **SM-1** | Selftest warpers sur fixtures | 100 %, **cas de bord inclus** (S3) |
| **SM-2** | Distributionnel | χ² α = 0,01, distribution théorique **indépendante** ; biais 5 % ⇒ FAIL (S4) |
| **SM-3** | Reproductibilité | même seed ⇒ sortie identique à l'octet |
| **SM-4** | Non-vacuité du RNG | ≥ k sorties distinctes sur N seeds, k et N **calculés depuis la fixture avant** la mesure |
| **SM-5** | Bout en bout vs HF | config Google complète ; séquences identiques **ou** écarts tous sous le seuil, publiés |

**Règle d'arrêt** : RP-PONT, RP-3, RP-5 et SM-2 FAIL ⇒ **STOP**. Aucune requalification sans
décision écrite de Régis.

---

## 8. Vigilances pré-enregistrées

1. **Un gate de fidélité position-par-position doit être TEACHER-FORCÉ.** La trajectoire libre du
   12B est **bistable** (`FINDING_NONDETERMINISME_TRAJECTOIRE.md`) : un critère posé sur une
   génération libre échouerait à tort ~45 % du temps. Les gates en roue libre sont **statistiques**.
2. **Séparer stdout et stderr** dans toute capture destinée à un grep (piège 24) : `> out 2> err`.
3. **Un détecteur ne doit pas pouvoir matcher la ligne de configuration** qui mentionne la valeur
   cherchée (piège 25) — restreindre à la ligne qui porte la donnée, puis contre-prouver.
4. **La penalty comprime les marges** (÷1,15 sur la branche positive) : elle fabrique donc des
   quasi-ties. C'est pourquoi RP-1 (instrument fp32) précède RP-3, et pourquoi l'ε de quasi-tie
   est **dérivé** du bruit mesuré (2e-3) et non deviné.
5. **`temperature: 1.0` ⇒ HF n'instancie pas le warper** (F8). Ne pas porter une division par 1.0
   « pour la forme » : elle n'est pas neutre en f32 sur toutes les valeurs.

---

## 9. Hors périmètre, et dettes écrites

**Hors périmètre** (décision Régis, 29 juil — périmètre « conformité de génération ») : multi-tour
avec contexte accumulé · stop sequences · streaming token par token · batching du 12B (B > 1) ·
beam search · `no_repeat_ngram_size` · câblage E2B.

**Dettes** : l'**E2B** ne sort pas ses logits du graphe (`gen_auto.zig:753`, 6 sorties) — le câbler
exigerait +1 sortie au tuple racine, donc un arbitrage HLO. Et l'**équivalence de l'arrêt**
runner ↔ HF n'est prouvée par aucun gate (héritée du chantier précédent).

---

## 10. Ordre d'exécution

1. RP-0, RP-1 (témoins et instrument) — **avant toute édition**
2. `sampling.zig` + `zig test` (RP-2) — sans GPU
3. Câblage du chemin B + **RP-PONT** — la mécanique complète, en régime neutre
4. Penalty armée : RP-3, RP-4, RP-5, RP-6
5. Warpers + fixtures : SM-1 (sans GPU), puis SM-0
6. Tirage : SM-2, SM-3, SM-4
7. SM-5, documentation, passe de nuance sur la claim (le gate GC11 doit rester vert), PR
