# P5.7.5 — Contrat de précision (oracle hybride fp32 / `embed_tokens_per_layer` bf16)

> **Gate `P5.7.5-prep`** — 2 juin 2026. **Périmètre : documentation + contrat d'oracle uniquement.
> AUCUN runner 35 couches dans ce gate.** Verrouille la stratégie de précision *avant* de composer
> le moteur prefill. Décision Régis : **option 1 — oracle hybride** (fp32 partout sauf
> `embed_tokens_per_layer` en bf16). Options 2 (moteur ZML bf16) et 3 (tolérance structurelle 1e-1
> comme critère premier) rejetées — cf §8.
>
> **Interdit tant que ce contrat n'est pas committé** : démarrer le moteur prefill 35 couches.
> Référence : `docs/ROADMAP_to_full_forward.md` (P5.7.5), `docs/P5_6_closeout.md` (base saine, 0 gap).

---

## 1. Décision (verbatim normalisé)

| Champ | Valeur |
|---|---|
| Choix | **option_1_hybrid_oracle** |
| Dtype principal oracle | **fp32** |
| Exception | `embed_tokens_per_layer` → **bf16** |
| Motif exception | Le `Gemma4TextModel` full fp32 ne tient pas en mémoire ; ce tensor seul = 9,40 Go fp32. Son chemin est **déjà validé indépendamment** en P4.4 (gates PLE) avec des tolérances *BF16-aware*. |
| Moteur ZML | **fp32** (inchangé — invariant projet : PJRT-CPU, déterminisme matmul) |
| Seuil PASS | `max_abs ≤ 1e-2` **ET** `mean_abs ≤ 1e-4` |
| Seuil WARN | `1e-2 < max_abs ≤ 1e-1` → investiguer (localisation, points fixes, distribution, câblage) **avant** d'accepter |
| Seuil FAIL | `max_abs > 1e-1` **OU** NaN/Inf **OU** mismatch de shape ou de distribution |

---

## 2. Pourquoi l'oracle full fp32 ne tient pas en mémoire

Cible compute = VM 3090 `user@gpu-host`, **23 Go** utilisables (VM 24 Go). Chiffres mesurés sur
`fixtures/p5_7_0_loader_manifest.json` (manifest réel du checkpoint, P5.7.0, vérifié 600/600 clés).

### 2.1 Params texte **résidents** (ce que HF instancie : `load_for_runtime=true`)

| Bloc | Params | fp32 | bf16 |
|---|---:|---:|---:|
| **`embed_tokens_per_layer`** `[262144, 8960]` | **2,349 B** (50,7 %) | **9,40 Go** | 4,70 Go |
| `embed_tokens` `[262144, 1536]` (main, sert aussi de lm_head *tied*) | 0,403 B | 1,61 Go | 0,81 Go |
| PLE frontend (`per_layer_model_projection` + norm + `final_norm`) | 0,014 B | 0,06 Go | 0,03 Go |
| 35 couches (attn + MLP + normes + PLE per-layer) | 1,863 B | 7,45 Go | 3,73 Go |
| **Total résident** | **4,629 B** | **18,51 Go** | 9,26 Go |

> **Résident ≠ disque.** Le checkpoint disque porte **4,647 B** (601 clés langage). L'écart de **0,019 B**
> = les K/V des 20 readers (couches 15-34), **présents sur disque mais non instanciés au runtime** (YOCO :
> les readers lisent le KV des producers 13/14, cf P5.0/P5.1) ; `lm_head` est *tied* (absent du fichier). Le
> **résident** (4,629 B) est ce que le modèle occupe en mémoire ; le **disque** (4,647 B) ne pèse que sur le
> *state_dict transitoire* du chargement (§2.2). C'est cette distinction qui explique pourquoi un comptage
> « config » naïf (qui sommerait les K/V des 35 couches) sur-estime de ~0,019 B.

### 2.2 Empreinte des trois oracles possibles

| Oracle | Params résidents | Pic de chargement¹ | Tient en 23 Go ? |
|---|---:|---:|---|
| **full fp32** | 18,51 Go | **≈ 27,8 Go** | ❌ non (déborde au `load_state_dict`) |
| **hybride** (embptl bf16, reste fp32) | **13,82 Go** | ≈ 23,1 Go² | ✅ oui (avec chargement *streaming/assign*) |
| full bf16 (oracle actuel, script 38) | 9,26 Go | ≈ 18,5 Go | ✅ oui — **mais déprécié, cf §6** |

> ¹ **Pic** = modèle matérialisé **+** `state_dict` transitoire lu depuis le `.safetensors` (bf16, **9,29 Go**
> = les 4,647 B de clés disque, K/V readers inclus, lus puis droppés au `load_state_dict`). Co-résidents
> pendant le `load_state_dict`. C'est ce pic, pas la taille finale, qui fait déborder le full fp32
> (18,51 + 9,29 ≈ 27,8 Go) alors même que le modèle final (18,51 Go) tiendrait sous 23 Go.
> ² Le pic hybride (≈ 23,1 Go) est **au ras du plafond**. L'oracle hybride doit charger en *streaming*
> (assigner tensor par tensor en libérant chaque source) ou via `low_cpu_mem_usage` / init `meta`, pour
> ne jamais co-résider le `state_dict` complet. Détail d'implémentation = phase moteur P5.7.5 (§6).

**Conclusion** : garder le seul `embed_tokens_per_layer` en bf16 économise 4,70 Go (la moitié du « gras »)
et fait passer l'oracle sous le plafond, **sans toucher au dtype du chemin de calcul validé** (les 35 couches).

---

## 3. Pourquoi `embed_tokens_per_layer` bf16 est acceptable (en fait : quasi-gratuit)

Trois arguments cumulés, du plus fort au plus contextuel :

1. **Le tensor est déjà bf16 sur disque.** Le checkpoint `model.safetensors` stocke *tous* les poids en
   bf16 (manifest §`dtype: bfloat16`). Un oracle « full fp32 » ne fait qu'**upcaster** bf16→fp32 (élargissement
   *exact*, IEEE-754) : il ne récupère aucune précision absente. Garder `embed_tokens_per_layer` en bf16
   est donc **sans perte vis-à-vis du checkpoint** — la seule différence entre « fp32 » et « bf16 » pour ce
   tensor est l'empreinte mémoire, pas les valeurs.

2. **L'effet bf16 est confiné — et figé en fp32 dès le frontend PLE, avant la boucle de couches.** La seule
   conséquence de garder ce tensor en bf16 touche la composante *token_identity* du **frontend PLE** (P4.4) :
   `token_identity = embed_tokens_per_layer(ids) × √256` (bf16). Cette composante est **fusionnée avec la
   branche contexte fp32 AVANT la boucle** : `per_layer_input = (token_identity + context_normalized) / √2`.
   L'addition `bf16 + fp32 → fp32` **promeut le résultat en fp32**. Donc `per_layer_input` est **déjà fp32**
   quand il entre dans chaque couche, et le bloc PLE per-layer
   (`per_layer_projection(gelu(per_layer_input_gate(h)) × per_layer_input)`) est un calcul `fp32 × fp32`.
   Le seul arrondi bf16 porte sur `token_identity` *avant* la fusion — et il est **sans effet** (point 2bis).

   **2bis. `token_identity` est bit-identique en hybride et en full fp32.** Le lookup est un *gather*
   (sélection exacte) ; le scaling `× √256 = × 16` est une **puissance de 2 exacte** en bf16 comme en fp32
   (seul l'exposant change, la mantisse non) ; et les valeurs sont bf16 sur disque des deux côtés. Donc
   `bf16(w) × 16` (hybride) `== float32(w) × 16` (full fp32), **bit pour bit**. La fusion fp32 reçoit des
   entrées identiques → `per_layer_input` identique → `last_hidden` **bit-identique**. L'exception bf16 n'est
   donc pas seulement « acceptable » : elle est **exactement gratuite** (un full fp32 ne produirait pas un
   autre résultat). *(NB : le scaling de l'embedding principal `× √1536` n'est PAS une puissance de 2, mais
   `embed_tokens` reste fp32 dans l'hybride — aucun enjeu.)*

3. **Ce chemin précis est déjà validé — sur deux gates distinctes.** La caractérisation *BF16-aware* vient
   du gate **P3** (reproduction PyTorch brute du PLE : référence sauvegardée en bf16, seuil 1e-3 jugé trop
   strict, résidu réel ≈ 1 ULP bf16 sur les outliers — `context_normalized` 1.32e-1, `ple_final` 1.45e-1, cf
   `project_gemma4_zml_probe.md` §P3). Le gate **P4.4.2 (A→J)** a ensuite **revalidé la même chaîne en ZML**
   contre un fixture **fp32** (`ple_reference_final.npy`) à tolérance 1e-4 (bit-exact à 1.5e-5). Ensemble,
   les deux couvrent `embed_tokens_per_layer` en bf16 ET en fp32 : on ne re-valide rien de neuf ici.

**Synthèse §3** : l'oracle hybride est **bit-identique à un oracle full fp32** (point 2bis : `token_identity`
exact, fusion fp32, reste du flux fp32). Le coût payé est **uniquement mémoire**, zéro rigueur. La précision
fp32 est intégralement préservée sur le chemin nouvellement validé (les 35 couches).

---

## 4. Inventaire fp32 — ce qui reste en pleine précision

Tout le **chemin de calcul validé** reste fp32 dans l'oracle ET dans le moteur ZML.

| Étage | Composant | dtype oracle | dtype ZML | Validé (tag) |
|---|---|---|---|---|
| Embedding main | `embed_tokens[ids] × √1536` | **fp32** | fp32 | p5.4-embed (bit-exact) |
| PLE frontend (contexte) | `per_layer_model_projection`, `per_layer_projection_norm`, fusion `/√2` | **fp32** | fp32 | P4.4.2 A→J |
| PLE frontend (identity) | `embed_tokens_per_layer × √256` | **bf16** (exception) | bf16→gather→fp32 | P3 (bf16-aware) + P4.4.2 C/D (×16 bit-exact) |
| Attention sliding | q/k/v proj + norm + RoPE `zml.nn.rope` + QK + masque sliding + softmax + context + o_proj | **fp32** | fp32 | p5.2-c/d/e/f |
| Attention full (4,9,14,19,24,29,34) | head_dim **512**, RoPE manuelle partielle (θ1e6, partial 0.25), o_proj | **fp32** | fp32 | p5.6 / p5.6k / p5.7.4 |
| MLP | gate/up + `gelu_pytorch_tanh` + down (**6144** prod. / **12288** readers double-wide) | **fp32** | fp32 | p5.2-h |
| Bloc PLE per-layer | gate→gelu→`× per_layer_input`→proj→norm→+res, puis `× layer_scalar` | **fp32**¹ | fp32 | p5.3-layer |
| Normes | toutes RMSNorm (input/post-attn/pre-ff/post-ff/q/k/v/final), ε=1e-6 | **fp32** | fp32 | p5.2 / p5.5 |
| KV-sharing YOCO | routing producers 0-14 / readers 15-34 (sliding→13, full→14) | (entier) | (entier) | p5.1 / p5.2-a/b |
| Tête | final norm → lm_head (tied) → softcap `30·tanh(x/30)` | **fp32** | fp32 | p5.5-head |

> ¹ `per_layer_input` entrant est **fp32** (promu à la fusion frontend `(token_identity+context)/√2`, §3
> point 2) ; `token_identity` a subi un arrondi bf16 en amont, mais `×√256=16` exact → **bit-identique** au full fp32 (§3 2bis).

**Aucun autre tensor que `embed_tokens_per_layer` n'est dégradé.** Le softmax est fp32 (invariant Gemma4).

---

## 5. Modèle de divergence attendu (calibre les seuils)

Oracle hybride et moteur ZML utilisent **les mêmes valeurs de poids** (bf16 disque, upcastées de façon
identique) et calculent **les 35 couches en fp32**. La **seule source légitime** de divergence est donc la
différence de bibliothèque matmul :

> **PJRT-CPU (Eigen-like) vs PyTorch BLAS** — résidu ≈ **1e-5 par matmul**, **linéaire en longueur de
> réduction**, **s'accumulant avec la profondeur**, **concentré sur les activations de grande magnitude**.

Caractérisation établie par le projet (cf `docs/P5_6_closeout.md` §3, audit `ALL_JUSTIFIED`) :
- `.h=1536` → 1.14e-5 ; `.m=2048` → 2.29e-5 ; `.f=12288` → 5.34e-5 (linéaire en réduction).
- RMSNorm atténue (÷ ~RMS) ; RoPE n'ajoute aucun drift (rotation orthogonale).
- Couche décodeur **composée** (P5.3) : 6.72e-5 ; couche full (P5.7.4) : 1.63e-5.

**Extrapolation 35 couches** : ~9-11 matmuls/couche (4 projections attn q/k/v/o + 3 MLP gate/up/down + 2 PLE
per-layer + 2 matmuls d'attention QK/context) × 35 couches, avec amplification par les connexions
résiduelles (le flux résiduel accumule) et mise à l'échelle par la magnitude croissante des activations.
Plage de drift honnête attendue : **max_abs ~1e-3 à ~1e-2**, `mean_abs` deux à trois ordres en dessous.
→ C'est exactement ce que borne le **PASS `max_abs ≤ 1e-2` / `mean_abs ≤ 1e-4`**. Le `mean_abs` serré garantit
que le drift reste *concentré* (signature fp), pas *diffus* (signature bug).

> ⚠️ La tolérance de couche unique du projet est 1e-4 (5e-4 pour P5.3 composée). Le PASS 35 couches est
> **délibérément plus lâche (1e-2)** : c'est de l'accumulation fp32 légitime, **pas** un relâchement de
> rigueur. La rigueur tient parce que la *signature* du drift est vérifiée (§6), pas seulement sa magnitude.

---

## 6. Distinguer un drift numérique d'un bug de câblage (cœur du gate)

Puisque oracle et ZML partagent valeurs et dtype de calcul, **toute divergence qui ne ressemble pas à la
signature §5 (Eigen-vs-BLAS) est un bug de câblage**, pas du bruit. Tableau des signatures :

| Symptôme | Drift numérique (acceptable) | Bug de câblage (à corriger) |
|---|---|---|
| Magnitude `max_abs` | dans [1e-5, 1e-2], **croît lissement** avec la profondeur | ≥ 1e-1, **ou saut brusque** à une couche |
| Ratio `max_abs / mean_abs` | **élevé** (≥ 100×) : erreur concentrée sur peu d'activations | bas & diffus (tout le tenseur faux) **ou** catastrophique localisé |
| Distribution (mean/std/min/max ZML vs oracle) | **se recouvre** étroitement | dérive (échelle/offset/clipping faux) |
| Croissance par couche | **monotone, graduelle** | plateau puis **marche** à la couche k = site du bug |
| Points fixes `last_hidden[0,q,:8]` | concordent à ~1e-3 | coords précises très fausses |
| NaN/Inf | absent | possible (masque faux, /0, overflow) |
| Pattern d'erreur | non structuré (bruit fp) | **structuré** (transpose, off-by-one d'indice de couche, sliding/full inversés, mauvais target KV, permutation de têtes) |

### Procédure de diagnostic (dans l'ordre)

1. **Garde shape & finitude.** Shape fausse ou NaN/Inf ⇒ **FAIL**, bug de câblage. Stop.
2. **Distribution globale.** Comparer (mean, std, min, max) du `last_hidden` ZML complet vs oracle. Un
   décalage d'échelle/offset ⇒ câblage (ex. `layer_scalar≈0.0884` oublié, final norm manquante, mauvais
   `embed_scale √1536`, softcap appliqué/absent). **Le drift préserve la distribution.**
3. **`max_abs` vs `mean_abs`.** Drift ⇒ `mean_abs ≪ max_abs` (concentré). Un `mean_abs ≳ 1e-3` *diffus* sur
   tout le tenseur ⇒ bug systématique (même si `max_abs` semble passer).
4. **Localisation par couche** *(test décisif)*. Instrumenter l'oracle pour exporter le hidden state après
   chacune des 35 couches ; relancer le moteur ZML avec les mêmes prises par couche. La couche où
   `|ZML − oracle|` **saute** de la bande de drift (~1e-4..1e-3) à ≥ 1e-1 est le **site du bug**. Le drift
   croît de façon monotone et lisse ; un bug apparaît comme une **marche**.
5. **Liste de suspects Gemma4 (une fois la couche k localisée)** :
   - **KV-sharing** : reader k lisant le mauvais producteur (policy P5.1 : sliding→13, full→14) ;
     off-by-one sur `first_kv_shared_layer_idx = 15`.
   - **Dispatch sliding/full** : couche full attendue en {4,9,14,19,24,29,34} ; `head_dim` 256 vs **512** ;
     RoPE `zml.nn.rope` sliding vs **manuelle partielle** full (θ1e6, partial_rotary 0.25).
   - **Largeur MLP** : double-wide **12288** (readers 15-34) vs 6144 (producers 0-14).
   - **Bloc PLE per-layer** manquant/mal placé, ou `layer_scalar` non appliqué.
   - **Masque** : fenêtre sliding 512 ; à S=4 elle dégénère en causal (garde `qlen ≥ window`), mais à S plus
     grand la fenêtre doit mordre — tester un S ≥ 512 si doute.
   - **Résiduels** : sandwich norm dropé (input_ln→attn→post_attn_ln→+res→pre_ff_ln→mlp→post_ff_ln→+res).
6. **Perturbation / non-vacuité.** Corrompre un `input_id` ou un poids : oracle ET ZML doivent changer. Un
   PASS qui **survit** à la corruption est vacant (cf leçon v_norm D.0 : oracle et ZML partageaient une
   hypothèse fausse → accord trompeur à ~5e-6).
7. **Indépendance de l'oracle.** L'oracle dérive du module réel `Gemma4TextModel` (`modeling_gemma4.py`),
   **jamais** des hypothèses du code ZML. C'est garanti ici : l'oracle hybride *est* le module HF réel
   (cf `feedback_oracle_independence`).

### Traitement de la bande WARN (`1e-2 < max_abs ≤ 1e-1`)

Conformément à la décision, **ne pas accepter sur le nombre seul**. Exécuter les étapes 2-4 :
- Si l'erreur est **monotone, concentrée** (`mean_abs` petit), **préserve la distribution**, et **ne se
  localise sur aucune couche unique** ⇒ drift de haut de plage → **accepter avec justification documentée**.
- Si l'erreur **marche** à une couche ou **casse la distribution** ⇒ **bug de câblage** → corriger, ne pas
  accepter.

---

## 7. Sémantique de comparaison (résumé exécutable)

```
PASS  ⟺  shape == (1, S, 1536)  ET  finite  ET  max_abs ≤ 1e-2  ET  mean_abs ≤ 1e-4
WARN  ⟺  shape/finite OK  ET  ( [ 1e-2 < max_abs ≤ 1e-1 ]  OU  [ max_abs ≤ 1e-2  ET  mean_abs > 1e-4 ] )  → §6 avant verdict
FAIL  ⟺  max_abs > 1e-1  OU  NaN/Inf  OU  shape ≠ attendue  OU  distribution (mean/std) divergente
```

> **Cas diffus** `max_abs ≤ 1e-2` **mais** `mean_abs > 1e-4` = **NON-PASS** → WARN/investiguer : une erreur
> *diffuse* (moyenne haute, pic bas) est la signature d'un **bug systématique** (§6 étape 3), pas du drift
> concentré. Sans cette branche, ce cas tomberait dans aucun verdict.

Comparé : `last_hidden_state` ∈ ℝ^[1, S, 1536] (sortie de la final norm `Gemma4TextModel.norm`,
**avant** lm_head — le test logits est P5.7.6). Scan global (toutes les `S × 1536` valeurs) **et** points
fixes `[0,0,:8]` / `[0,S-1,:8]`. `max_abs = 0.0` global = drapeau **jaune** → test de perturbation (§6.6).

---

## 8. Options rejetées (traçabilité de la décision)

| Option | Nom | Raison du rejet |
|---|---|---|
| 2 | Moteur ZML bf16 | Trop de bruit pour P5.7.5 : déplace la cible de la *validation du câblage* vers le *comportement runtime bf16*. Le moteur ZML reste fp32 (invariant projet). |
| 3 | Tolérance structurelle 1e-1 comme critère **premier** | Acceptable seulement en **diagnostic de repli** (§6, bande WARN), pas comme critère de PASS d'entrée. Ne prouverait que le câblage, pas la précision. |

L'option 1 retenue est une variante *réalisable* de l'idée « oracle fp32 » (option (b) de la ROADMAP) :
fp32 rendu compatible mémoire par la seule exception bf16 du tensor dominant, **déjà validé en bf16**.

---

## 9. Hors périmètre de ce gate — suite

Ce gate `P5.7.5-prep` produit **uniquement** : ce contrat, la MAJ `PLANNING.md`, la MAJ
`docs/ROADMAP_to_full_forward.md`. **Il ne construit pas le moteur.**

**À faire en P5.7.5 (phase moteur, après commit de ce contrat)** :
1. **Régénérer l'oracle en mode hybride.** Le script actuel `scripts/38_p5_7_5_prefill_oracle.py` construit
   le modèle **entièrement en bf16** (`torch.set_default_dtype(torch.bfloat16)`) — **incompatible** avec ce
   contrat (un oracle bf16 ne peut pas valider un moteur fp32 au-delà de ~1e-1). Le fixture
   `fixtures/p5_7_5_prefill.safetensors` produit par ce script est donc **périmé**. Modification requise
   (esquisse) :
   ```python
   torch.set_default_dtype(torch.float32)          # modèle fp32...
   model = Gemma4TextModel(tc)
   model.model.embed_tokens_per_layer.to(torch.bfloat16)   # ...sauf ce module (exception mémoire)
   # chargement STREAMING (assigner tensor par tensor en libérant la source) pour rester < 23 Go :
   #   for k in safe_open(...): module_param.copy_(tensor.to(param.dtype)); del tensor
   #   (ou low_cpu_mem_usage / init meta) — NE PAS co-résider le state_dict complet (cf §2.2 note ²)
   ```
   Mettre aussi à jour `expected_zml_max_abs_le` du manifest (actuellement `2.0e-3`) pour refléter le PASS
   contractuel `1e-2` (le `2e-3` reste une *attente serrée* souhaitable, pas le seuil de PASS).
2. **Parité mémoire côté ZML.** Le moteur ZML doit lui aussi garder `embed_tokens_per_layer` en bf16 au
   chargement et faire `gather → fp32` sur les seules lignes des `input_ids` (comme P5.4), pour tenir sur la
   même VM 23 Go.
3. **Build moteur** = composition d'ops déjà validées : embedding (P5.4) + PLE frontend (P4.4) → boucle 35
   couches dispatchées sliding/full (P5.3 / P5.7.4) avec KV-sharing YOCO (capter K/V de la couche 13 sliding
   et 14 full, réutilisés par les readers 15-34) → final norm. Comparaison vs ce contrat.
4. **Puis** : P5.7.6 (logits vs HF), P5.7.7/.8 (decode).

**Discipline inchangée** : oracle = source de vérité (lire `modeling_gemma4.py`, ne rien supposer), gate
atomique, commit + tag, fixtures binaires gitignorées (régénérables), repo local-only.
