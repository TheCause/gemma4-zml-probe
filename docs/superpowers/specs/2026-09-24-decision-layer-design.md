# Spec — SD : couche de décision typée par lecture des logits (Choice mono-token)

> **Date** : 2026-09-24 · **Niveau de travail** : standard · **Branche** : `sd-decision-layer`
> (depuis `main` `81d98f8`) · **Statut** : **rév. 4**, écrite sur ancrages code AVANT toute
> mesure et avant la première ligne de code. Les prédictions §8 sont pré-enregistrées : ce
> fichier est committé avant le premier run. Le `git log` fait foi.
> **Décision Régis (24 sept 2026)** : « je veux que l'on implémente pour tester les gains
> potentiels », niveau standard, GO sur la conception en 4 blocs.
> **Origine** : une proposition d'architecture « Jev-style » et sa revue externe (non
> versionnées ici), vérifiées contre le code le 24 sept — toutes les affirmations de la
> revue sur le dépôt sont exactes (§1). Le périmètre retenu est celui que la revue
> recommande : Choice mono-token, runner séparé, oracle HF, moteur intouché.
>
> **Rév. 2 (24 sept, même session)** — 1ʳᵉ relecture par agent, 9 points, tous vérifiés sur
> disque et traités : (1) C-SD-D ne pouvait pas voir un cache non remis à zéro (les masques
> annulent tout `j > p`) → faute visée renommée ; (2) `exe.call` est asynchrone
> (`zml/exe.zig:288`, `wait=false`) → frontières de chronométrage définies, P1 rendue
> décidable ; (3) P2 ignorait l'écart de longueur des prompts → formule à deux termes ;
> (4) C-SD-A se vérifiait lui-même → dérivation indépendante des `label_ids` + contrôle de
> préfixe ; (5) la bannière `BUILD: mode=` n'existe que dans `gemma4_g12auto.zig:2027` →
> reprise de là, et C-SD-E restreint aux runs chronométrés ; (6) le choix de l'option n'était
> contrôlé nulle part → selftest à option attendue + recalcul au dépouillement ; (7) le
> « ~10× » du seuil 0,05 confondait marge et écart → retiré ; (8) deux ancrages vivent sur
> `b1-eviction-simulator`, pas `main` → corrigés ; (9) BOS, formes `{b,s,…}`, étape de
> déploiement → précisés.
>
> **Rév. 3 (24 sept, même session)** — 2ᵉ relecture, 9 points (6 importants) : (1) le mutant
> « cache non remis à zéro » de 87 visait une faute invisible → remplacé ; (2) l'évaluateur
> ne recevait pas l'affectation lettre→classe → `classes[4]` ajouté ; (3) C-SD-B′ ambigu →
> oracle indexé par id, mutation sur `zc` ; (4) BOS doublé invisible → mode de tokenisation
> fixé + contrôle `ids[0]==2, ids[1]!=2` ; (5) C-SD-D comparait deux compiles avec un témoin
> intra-processus → `fwd` et `rev` dans **un même processus** ; (6) P2 mal conditionnée →
> tolérance absolue, agrégat défini, signe prédit — **et le signe du terme de longueur était
> faux dans la rév. 2** (le prompt JSON plus long coûte plus au bras A : terme **positif** ;
> le relecteur portait la même erreur, corrigée ici par le calcul) ; (7-9) lectures du
> dernier pas, cas d'échauffement, flux de C-SD-C et syntaxe `TARGETS=…`.
>
> **Rév. 4 (24 sept, même session)** — 3ᵉ et dernière relecture, 3 points, **tranchés
> provisoirement par Claude sans 4ᵉ tour** (méthode : 3 tours max, puis l'humain) : (1) un
> oubli COMPLET de remise à 0 de `step` fait échouer le témoin de C-SD-D en même temps que
> le contrôle, et la règle le rangeait en INEXÉCUTABLE → `step0` journalisé, exigé à 0,
> mutation 87 dédiée ; (2) P2 : « chaque cas » contre « médiane », `t_C` et `len_lettre`
> non définis sur deux permutations, exclusion EOT → tout défini ; (3) C-SD-B′ : critère de
> choix du cas aligné sur ce qui fait mordre la mutation.
>
> **Rév. 4a (24 sept)** — C-SD-C : l'empreinte est le **sha256 normalisé** de
> `53_g2_3_hlo_check.py::normalized_hash` (script `88_sd_hlo_fingerprint.py`), pas un md5 brut ;
> même rôle, neutralisations éprouvées. Témoin `main` : `docs/evidence/sd/hlo_witness_main.txt`.

---

## 1. Faits établis (ancrés, `main` `81d98f8`)

| Fait | Ancrage |
|---|---|
| Le moteur calcule déjà `hidden → norm finale → embed_tokens (lm_head liée) → softcap → logits` et **renvoie les logits** | `zml_runner/engine.zig:765-766` (`forward`), `:806-807` (`forwardStageGen`), `:849-850` (`forwardStep`) |
| Le runner autonome E2B compile une seule entrée, `StepTok.forward` = gather embeddings + `forwardStep` + **`topK(5)` global** | `zml_runner/gemma4_gen_auto.zig:742-755` |
| Le top-5 global **ne suffit pas** à une distribution sur des options : une étiquette peut être hors top-5 | conséquence directe de la ligne précédente |
| Le prompt est absorbé **un token par pas** (prefill-par-decode) ; le vrai prefill S>1 est classé YAGNI | `docs/GEN_AUTONOME_DESIGN.md:32` |
| Débits mesurés du runner autonome (3090, fp32) : prefill **71,5 tok/s** (~14 ms/pas), génération **110-113 tok/s** (~9 ms/pas), compile **16,7-17,7 s** | `docs/L3_INGRAPH_DESIGN.md:111-116` |
| Régime de précision du runner autonome : **fp32** (PrecRt défaut) | `docs/GEN_AUTONOME_DESIGN.md:31,57` |
| « == HF » veut dire **même argmax sur les logits bruts** ; en bf16 le contrat est une enveloppe mesurée | `README.md:4`, `:26`, `:99` |
| En fp32, les bifurcations ZML/HF observées tombent à des marges top1−top2 de **0,006** et **0,0034** (après 590-960 pas) | `docs/GEN_AUTONOME_DESIGN.md:14`, `docs/L3_INGRAPH_DESIGN.md:116` |
| Le tokenizer ZML **diverge de HF sur du texte long** (11 ids sur 535) ; parade éprouvée : piloter par les **ids** | branche `b1-eviction-simulator` (commit `923b461`, **pas sur `main`**) : `docs/evidence/b1evict/FINDING_tokenizer_zml_vs_hf.md` ; résumé dans `PLANNING.md` de `main` |
| Sur le 12B, les bascules dépendent de la **marge**, pas de la précision | commit `ceaddb3`, branche `b1-eviction-simulator` (pas sur `main`) |
| Le mode de build Zig est un flag indépendant ; un log sans `mode=ReleaseFast` est **INEXÉCUTABLE** | `zml_runner/build_3090.sh:5-11` ; bannière `BUILD: mode=…` émise par `gemma4_g12auto.zig:2027` **seulement** — `gemma4_gen_auto.zig` ne l'a pas |
| `exe.call` est **asynchrone** par défaut ; seule une lecture device→hôte (`toSliceAlloc`) ou `callOpts(.{ .wait = true })` bloque | `~/dev/zml/zml/exe.zig:259-288` (ZML local `adee932e`, à reconfirmer sur la 3090) |
| Les masques additifs mettent `MASK_MIN` sur toute position `j > p` : un cas ne lit jamais les positions qu'il n'a pas écrites | `gemma4_gen_auto.zig:227-233` |
| `Tensor.logSumExp(axis)` existe (forme de sortie : l'axe réduit garde la taille 1) | `~/dev/zml/zml/tensor.zig:1399` |
| Les oracles HF tournent sur la 3090 (venv `gemma4-probe`, transformers 5.9.0, `HF_HOME=/data/hf_cache`, hors ligne) | `scripts/regen_fixtures.sh:9-17` |

**Mesure voisine (banc agentique local de l'auteur, hors dépôt, 23 sept 2026)** : l'adaptateur officiel TypeSafe 0.2.1 (génération de texte
structuré, **pas** de lecture de logits), avec Qwen3.8 27B à la fois décideur et exécutant, a
choisi la même action qu'une consigne fixe sur 2/2 reprises : **zéro gain de réussite**, plus
d'appels. Le même repérage liste trois moteurs open source qui **lisent les logits**
(OpenSourceJev/llama.cpp, open-jev/Gemma 3 4B MLX, OpenJev/Transformers), examinés par leurs
sources, non mesurés. Conséquences pour SD : (a) l'idée n'est pas neuve, l'apport de SD est un
moteur **Zig/ZML natif prouvé contre HF** ; (b) le gain de vitesse n'a de sens que si le
décideur est **plus petit que l'exécutant** — le cas d'usage visé est E2B décidant devant un
exécutant lourd (Qwen 27B sur la même 3090). Ce cas d'usage est **nommé, pas mesuré** ici.

## 2. Problème

Un appel d'outil par génération de texte coûte, après lecture du prompt, un pas de décodage
par token de la réponse structurée, puis un parsing fragile. Le moteur calcule à chaque pas
la distribution complète sur le vocabulaire : une décision parmi N options fermées peut se
lire **en un pas**, sur N logits, sans parsing. On veut **mesurer** ce que ça fait gagner, en
latence et en information (distribution, abstention), sur ce moteur précis.

## 3. Décision

1. **Runner séparé** `zml_runner/gemma4_decide.zig`, calqué sur `gemma4_gen_auto.zig` (même
   chargement, mêmes tables, même boucle prefill-par-decode), avec une entrée compilée neuve
   `StepDec`. `engine.zig` et `gemma4_gen_auto.zig` : **0 octet modifié**.
2. **V1 = Choice mono-token** : une question, N=4 options fermées, une étiquette mono-token
   par option, distribution lue à la position où le modèle doit émettre l'étiquette.
3. **Pilotage par ids** : les ids de chaque prompt sont produits par HF (template officiel),
   écrits dans un fichier de cas, et consommés tels quels par le runner. Le tokenizer ZML
   n'intervient pas. **Les ids du manifest incluent le BOS** : le runner ne préfixe **rien**
   (contrairement à `gen_auto`, qui préfixe `BOS_ID`, `gemma4_gen_auto.zig:53-55` — le
   recopier doublerait le BOS).
4. **Banc à 3 bras** sur les mêmes cas, même binaire, à chaud (§6).
5. Les probabilités sont nommées pour ce qu'elles sont : `p` (softmax sur l'ensemble
   autorisé), `p_max`, `margin` (p1−p2), `mass_in` (masse du vocabulaire tombée sur
   l'ensemble autorisé). **Jamais** « probabilité que la décision soit juste ».

## 4. Composants

### 4.1 Cas et prompts — `fixtures/sd_cases.json` (versionné, petit)

- **24 requêtes** en anglais, **6 par classe** : `direct` (répondre sans outil), `search`
  (chercher dans la documentation), `calculate` (faire un calcul), `insufficient` (pas assez
  d'information pour choisir). Chaque requête porte sa classe attendue. **Biais déclaré** :
  requêtes et classes attendues écrites par Claude, avant toute mesure, sans itération sur
  les résultats (le fichier est committé avant le premier run).
- **Prompt lettre** (bras B et C), tout dans le tour utilisateur :
  ```
  Decide how to handle the request below.
  A: answer directly without any tool
  B: search the documentation
  C: perform a calculation
  D: not enough information to choose

  Request: {q}

  Answer with a single letter: A, B, C or D.
  ```
  **Permutation** : même texte, affectation inversée des options aux lettres (A=insufficient,
  B=calculate, C=search, D=direct). Chaque cas existe en `perm=orig` et `perm=rev`.
- **Prompt JSON** (bras A) : mêmes options, mêmes descriptions, dernière ligne remplacée par
  `Answer with JSON only: {"choice": "<direct|search|calculate|insufficient>"}`.
  **Tranché** : JSON demandé dans le prompt, **pas** le format natif d'appel d'outil de
  Gemma — le format natif ne sait pas distinguer `direct` de `insufficient` (deux réponses
  sans appel), le critère de qualité deviendrait non décidable. Limite assumée : la longueur
  de réponse du bras A dépend du format ; `n_gen` est rapporté par cas pour que le lecteur
  puisse rééchelonner.
- Rendu : `apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True,
  enable_thinking=False)` si le template accepte ce paramètre ; sinon rendu sans le paramètre
  et **consigné** dans `docs/SD_RESULTS.md` (vérifié en SD0, pas supposé).

### 4.2 Oracle — `scripts/84_sd0_oracle.py` (3090, venv `gemma4-probe`, HF fp32 CUDA)

Pour chaque cas × perm (48 prompts lettre) et chaque cas (24 prompts JSON) :
- obtient `ids` par `apply_chat_template(..., tokenize=True)` **uniquement** (jamais
  rendu texte puis `tok(text)`, qui ajouterait un 2ᵉ BOS au `<bos>` du template) et refuse,
  nommément, tout prompt où `ids[0] != 2` ou `ids[1] == 2` ;
- écrit `ids` (liste d'entiers, BOS inclus) dans `fixtures/sd_manifest.json` avec, pour les
  prompts lettre, `label_ids` = ids des 4 lettres obtenus **indépendamment du rendu** :
  `tok.encode(L, add_special_tokens=False)`, qui doit rendre **exactement un** id (sinon refus
  nommé) ; l'affectation lettre→classe ; l'id de fin de tour (EOT) lu dans le tokenizer ;
- calcule, au dernier pas, les **logits HF fp32 après softcap** des 4 candidats, **stockés
  indexés par id de token** (`{"<id>": z}`, pas par position), le **logsumexp** sur tout le
  vocabulaire, et le top-5 ; écrit `fixtures/sd_oracle.json` ;
- produit aussi un prompt d'**échauffement** (`warmup`, hors des 24 cas, texte fixe) dans le
  manifest ;
- journalise versions (transformers, torch), révision du modèle, md5 du tokenizer.

### 4.3 Runner — `zml_runner/gemma4_decide.zig`

Entrée compilée (nom court, piège du quota comptime `@typeName` — cf. commentaire
`gemma4_gen_auto.zig:739-741`) :

```zig
const StepDec = struct {
    pub fn forward(model: Model, tabs: Tabs, tok: zml.Tensor, cand: zml.Tensor,
                   p: PackedLong, cache: engine.Cache, ctrl: engine.Ctrl)
        struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        // gather + forwardStep : IDENTIQUES à StepTok
        // t5   = logits.topK(.{ .voc = .voc }, 5, .{})      → bras B, diagnostic
        // zc   = logits.gather(.{ .voc = cand }, .{})        → {b,s,4} f32, bras C
        // lse  = logits.logSumExp(.voc)                      → {b,s,voc=1} f32 (mass_in)
        // (b = s = 1 : l'hôte lit 4 valeurs et 1 valeur)
        // retour : t5.values, t5.indices, zc, lse, slk, slv, flk, flv
    }
};
```

- `cand` : `{4}` u32, fourni par l'hôte à chaque pas (ids des étiquettes ; ignoré hors du
  dernier pas). N=4 est **compile-time** en V1.
- `logSumExp` : l'op ZML existante (`tensor.zig:1399`) ; si la version de ZML de la 3090 ne
  l'a pas, composition `max + log(sum(exp(x − max)))` sur `.voc`, notée dans le code.
- **Un seul compile** pour tous les cas et les trois bras. Le bras A réutilise `StepDec` en
  décodage glouton (top1 du top-5), `cand` ignoré.
- **Entre deux cas** : `step` repart de 0 (donc positions, RoPE et masques aussi) et le
  cache est ré-uploadé à zéro depuis `HostInputs` (`gemma4_gen_auto.zig:240-250`). La remise
  à zéro du cache est une **défense**, pas une nécessité prouvée : les masques annulent déjà
  toute position non réécrite (§1). Coût mesuré à part, **hors** des temps de décision.
- **Synchronisation** (fixe les frontières de §6) : pendant l'absorption, chaque pas **sauf le
  dernier** lit son top-5 (lecture bloquante, comme `gen_auto`) ; le **dernier** pas d'absorption est appelé
  avec `callOpts(.{ .wait = true })`, puis `t_absorb` s'arrête ; ensuite viennent les deux
  lectures, top-5 (`t_read_B`) et `zc`+`lse` (`t_read_C`), dans un **ordre alterné** selon
  la parité de la répétition (impaire : B puis C ; paire : C puis B).
- **Bannière de mode** : reprise de `gemma4_g12auto.zig:2027` (`BUILD: mode=…` via
  `builtin.mode`).
- CLI : `gemma4_decide <model.safetensors> --manifest fixtures/sd_manifest.json
  --arm {letter,json} --reps R --out results.jsonl [--order {fwd,both}] [--policy-selftest]
  [--allow-cpu]`. Garde VRAM reprise de `gen_auto`. Chaque ligne de résultat porte `rep`,
  `order`, `case_id`, `perm`, la `Decision` du runner (variante, option, `p`) et les valeurs
  brutes, `zc` **indexé par id** comme l'oracle, et `step0` (valeur passée à `Ctrl.step` au
  premier pas du cas). `--order both` : les R répétitions dans
  l'ordre de la liste, puis **une** passe en ordre inverse, **dans le même processus** (même
  compile). Le prompt `warmup` passe une fois après le compile, avant tout cas ; il est
  journalisé (froid) et exclu des statistiques.
- Arrêt du bras A : EOT (id lu dans le manifest, pas en dur) ou 96 tokens (plafond
  **journalisé** : `stop=max_tokens` rend le cas « format non conforme »). `n_gen_A` compte
  **tous** les tokens produits, EOT inclus (comme `generated` de `gen_auto`, l.1100-1105) ;
  le premier sort du dernier pas d'absorption, donc `n_gen_A − 1` pas de génération.

### 4.4 Évaluateur hôte (Zig, dans le même fichier)

Entrée : `zc[4]`, `lse`, `label_ids[4]`, `classes[4]` (affectation lettre→classe du cas,
lue dans le manifest, **propre au `perm`**), `vocab`. L'`option` rendue est une **classe**
(`direct|search|calculate|insufficient`), pas une lettre. Sortie : une union `Decision` :

| Variante | Condition |
|---|---|
| `.decision{ option, p[4], p_max, margin, mass_in }` | cas nominal ; `p` = softmax stable T=1 sur l'ensemble (formule §4.5) |
| `.abstain{ reason }` | option lue = `insufficient` **ou** `margin < θ` ; **θ = 0 en V1** (aucune abstention par marge dans les runs mesurés ; la courbe θ est calculée **au dépouillement**, descriptive) |
| `.err{ kind }` | `empty_set`, `duplicate_label`, `label_out_of_vocab`, `non_finite` |

Les validations `.err` passent **avant** toute lecture de `p` : une distribution concentrée
ne court-circuite jamais une erreur de schéma.

### 4.5 Formules (normatives)

- `p_i = exp(z_i − m) / Σ_{j∈A} exp(z_j − m)`, `m = max_{j∈A} z_j`, `z` = logits **après
  softcap**, fp32.
- `mass_in = exp(logsumexp_{j∈A} z_j − lse)` : fraction de la distribution du vocabulaire
  complet qui tombe sur les 4 étiquettes. Rapportée, jamais utilisée pour décider en V1.
- `margin = p_(1) − p_(2)`.

### 4.6 Dépouilleur — `scripts/86_sd_report.py` (M1, sans GPU)

Lit `results.jsonl`, `sd_oracle.json`, `sd_manifest.json` ; calcule les contrôles §7 et les
mesures §6 ; **recalcule** pour chaque cas l'option depuis `zc` et l'affectation lettre→classe
du manifest (propre au `perm`) et exige l'égalité avec la `Decision` du runner ; écrit `docs/SD_RESULTS.md` (tables) et sort non-zéro si une preuve manque
(**INEXÉCUTABLE**, jamais PASS par défaut). Contre-épreuve `scripts/87_sd_selfproof.py` :
fabrique des résultats mutés — (a) `zc` permutés d'un cran dans un résultat (C-SD-B′),
(b) la passe `rev` d'un cas altérée d'1 ulp (effet d'un `step` non remis à 0, C-SD-D),
(c) l'`option` du runner remplacée par celle de l'affectation `orig` sur un cas `perm=rev`
(C-SD-F), (d) un cas manquant, (e) `step0 ≠ 0` sur une ligne (doit rendre **FAIL** de C-SD-D, pas
INEXÉCUTABLE) — et vérifie que 86 les **condamne** ; déclare INEXÉCUTABLE si la mutation n'a rien
changé (leçon `scripts/81_selfproof_80.py`).

## 5. Flux

```
M1 : fixtures/sd_cases.json (committé)
  → 3090 : 84_sd0_oracle.py → sd_manifest.json + sd_oracle.json      [SD0, C-SD-A]
  AVANT la 1ʳᵉ ligne de code, sur main 81d98f8 :
  → 3090 : TARGETS=//examples/rqz:gemma4_gen_auto ./build_3090.sh ; run A1 avec
           XLA_FLAGS=--xla_dump_to=<d0> → md5 HLO témoin                     [C-SD-C]
  Après le code :
  → M1→3090 : deploy_to_3090.sh
  → 3090 : TARGETS="//examples/rqz:gemma4_decide //examples/rqz:gemma4_gen_auto" ./build_3090.sh
  → 3090 : gemma4_gen_auto (branche) A1 + dump HLO → md5 == témoin             [C-SD-C]
  → 3090 : gemma4_decide --policy-selftest                                     [C-SD-F]
  → 3090 : gemma4_decide --arm letter --reps 5 --order both → r_letter.jsonl   [C-SD-D]
           gemma4_decide --arm json   --reps 5 --order fwd  → r_json.jsonl
  → M1   : rapatriement, 86_sd_report.py, 87_sd_selfproof.py → docs/SD_RESULTS.md
```

## 6. Banc de mesure

- **Bras A** — prompt JSON, décodage glouton jusqu'à EOT. Temps = absorption + génération.
- **Bras B** — prompt lettre, top1 du top-5 au dernier pas d'absorption. Temps =
  `t_absorb + t_read_B`. Sortie hors étiquettes possible (comptée).
- **Bras C** — prompt lettre, même pas que B : `zc` + `lse` + évaluateur. Temps =
  `t_absorb + t_read_C + t_policy`.
- B et C sortent **du même run** (même pas, mêmes buffers) : l'écart B/C isole le coût de la
  lecture et de la politique.
- Segments chronométrés séparément (horloge `.awake`, API déjà utilisée par `gen_auto`) :
  `t_reset`, `t_absorb`, `t_read_B`, `t_read_C`, `t_policy`, `t_gen` (bras A : du retour du
    dernier pas d'absorption à la lecture de l'EOT). Frontières : §4.3 « Synchronisation ».
  Compile et prompt `warmup` (froid) rapportés à part, exclus des statistiques.
- 5 répétitions par bras, chaque répétition parcourant **toute la liste** de cas (cf. C-SD-D) ;
  médiane et p95 par bras ; gain relatif rapporté par cas en fonction de la longueur du prompt.
- **Qualité** (descriptive, sans seuil) : exactitude par bras et par classe ; bras A : JSON
  décodé (tokenizer HF, au dépouillement) et parsé, non-parsable = `format_error` ; bras B :
  taux hors étiquettes ; bras C : distribution de `mass_in`, taux de bascule orig↔rev
  (même classe prédite ou non), courbe couverture/erreur en fonction de θ.

## 7. Contrôles (chacun avec le résultat qui le fait échouer)

| Id | Contrôle | Échoue si… |
|---|---|---|
| **C-SD-A** (SD0) | Pour chaque prompt : `ids[0]==2` et `ids[1]!=2`. Pour chaque prompt lettre et chaque lettre L : rendre la conversation complète avec la réponse `L` en tour assistant (`add_generation_prompt=False`) → `full` ; exiger (i) `full[:len(ids)] == ids` (le rendu d'historique prolonge exactement le prompt de génération) et (ii) `full[len(ids)] == label_ids[L]`, où `label_ids[L]` vient de `encode(L)` **seul** (§4.2), pas de ce rendu | le template insère un marqueur (réflexion, saut de ligne) avant la réponse, rend l'historique autrement que le prompt, ou fait fusionner la lettre avec son voisin → (i) ou (ii) échoue ; BOS absent ou doublé. Un échec de C-SD-A **arrête** SD0 et remonte à Régis (pas de requalification automatique) |
| **C-SD-B** (SD1, numérique) | Sur les 48 prompts lettre : `max |zc_ZML − zc_HF| ≤ 0,05` et `max |lse_ZML − lse_HF| ≤ 0,05` (fp32 contre fp32 ; seuil **fixé a priori** : aucun écart ZML−HF à pas unique n'est publié dans le dépôt, le max observé est rapporté) | mauvais mapping des candidats, mauvaise position, softcap oublié, dérive numérique réelle. Les cas où la marge HF en logits est < 0,1 sont listés à part (lecture « à marge serrée », sans verdict) |
| **C-SD-B′** (morsure de B) | Mutant (a) de 87 : `zc` d'un résultat permutés d'un cran (valeurs échangées entre ids), comparés à l'oracle **par id** | le mutant **passe** → C-SD-B est vacueux. 87 choisit un cas où `max_i |z_HF[σ(i)] − z_HF[i]| > 0,1` (σ = décalage d'un cran, 2× le seuil) ; INEXÉCUTABLE s'il n'en existe aucun |
| **C-SD-C** (non-régression) | md5 du module HLO **pré-optimisation** (`*before_optimizations*`, dump `XLA_FLAGS=--xla_dump_to=<dir>`, neutralisation des éléments éphémères selon `scripts/53_g2_3_hlo_check.py` : chemins de dump, id numérique du nom de module) de `StepTok.forward` compilé par `gemma4_gen_auto` : témoin capturé sur `main` `81d98f8` **avant la 1ʳᵉ ligne de code**, re-capturé sur le binaire de la branche, égaux ; `git diff main -- zml_runner/engine.zig zml_runner/gemma4_gen_auto.zig` vide ; gate A1 de gen_auto (48/48 == HF) rejoué sur le binaire de la branche | un octet change dans l'un des deux fichiers, ou le HLO de `StepTok` change, ou A1 échoue |
| **C-SD-D** (isolation) | Les répétitions bouclent sur **la liste entière** (rep 1 = tous les cas, puis rep 2…). Témoin de déterminisme : `zc`/`lse` de chaque cas bit-identiques entre rep 1 et rep 2. Contrôle : bit-identiques entre rep 1 et la passe inverse. Témoin et contrôle viennent du **même processus** (`--order both`, même compile) | contrôle : **`step`/positions/masques non remis à 0 entre deux cas** (un cas dépend de la longueur de celui qui le précède). ⚠ Ce contrôle **ne voit pas** un cache non remis à zéro — les masques l'annulent (§1, §4.3) ; c'est pourquoi la remise à zéro est qualifiée de défense. Préalable : `step0 == 0` sur **toutes** les lignes, sinon **FAIL** (l'oubli complet de remise à 0, qui ferait aussi échouer le témoin, est attrapé ici). **Si ce préalable passe et que le témoin échoue** (GPU non déterministe d'un run à l'autre), C-SD-D est **INEXÉCUTABLE**, pas FAIL : l'instrument ne distingue plus fuite et bruit |
| **C-SD-E** (build) | chaque log **chronométré** de `gemma4_decide` porte `BUILD: mode=ReleaseFast`. Le rejeu A1 de `gen_auto` (C-SD-C, contrôle de justesse, non chronométré) en est exempté : sa preuve de build est la commande `build_3090.sh` journalisée + le sha256 du binaire | un log chronométré sans la bannière → run INEXÉCUTABLE (pas FAIL, pas PASS) |
| **C-SD-F** (évaluateur) | `--policy-selftest` : 4 entrées fabriquées déclenchent chacune leur `.err` nommée ; 2 entrées nominales à **option attendue connue** (argmax en position 0, puis en position 3) rendent `.decision` avec cette option et `Σp = 1 ± 1e-6` ; une entrée où l'option lue est `insufficient` rend `.abstain` ; une entrée **`perm=rev`** (mêmes `zc`, `classes` inversées) rend la classe inversée attendue. Au dépouillement, 86 exige option runner == option recalculée pour **tous** les cas (§4.6) | une erreur de schéma rend une décision ; une entrée nominale rend une erreur ; l'évaluateur rend l'argmin ou applique l'affectation `orig` à `perm=rev` |

## 8. Prédictions pré-enregistrées

- **P1** — `|médiane(t_read_C + t_policy) − médiane(t_read_B)| < 1 ms` (frontières §4.3,
  ordre des lectures alterné) : lire 4 logits + 1 scalaire au lieu d'un top-5 ne change rien
  de mesurable. *Réfutée si* l'écart dépasse 1 ms.
- **P2** — pour chaque cas, formule `F = (len_json − len_lettre) × c_abs + (n_gen_A − 1) ×
  c_gen` avec les coûts **publiés** `c_abs` = 14 ms, `c_gen` = 9 ms (§1). Les deux termes sont
  **positifs** (le prompt JSON est plus long, et le bras A génère). Gain mesuré par cas :
  `G = médiane_reps(t_A) − médiane_reps(t_C)`, où `t_C` d'un cas = médiane sur les 2
  permutations × répétitions 1-5 (la passe inverse de `--order both` est **exclue** : elle sert
  à C-SD-D seulement) et `len_lettre` = moyenne des deux permutations. **Population** : les cas
  dont le bras A se termine par EOT (les `stop=max_tokens` sont exclus de (a), (b) et (c), et
  comptés à part). Prédictions :
  (a) **signe** : `G > 0` pour **chaque** cas de la population ;
  (b) **amplitude** : `médiane_cas(|G − F|) ≤ max(0,3 × médiane_cas(F), 20 ms)` ;
  (c) **part** : `médiane_cas(G / médiane_reps(t_A)) < 25 %`.
  Ordre de grandeur attendu : ~10 tokens de prompt en plus et ~10 tokens de JSON, donc
  `F ≈ 220 ms` sur `t_A ≈ 1,6 s` (~14 %). Les coûts **mesurés** (`t_absorb/len`,
  `t_gen/(n_gen_A − 1)`) sont rapportés à côté sans remplacer la prédiction ; un écart aux
  publiés (les 71,5 tok/s de prefill viennent d'un run court juste après compile,
  `L3_INGRAPH_DESIGN.md:113`) est **publié**, pas absorbé. La part attribuable à la méthode
  elle-même est le **second terme** (génération évitée) ; le premier est un artefact de
  format du prompt, rapporté comme tel. *Réfutée si* (a), (b) ou (c) échoue.
  *24 sept 2026* — Longueurs mesurées en SD0 (`docs/evidence/sd/84_oracle.log`) : lettre
  médiane 77 (min 69, max 93), JSON médiane 86 (min 78, max 102).
- **P3** — **aucune prédiction de qualité.** On mesure.
- **Lecture attendue si P1 et P2 tiennent** : sur ce moteur, la vitesse vient de « une
  étiquette au lieu d'un JSON », que B obtient déjà ; l'apport propre de C est
  **l'information** (distribution, marge, masse hors ensemble), et le vrai levier de latence
  est le prefill S>1, hors périmètre.

## 9. Cas d'erreur

| Situation | Comportement | Contrôlé par |
|---|---|---|
| Label multi-token ou hors vocabulaire | SD0 refuse d'écrire le manifest, message nommant la lettre et le contexte | C-SD-A |
| Manifest absent, illisible, ou `label_ids` de taille ≠ 4 | refus au démarrage, `error.BadManifest` avec le champ fautif | test de démarrage dans le plan |
| Prompt > L_MAX (1024) | refus du cas, nommé, les autres continuent ; compté dans le rapport | dépouilleur |
| Bras A atteint 96 tokens | `stop=max_tokens`, `format_error` au dépouillement | §6 |
| Bras B sort une non-étiquette | compté `off_label`, pas d'erreur | §6 |
| `zc` ou `lse` non fini | `.err{non_finite}` | C-SD-F |
| Log sans `mode=ReleaseFast` | run INEXÉCUTABLE | C-SD-E |
| Fichier de résultats incomplet (cas manquant) | 86 sort non-zéro, INEXÉCUTABLE | 87 |

## 10. Hors périmètre, à dessein

Score et Noul ; questions indépendantes en lot et partage de préfixe/cache ; projection
réduite de la tête (`W_A` seules) ; prefill S>1 ; résolution et validation des arguments
d'outil ; format natif d'appel d'outil de Gemma ; calibration au sens de Guo et al. (la
courbe θ est descriptive) ; le 12B ; le cas d'usage « E2B décide devant Qwen 27B » (nommé
§1, non mesuré) ; toute intégration dans un agent.

## 11. Fichiers touchés

| Fichier | Nature |
|---|---|
| `docs/superpowers/specs/2026-09-24-decision-layer-design.md` | cette spec |
| `fixtures/sd_cases.json` | neuf, versionné |
| `fixtures/sd_manifest.json`, `fixtures/sd_oracle.json` | neufs, produits par 84, versionnés (petits, preuves) |
| `scripts/84_sd0_oracle.py`, `scripts/86_sd_report.py`, `scripts/87_sd_selfproof.py` | neufs |
| `zml_runner/gemma4_decide.zig` | neuf |
| `zml_runner/BUILD.bazel` | + une cible `gemma4_decide` |
| `docs/SD_RESULTS.md`, `docs/evidence/sd/*` | neufs (logs rapatriés) |
| `PLANNING.md` | une entrée « chantier SD » |
| `zml_runner/engine.zig`, `zml_runner/gemma4_gen_auto.zig` | **inchangés** (C-SD-C) |

Dépôt **public** : aucune IP, aucun hostname ni utilisateur réel dans les fichiers versionnés
(`user@gpu-host`, `/data/...` génériques) ; `git grep` des motifs avant tout push.

## 12. Estimation

SD0 (cas + oracle + C-SD-A) ~1,5 h · runner + évaluateur + build ~2-3 h (compiles 3090 de
~17 s, itérations de build Bazel) · runs ~20 min de 3090 · dépouilleur + contre-épreuve +
`SD_RESULTS.md` ~1,5 h. Total ~5-7 h de session, ~1,5-2 M jetons.
