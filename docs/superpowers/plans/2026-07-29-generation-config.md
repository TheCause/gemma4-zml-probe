# generation_config — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** faire appliquer `generation_config.json` (suppress_tokens + 3 EOS) au runner 12B et à
l'oracle, avec 12 gates pré-enregistrés, sans toucher au graphe.

**Architecture:** la politique est **host-side**. Le graphe sort déjà un top-5 trié
(`gemma4_g12auto.zig:825`, D2H `:1366-1384`) ; comme `|suppress| = 2`, l'argmax post-suppression est
de rang brut ≤ 3 ⇒ toujours dans ce top-5. Un module neuf `zml_runner/gencfg.zig` porte le parsing,
la découverte 1-hop, les validations, `isSuppressed`/`isEos` et le selftest ; `generateOnce` le
reçoit **par pointeur** à ses 3 sites d'appel. `engine.zig` : 0 octet.

**Tech Stack:** Zig `0.16.0-dev.2722` (API **`std.Io`**, pas `std.fs`) · Bazel sur la VM GPU ·
transformers **5.14.1** (venv `g12b` — `gemma4-probe` 5.9.0 ne charge pas `gemma4_unified`) ·
`ZML_REMOTE`/`ZML_DST` **obligatoires** en env (défauts = placeholders, le deploy rate en silence) ·
`--@zml//platforms:cuda=true` sur **chaque** run GPU (sinon repli CPU).

**Spec:** `docs/superpowers/specs/2026-07-28-generation-config-design.md` (**rév. 2**, `089270c`).

**Référence gates :** GC0 graphe · GC1 selftest · GC2 non-régression · GC3 mordant · GC4 vacuité ·
GC5 oracle · GC6 multi-EOS · GC7 équivalence · GC8 décisif C1 · GC9 @47 · GC10 repl · GC11 claim.

---

## Conventions d'exécution (à lire avant la Task 0)

- **Deploy** : `ZML_REMOTE=<user@host> ZML_DST=/data/rqz_workspace/zml/examples/rqz zml_runner/deploy_to_3090.sh`
  — exiger une sortie `rsync` **non vide**. Sans les variables, échec DNS silencieux.
- **Scripts Python** : la VM **n'est pas un dépôt git** et sa copie de `69_u8_gen_oracle.py` est
  **périmée**. Avant tout run oracle : `md5` des deux côtés, **identiques**, sinon on redéploie.
  C'est déjà arrivé (`tf_probe/tf200.json` produit par un script absent du chemin canonique).
- **Patch ZML** : `grep -n "local patch rqz" /data/rqz_workspace/zml/pjrt/pjrt.zig` **avant** tout
  build (fichier tracké modifié non committé ⇒ effaçable par un `git checkout` upstream).
- **VRAM** : `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` avant
  chaque run GPU. Seuil runner 20 GiB. `ollama serve` est actif mais sans modèle chargé.
- **Longs runs** : `nohup … > log 2>&1 < /dev/null; echo "DONE rc=$?" >> log` puis
  `until grep -q "^DONE rc=" log; do sleep 30; done`. **N'attendre `PASS` que des runs qui en
  émettent** (un filtre trop large a déjà matché une bannière).
- **Arguments pinnés** : figer `RUN_ARGS` **une fois** dans un tableau bash et le réutiliser
  partout — un argument qui bouge entre AVANT et APRÈS casse l'A/B à un seul facteur.
- **HLO** : comparer **`module_0001.zml.before_optimizations.txt`** (md5). Pas de bit-à-bit entre
  deux compiles XLA-GPU sur les fichiers post-opt.

---

## Task 0 : Témoins AVANT toute édition — et GC9 en parallèle

**Files:** aucun édit.

- [ ] **0.1** Worktree propre, noter le HEAD : `git status --short && git rev-parse --short HEAD`.
- [ ] **0.2** **VÉRIFIER que `rp0_witness/` est un témoin de la source COURANTE** (correction 37 :
      ne pas refaire ce qui est valide) :
      `git log -1 --format=%cI -- zml_runner/` doit être **antérieur** aux horodatages de
      `/data/gemma4-zml-probe/rp0_witness/`. Si oui → figer les md5 et **réutiliser**. Si non → tout
      re-tirer.
      *Attendu : la source n'a pas bougé depuis le 27 juil ⇒ témoins réutilisables.*
- [ ] **0.3** **Seule exception, à re-tirer** : le témoin 200 **avec `--dump-top5`** (GC3 a besoin
      du top-5 **ZML** @57, jamais mesuré — la prédiction `11814` est aujourd'hui dérivée du top-5
      **HF**). Même `RUN_ARGS`, un flag en plus.
      *Attendu : `258882` @57, et le top-5 ZML publié.*
- [ ] **0.4** Figer `RUN_ARGS` dans le plan d'exécution (tableau bash), et consigner le nombre de
      fichiers HLO du témoin (**510** attendus par dump).
- [ ] **0.5** **GC9 — instruire @47** (ne dépend pas du code modifié, donc ici) : teacher-forcing
      fp32 sur le témoin 200, `--dump-top5` côté ZML, comparer les logits ZML et HF **à la seule
      position 47**. Publier : marge, top-5 des deux côtés, écart par logit.
      **Verdict à publier, pas PASS/FAIL** : artefact de protocole (**C1 tient**) ou écart réel du
      forward (**C1 réfutée** ⇒ STOP et décision Régis).
- [ ] **0.6** Rapatrier dans `logs/`, md5 partout. Commit : `chore(gencfg): témoins T0 + verdict GC9`.

## Task 1 : `gencfg.zig` + selftest — TDD, sans GPU

**Files:** Create `zml_runner/gencfg.zig` · Modify `zml_runner/BUILD.bazel` (srcs de **3** cibles).

- [ ] **1.1** Écrire le **selftest d'abord** (il doit échouer) : `--selftest-gencfg <fixture>`,
      early-return sur le patron `:877-880` (avant tokenizer/VRAM/Platform).
- [ ] **1.2** Builder → **doit NE PAS compiler** (le module n'existe pas). C'est le test rouge.
- [ ] **1.3** Écrire `gencfg.zig` : `TOP_K`, struct `GenCfg` (spec §4.2bis), découverte 1-hop
      (`std.Io.Dir.cwd().readLink(io, ckpt, &buf)` → **longueur**, `error.NotLink` ; cible
      possiblement **relative** ⇒ résoudre contre `dirname(ckpt)`), parsing
      (`std.json.parseFromSlice(std.json.Value, …, .{ .allocate = .alloc_always })`, patron
      `gemma4_bbatch.zig:406-426`), les **6 validations** de la spec §4.1.
- [ ] **1.4** Référencer `TOP_K` aux **3 sites existants** (`:710`, `:825`, `:1380`) — pas de 4ᵉ
      copie du littéral `5`.
- [ ] **1.5** **GC1** : lancer le selftest. PASS = 100 % des cas de sélection **et** les **6**
      compteurs de non-vacuité non nuls (id supprimé top-1 · `EotNotInEosList` ·
      `SuppressIdOutOfRange` · `BeginSuppressUnsupported` · `EosListEmpty` · doublons dédupliqués).
      **FAIL ⇒ STOP** (règle d'arrêt).
- [ ] **1.6** Commit + `git tag gate/gc-1-pass` avec les chiffres.

## Task 2 : la fixture de GC1 (côté oracle)

**Files:** Create `scripts/71_gc1_fixture.py` · Create `fixtures/gc1_cases.safetensors` + manifest.

- [ ] **2.1** Écrire le producteur : il instancie le **vrai** `SuppressTokensLogitsProcessor` et
      calcule `expect_tok` = **argmax du vecteur COMPLET post-suppression** (c'est ce qui rend C2
      testable), plus le top-5 brut.
- [ ] **2.2** Cas de validation → sidecar JSON `{nom, contenu, erreur attendue}`.
- [ ] **2.3** Cas de découverte → étape shell : `mkdir -p <tmp>/snap && ln -s` reproduisant la
      topologie `weights_12b/` (symlink **absolu** *et* cas **relatif**).
- [ ] **2.4** ⚠ La fixture **porte l'`eot_id` de chaque cas** (le tokenizer n'est pas chargé sur ce
      chemin).
- [ ] **2.5** Commit.

## Task 3 : câblage dans le runner

**Files:** Modify `zml_runner/gemma4_g12auto.zig` (`Args` `:125-142`, `usage` `:144-150`,
`parseArgs` `:157+`, **commentaire CLI d'en-tête**, chargement avant `checkVram` `:948`, sélection
`:1384`, logs `:1387`/`:1389`, arrêt `:1424`, log de fin `:1542-1543`, garde d'exclusivité
`:868-874`).

- [ ] **3.1** Flags `--gen-config <FICHIER>` / `--no-gen-config` (**4** endroits par flag).
      `--selftest-gencfg` **rejoint** la garde d'exclusivité `--repl` ; `--gen-config` **non**.
- [ ] **3.2** Chargement + validations **après** l'early-return `--ids-only`, **avant** `checkVram`
      (fail-fast). Ordre : tokenizer → `eot_id` → politique → VRAM → compile.
- [ ] **3.3** Log `GENCFG:` au **format littéral imposé** (boucle manuelle : `[a,b]`, **pas**
      `{any}` qui produirait `{ a, b }`), avec `ignored=[…]` dérivé des clés présentes.
- [ ] **3.4** **Remplacer** `:1384` par le bloc de sélection (spec §4.2 — `?usize`, pas `?i32`) ;
      `var n_suppress_hits: usize = 0;` en tête de `generateOnce` ; compteur gardé par
      `in_gen_phase`, **sélection inconditionnelle**.
- [ ] **3.5** Logs `:1387`/`:1389` : ajouter `rank_used` / `chosen` + la **marge décisionnelle**
      (instrument de C2 — sans lui, l'histogramme prédit n'est pas mesurable).
- [ ] **3.6** Arrêt multi-EOS à `:1424` (`isEos`), **uniquement hors mode `--oracle`** ; log de fin
      **nommant** l'EOS qui a arrêté.
- [ ] **3.7** Propager la politique **par pointeur** aux **3** sites (`:1244`, `:1254`, `:1283`).
- [ ] **3.8** Deploy + build (`--@zml//platforms:cuda=true`) après `grep` du patch `pjrt.zig`.
- [ ] **3.9** **GC0** : md5 de `module_0001.zml.before_optimizations.txt` == témoin. **FAIL ⇒ STOP**
      (la thèse « politique host, graphe intact » serait fausse).
- [ ] **3.10** **GC2** : témoins 48 et 124 → ids bit-identiques, `n_suppress_hits == 0`, ligne
      `suppress=[258883,258882]` **littérale** présente. **FAIL ⇒ STOP**.
- [ ] **3.11** **GC3** : prompt du finding, 200 tokens. `258882` absent ; ids 0..56 bit-identiques ;
      `n_suppress_hits ≥ 1` ; token @57 == `11814` (comparé au top-5 ZML de 0.3) ; 1ʳᵉ divergence
      publiée.
- [ ] **3.12** **GC4** (a/b/c) avec les **contenus de fichiers imposés** par la spec. (b) doit
      **aller au bout**, pas échouer au chargement.
- [ ] **3.13** **GC10** : GC3 rejoué en `--repl`, ids lus via `--dump-top5` ; `n_suppress_hits`
      remis à zéro entre deux prompts.
- [ ] **3.14** Commits + tags `gate/gc-0-pass` … `gate/gc-4-pass`, `gate/gc-10-pass`.

## Task 4 : l'oracle — politique aux deux sites

**Files:** Modify `scripts/69_u8_gen_oracle.py`.

- [ ] **4.1** `--no-gen-policy` (défaut = politique **ON**), effectif aux **deux** sites (`:133`
      `step_top5` **et** `:304` boucle chunk).
- [ ] **4.2** Ordre imposé : asserts (`:130`, `:132`) sur les logits **BRUTS** → politique sur une
      **copie** → `topk`.
- [ ] **4.3** Schéma de sortie : `top5_per_pos` reste **le brut** ; ajouter `top5_policy_per_pos`
      (si `applied`), `gen_policy{…}` et **`script_md5`** (GC7 l'exige déjà). Assertion anti-`-inf`
      avant écriture. Assertion de version transformers.
- [ ] **4.4** Redéployer vers la VM + **contrôle md5 des deux côtés** (bloquant).
- [ ] **4.5** **GC5** : deux runs, même md5, un seul flag, `--compute-fp32`. Branche nue →
      `top5_per_pos[57].ids[0] == 258882` (**reproduction** : si elle échoue, c'est l'instrument) ;
      branche politique → `top5_policy_per_pos[57].ids[0] == 11814` ; `n_match` **199 → 198**.
- [ ] **4.6** **GC7** : témoin 200, politique des deux côtés → `n_match == n_total − 1`, **unique
      mismatch autorisé @47**.
- [ ] **4.7** Commits + tags.

## Task 5 : GC6 — exercer le chemin multi-EOS pour de vrai

**Files:** Modify `scripts/48_detokenize.py:37` (ajouter `"ids"` aux `choices`).

- [ ] **5.1** Confirmer sur le témoin AVANT que `496` a sa 1ʳᵉ occurrence à **gen=34** (< 57) et
      qu'il est absent des 29 `prompt_ids`.
- [ ] **5.2** Fabriquer `{"eos_token_id":[1,106,50,496],"suppress_tokens":[258883,258882]}`.
- [ ] **5.3** **GC6** : arrêt **exactement** à la position mesurée ; `stop_reason` nomme `496` ;
      sans le fichier, le même run va au bout.
- [ ] **5.4** Commit + tag.

## Task 6 : GC8 — le test décisif de C1

**Files:** Modify `scripts/69_u8_gen_oracle.py` (`--gen-policy-stop`, levée de la garde `:371-372`).

- [ ] **6.1** Lever la garde `--compute-fp32` hors teacher-force **ou** déclarer GC8 en bf16 **et
      recalculer son seuil** (le seuil 1,873e-3 est fp32 — le réutiliser en bf16 serait un
      changement d'instrument silencieux).
- [ ] **6.2** `--gen-policy-stop` (défaut **off**, pour ne pas casser la reproduction de
      `u8_gen48`).
- [ ] **6.3** **Mesurer sur 5 tokens** le coût réel avant de lancer (référence : 28,57 s/token en
      bf16 ⇒ 200 tokens ≈ 1 h 35 ; le fp32 est un autre ordre de grandeur). Dimensionner `n` ≥ 60
      (la zone décisive est 47-57).
- [ ] **6.4** **GC8** : séquences identiques jusqu'à l'arrêt, **ou** divergences toutes
      **< 1,873e-3**, publiées avec leur marge.
- [ ] **6.5** Croiser avec le verdict **GC9** de la Task 0 : C1 est **énonçable** ou **réfutée**.
      Écrire la conclusion, quelle qu'elle soit.
- [ ] **6.6** Commit + tag.

## Task 7 : docs, passe de nuance, clôture

- [ ] **7.1** `docs/GENERATION_CONFIG_RESULTS.md` : verdicts + chiffres + section **« Périmètre de
      la claim »** (modèle `MASKS_INGRAPH_RESULTS.md:70-73`) + **« Dette connue »** (E2B, EOS-only)
      + **impact sur D11** (`70_u8_corrupt.py` doit désormais passer `--gen-config <fichier>`).
- [ ] **7.2** Écrire explicitement que **`do_sample: true` n'est PAS appliqué** — « appliquer
      `generation_config.json` » signifie ici **suppression + EOS uniquement** (2 clés sur 8).
- [ ] **7.3** **GC11 — passe de nuance** : grep de recensement figé, exécuté **avant/après** ;
      0 site de catégorie (i) sans qualificatif ; **contre-preuve** : réintroduire une formulation
      nue doit faire **échouer** le grep.
- [ ] **7.4** `PLANNING.md` (cocher les 4 cases l. 40-43, ajouter la dette E2B),
      `docs/FINDING_GENERATION_CONFIG.md` (statut → corrigé), `README.md`.
- [ ] **7.5** **Anonymisation** : grep durci **avant** push (chemins `/Users/<user>`, IP LAN, alias,
      hôtes) — y compris logs rapatriés et manifests. ⚠ Le motif `regis` matche « en**regis**tré » :
      utiliser `/Users/[a-z]` et non le prénom nu, sinon le contrôle est illisible. **Contre-preuve
      exigée** : le grep doit détecter une chaîne test injectée.
- [ ] **7.6** Push + PR vers `main`, avec les 12 tags de gates.
