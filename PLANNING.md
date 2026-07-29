# Planning gemma4-zml-probe

> Sonde PLE puis portage ZML de `google/gemma-4-E2B-it`. Roadmap P-1 → P7 (section 10 procédure d'origine).

## État 9-10 juillet 2026 (🏁 portage validé CPU+GPU, G2 fidélité bf16 PASS — PR generation-longue → main)

**Le portage est complet et la claim de fidélité est solide aux deux régimes de précision.**

- **Génération longue validée (28/06, 3090)** sur les DEUX backends : CPU chunké L1a/L1b(ring)/L2
  autonome **1020/1020 == HF** ; GPU CUDA fp32 mono **1020/1020 == HF à 109 tok/s** (~350× vs CPU).
  Non-vacuité du masque prouvée par contre-test LOGITS. Tags `gen-long-validated-3090` +
  `gpu-baseline-validated-3090`.
- **Pipeline end-to-end texte→texte démontré** : `scripts/49_gen_custom_oracle.py` (oracle HF,
  prompt custom via chat template) → `gemma4_gen_long_gpu` (ZML == HF) → `scripts/48_detokenize.py`
  (détok + round-trip gate). Démo « capital of France » → « Paris », 48/48 == HF.
- **G2 fidélité bf16 PASS (4 juil, tags `gate/G2.*`)** — la claim « == HF » n'est PAS un artefact
  fp32. Méthode de l'enveloppe : G2.0 mesure combien HF-bf16 diverge de HF-fp32 (il ne se reproduit
  pas lui-même : 1016/1020, bifurcation step 21) ; G2.2 exige de ZML ≤ 2× cette enveloppe → ZML
  gemm-bf16 est **2 à 5× PLUS fidèle** au fp32 que HF-bf16 (max_abs p50 0.185 vs 0.425, KL 0.28×).
  Doc : `docs/G2_BF16_FIDELITY.md`.
- **Découverte G2.1** : poids DÉJÀ bf16 sur device (dtype du header safetensors), VRAM réelle
  **8,5 Go** (les ~22 Go = réserve BFC `memory_fraction 0.90`). Le « gain VRAM bf16 » du backlog
  n'existait pas ; le banc GPU tiendrait sur une carte 12 Go.
- **E1 rerun PASS 4/4** (`f74b8df`) : neutralité des édits G2 confirmée aux 3 niveaux (source/G1/E1).
- **Cette session (9 juil)** : PR `generation-longue` → `main` + rafraîchissement de ce PLANNING.

## 🔴 État 27 juillet 2026 — chantier `generation_config` PRIORITAIRE (découvert incidemment)

**Le portage n'applique pas `generation_config.json`.** Découvert en gelant les témoins du
chantier repetition penalty : le runner émet `<image|>` (id 258882) **en greedy** au milieu d'un
texte. A/B à un seul facteur sur le vrai 12B → le forward ZML est **innocenté** (ses logits
reproduisent HF, `<image|>` gagne de 0,25), mais Google déclare
`"suppress_tokens": [258883, 258882]` et **trois** `eos_token_id` `[1, 106, 50]`, que ni le
runner ni l'oracle n'implémentent. **`69_u8_gen_oracle.py` partage l'angle mort** : il fait aussi
un argmax nu, donc aucun gate existant ne pouvait détecter l'écart.
Preuves complètes : **`docs/FINDING_GENERATION_CONFIG.md`**.

À faire, dans cet ordre (décision Régis, 27 juil) :

- [x] **Cadrage (28-29 juil)** — spec `docs/superpowers/specs/2026-07-28-generation-config-design.md`
      (**rév. 3**, deux tours de revue adversariale, 86 findings arbitrés) + plan
      `docs/superpowers/plans/2026-07-29-generation-config.md` (8 tâches, 12 gates).
      **Pré-enregistrement** §2bis (5 claims falsifiables + prédictions chiffrées + « ce qui tue la
      claim ») committé **avant la première mesure** — exigence Régis « scientifiquement falsifiable ».
      Décisions Régis : périmètre **12B seul** · passe de nuance sur la claim **dans** ce chantier ·
      mode `--oracle` = suppression **ON**, arrêt EOS **OFF**.
- [x] **Task 0 (29 juil)** — témoins vérifiés **réutilisables** (source `zml_runner/` inchangée
      depuis le 26 juil, md5 local ≡ VM 4/4) ; **GC9 répondu** ; **GC12 mesuré** (N=20).
- [ ] **`suppress_tokens`** (logits → -inf avant l'argmax) + **3 EOS** dans le runner
- [ ] **Aligner `69_u8_gen_oracle.py` en même temps** — sinon l'angle mort persiste
- [ ] **Puis** le chantier repetition penalty (spec + plan déjà écrits et revus, ci-dessous)

## 🔴 29 juillet 2026 — FINDING : la trajectoire libre du 12B est BISTABLE

Découvert en Task 0, **avant toute modification de code**. 20 runs identiques du même binaire :
**exactement 2 trajectoires** (11/20 et 9/20), **un unique** point de bifurcation **@47** (marge
0,004587 ≈ 5× le bruit d'un logit), **149 des 200 ids** en aval. Positions 0..46 parfaitement
déterministes. Taux 45 % = signature d'un **tie**, pas d'un bruit.

- **La divergence @47 du finding `generation_config` §9 est EXPLIQUÉE** : ZML produit lui-même les
  deux valeurs. **Le forward est innocenté.**
- **2 gates pré-enregistrés retirés avant d'avoir coûté du GPU** (« ids 0..56 bit-identiques »,
  « reproduit le témoin bit-à-bit ») : ils auraient échoué **45 % du temps** pour une cause
  étrangère au chantier. **Contrôle qui échoue À TORT** — 3ᵉ membre de la famille. Remplacés par un
  critère **statistique** (`<image|>` dans 0/20 après vs **11/20** avant, p ≈ 6e-8 sous H0).
- **⚠ Les témoins du 27 juil ne sont PAS « caducs »** comme annoncé plus haut : ils sont valides
  (source inchangée) — mais **un témoin de trajectoire libre n'est pas un invariant**.
- **⚠ À ré-instruire** : les gates d'équivalence d'ids **en roue libre** (M1/M2, D1/D2, R0/R1,
  « 1020/1020 »). Les claims **teacher-forcées** (U8 48/48, U9 1150/1150, 4041 positions) ne sont
  **pas** affectées (marges ≥ 0,026).
- **Règle** : un gate de fidélité position-par-position doit être **teacher-forcé**.

Preuves : **`docs/FINDING_NONDETERMINISME_TRAJECTOIRE.md`**.

Deux conséquences à retenir :
1. La claim « ids == HF » est **vraie au sens « même argmax sur les logits bruts »**, et **fausse
   au sens « reproduit ce que `generate()` produirait »**. Nuance à écrire partout où la claim
   apparaît.
2. `do_sample: true, top_k: 64, top_p: 0.95, temperature: 1.0` → **le sampling est la
   configuration NOMINALE du modèle**, pas un raffinement optionnel. La « récitation en greedy »
   est plausiblement le symptôme d'un usage hors configuration prévue.
   ⚠ La récitation **n'a pas pu être reproduite** : 3 témoins greedy (2/200/1150 tokens) ne
   montrent aucune boucle (n-gramme max répété : 4 sur 200, 5 sur les 400 derniers de 1150 ;
   diversité plate 0,62-0,76 ; le run de 1150 atteint sa propre conclusion). Le prompt qui récite
   reste à identifier.

**Chantier repetition penalty — cadrage TERMINÉ, exécution suspendue** (branche
`sampling-penalty`) : spec rév. 4 + plan rév. 3, deux tours de revue chacun.
Task 0 faite (3 témoins gelés, archivés en durable côté GPU). Prérequis **RP-1** identifié :
l'oracle de décode 12B est **bf16 par construction** (`69:371` refuse `--compute-fp32` hors
teacher-force) — l'instrument déclaré corrompu le 25 juil ; à refonder en fp32 avant d'armer la
penalty.

### Planning courant

- [x] **PR `generation-longue` → `main`** — mergée le 9 juil (PR #3, `c4d483b`).
- [x] **G2.3 — cartographie de sensibilité bf16 par-op** — **LES 3 GATES PASS le 10 juil**
  (branche `g2.3-op-sensitivity`, tags `gate/G2.3.*`) : moteur `PrecRt` runtime (12 familles),
  sweep one-hot **12/12 SAFE** (classement softcap > norms > mlp > … > softmax), config combinée
  **12 familles SAFE (KL 0.486× l'enveloppe)**, interaction quasi-additive (1.06×), stabilité S49,
  VRAM kv_store −17 MiB (= ½ cache). Oracle anti-câblage-croisé 12/12 exact (2 découvertes :
  déduplication de nœuds au traçage ZML ; dédup inter-familles norms×ple nommée). Résultats :
  `docs/G2_3_OP_SENSITIVITY.md`. → PR vers main à merger.
- [x] **Batching / flash-attention** — **LIVRÉ 12 juil 2026** (branche `batching`, spec/plan
  `docs/superpowers/specs|plans/2026-07-12-batching-flash-attn*`, résultats
  `docs/BATCHING_RESULTS.md`, protocole pré-enregistré `docs/BATCH_BENCH_PROTOCOL.md`).
  **8 gates PASS** — Phase 1 : T0 (moteur **shape-polymorphe** : les 5 reshapes dérivent B/S des
  shapes d'entrée → **un binaire unique sert tous les B**, HLO **byte-identique** md5
  `ac9df2ae…`), B1 (primitives batchées : scatter batché **jamais exercé** + broad rank-égal
  prouvés), B2 (**48/48 == HF par lane** à B=2/B=4 + non-vacuité), B3 (lanes bit-identiques),
  B4 (sweep 1→64 : **113 → 2 106 tok/s**, ×18,5 ; non-régression ratio **0,999**).
  Phase 2 : S1 (byte-identique), S2 (sdpa 48/48 == HF), S3 (**PAS DE GAIN** : Δ ≈ 0 %).
  **3 findings** : (1) le **plafond n'est pas la VRAM mais le compute** — le pic ne bouge pas
  (16 670 MiB à tous les B, dominé par la compile) → la garde 20 GiB reste valide ; sweet spot
  **B=8-16** ; (2) le batching **n'introduit pas d'erreur, il expose la fragilité des ties**
  (même run B=4 : 4/4 en isolation, 3/4 dans le sweep — piège 15 sur l'argmax) ; (3) **sdpa ne
  gagne rien** car son chemin cudnn est du **code mort** (audit upstream
  `docs/ZML_UPSTREAM_AUDIT_2026-07-12.md` → **pas de bump ZML** : les 164 commits d'avance ne
  débloquent rien, FA2/FA3 assertent toujours B==1).
- [x] **Chantier W4 — poids 4-bit dans ZML → Gemma 4 12B sur la 3090** : **🏁 J1 LIVRÉ
  24 juil 2026** (branche `w4-brique`, 6 gates PASS, tags `gate/w4-w0-pass` …
  `gate/w4-wn-pass`) — brique `dequantW4` (int4 w4a16-ct groupé → bf16 en graphe) prouvée sur
  E2B : décode GPU **48/48 == HF-même-checkpoint** (40,9 tok/s, pic VRAM réel 10 524 MiB =
  **−37 % vs bf16**), `engine.zig` **0 octet modifié** (wrapper `W4Step`), finding
  « scale-invariance des norms » (10/11 familles — le contre-test scale doit cibler
  gate_proj) — **voir `docs/W4_RESULTS.md`**. **J2 (12B Unified) au tiroir, GO = décision
  Régis** (règle d'arrêt de la spec
  `docs/superpowers/specs/2026-07-18-w4-poids-4bit-12b-design.md` ; plan exécuté
  `docs/superpowers/plans/2026-07-24-w4-j1-brique-e2b.md` ; chiffres 12B déjà en poche :
  328 linears g32, 9,56 GiB, VRAM projetée ~10-12 Go ; ⚠ 8 couches full sans v_proj).
- [x] **Chantier W4-J2 — 12B `Gemma4Unified` : CLOS, 11 gates U0→U10 PASS (24-25 juil 2026,
  branche `w4-j2-12b`, **PR #12 MERGÉE 26 juil** — branche réécrite/anonymisée avant merge)** — **Gemma 4 12B décode sur la 3090** : 1150 tokens stables
  à **9,0 tok/s**, pic VRAM réel **16 680 MiB** (`--no-prealloc`, pic indépendant de la longueur ;
  bf16 infaisable : 24 Go de poids seuls), fenêtre 1024 mordante prouvée in-process (divergence
  exactement à q=1024). **Fidélité : == HF-fp32-même-checkpoint STRICT — 48/48 (U8 re-fondé) et
  1150/1150 (U9-iv), zéro requalification.** L'événement du chantier = **Amendement 3 (recadrage
  Régis)** : l'oracle bf16 (quantum 0.125) fabriquait des ties artificiels → requalifications en
  cascade (requalification U8 du 25 juil SUPERSEDED) ; réparation = oracle **fp32 sur stockage bf16** (script 69
  `--compute-fp32`, 626 hooks par-module — stockage ≠ arithmétique), 5× plus rapide que le bf16
  émulé. Moteur : Geom comptime, neutralité E2B **HLO byte-identique** (`gate/j2-u1-pass`).
  Contre-test gate_proj ×100 conforme (A1Mismatch, logits ×200). Findings : 12B hétérogène
  (sliding GQA 16/8×256, full MQA 1×512 K=V sans v_proj — erratum spec §4.2), q/k_norm full
  uniformes (QAT), leçon « instrument dégradé → requalifications en cascade » (mémoire + piège 21).
  **Voir `docs/U_12B_RESULTS.md`** ; plan + 3 amendements :
  `docs/superpowers/plans/2026-07-24-w4-j2-12b-unified.md`. Restes ouverts (backlog §9 : resserrer
  U7 à l'oracle fp32 ; arène XLA ~6 Go).
- [x] **Resserrage U7 à l'oracle fp32 : LIVRÉ (26 juil 2026, PR #14 mergée)** — max_abs
  **9,365e-4** vs oracle fp32 (×400 vs 0,376 ère bf16), seuil originel §3-U7 **1e-2 restauré
  et câblé**, top-5 ensemble+ordre exacts zéro tie. Les 3 requalifications bf16 de J2
  (U7/U8/U9) toutes résolues par l'instrument fp32 (Amendement 3).
- [x] **Contexte long 12B — variante 4k : LIVRÉE (26 juil 2026, PR #13 mergée)** :
  `G12Auto(comptime L_MAX)` (pattern bbs/bbatch), défaut 1280 **prouvé bit-identique**, cible
  `gemma4_g12a4k` (4096) **== HF-fp32 STRICT sur 4041 positions** (4000/4000 teacher-forcé
  oracle fp32, marge min 0,81) ; 8,2 tok/s, pic VRAM réel 22 234 MiB. ⚠ Mur documenté :
  masques `{L_MAX,L_MAX}` quadratiques → 8k exige des masques in-graph (backlog).
- [x] **Masques in-graph — LIVRÉ (26 juil 2026, branche `masks-ingraph`, gates M0-M3)** :
  le seul terme quadratique du design 12B (tables masques `{L_MAX,L_MAX}`) est supprimé —
  lignes générées dans le graphe (`engine.ingraphMaskLines` : iota+cmp+select, valeurs
  {0, MASK_MIN} exactes) depuis `positions[step]` + **fenêtre en scalaire runtime `window`**
  (rebindable : le contre-test de vacuité U9-ii survit tel quel). `Packed(MaskMode)` 3 variants,
  17 runners migrés. Gates : **M0 HLO byte-identique 4/4 témoins** ; **M1 48/48 ids + vacuity
  divergence exactement à q=1024 (zone mordante couverte)** ; **M2 124/124 ids** ; **M3 sonde
  8k = mur suivant chiffré** : compile PASS 38,9 s, OOM exécution — **double-buffering des
  caches KV** (2×5,6 GiB, alloc 2,50 GiB = un cache sliding), plus les masques. Pistes :
  donation PJRT des caches (8k passerait, statique ~17,3 GiB), ring sliding 1024.
  Doc : `docs/MASKS_INGRAPH_RESULTS.md`. Spec/plan : `docs/superpowers/{specs,plans}/2026-07-26-*`.
- [x] **Mode résident `--repl` — LIVRÉ (26 juil 2026 soir, branche `repl-mode`, gates
  R0-R2)** : load+compile payés UNE fois, prompts en boucle sur stdin (V1 indépendants,
  cache zéros par prompt), `generateOnce` extrait (un seul chemin one-shot/résident,
  engine 0 octet). R0 : 48/48 == témoin ; R1 : p1==p3==témoin + Canberra ; R2 : 20/20
  réponses, VRAM plate, RSS +1 Mo total. 2 pièges std.Io 0.16 consignés
  (`takeDelimiter` vs Exclusive ; writers multiples sur un fd s'entrelacent).
  Doc : `docs/REPL_RESULTS.md`.
- [x] **Donation des caches KV — LIVRÉE (26 juil 2026 soir, branche `cache-donation`,
  gates D0-D3)** : 4 `reuseBuffer` au return de `G12Step.forward` (`engine.zig` **0 octet**,
  gate D0 = diff vide) → input_output_alias PJRT, double-buffering supprimé. **Le mur M3
  est levé : le 12B décode 8028 positions sur la 3090** (2 passes vacuity complètes, fenêtre
  mordante exactement à q=1024 À 8k, pic 22 234 MiB — identique à 4k : le pic est le
  transitoire compile/autotune, pas le régime permanent). Équivalence D1/D2 : 48/48 et
  124/124 ids == témoins. Claim == HF inchangée (4041 positions).
  Doc : `docs/CACHE_DONATION_RESULTS.md`.
- [ ] **(option) 3e chantier — Triton paged attention** : seul chemin flash **B>1** crédible
  (B>1 natif, f32, scale custom, sliding window) mais exige un **bump ZML + refonte du cache
  YOCO vers un layout paginé**. Non démarré, cf. l'audit upstream.
- [x] **L3 in-graph** — **LIVRÉ 12 juil 2026** (branche `l3-ingraph`, spec/plan
  `docs/L3_INGRAPH_DESIGN.md`/`docs/L3_INGRAPH_PLAN.md`) : forward token→token, gather
  embeddings + `topK` désormais **dans le graphe** (`StepTok`/`Tabs`), le host ne thread plus
  qu'un scalaire u32 par step, `engine.zig` intact d'un octet. Gates SG/G1/G1v/G2/G2b/G3/VG
  **PASS** : génération **110-113 tok/s** ≥ replay 109 ≥ B0 pré-L3 (91,4/57,1 tok/s) ; G2b
  différentiel — bifurcation longue au step 960 ≥ replay 590 (non-déterminisme inter-compiles
  XLA-GPU déjà documenté, critère tenu dans les deux cas) ; VRAM pic mesuré 16,27 GiB (16 658
  MiB) → seuil de garde porté à **20 GiB** (`VRAM_CHECK_DESIGN.md` errata L3). Tag
  `gate/l3-ingraph-pass`.
- [x] **Check VRAM au lancement de `gemma4_gen_auto`** — **LIVRÉ 11 juil 2026** (branche
  `vram-check`, spec `docs/VRAM_CHECK_DESIGN.md`, gates V1-V3 PASS, tag `gate/vram-check-pass`) :
  garde intégrée `error.GpuBusy` si VRAM libre < 10 GiB (process occupants listés, suggestion
  `ollama stop`), échappatoire `--force-vram`, best-effort (nvidia-smi absent → warn+continue).
  Errata de revue : la garde tourne AUSSI en `--allow-cpu` (le flag ne force pas le CPU, l'init
  `.cuda` est tentée d'abord). Ne couvre QUE gen_auto — autres runners GPU : vérifier à la main.
- [x] **Runtime 100 % autonome** — **LIVRÉ 10-11 juil 2026** (branche `gen-autonome`,
  spec `docs/GEN_AUTONOME_DESIGN.md`, plan `docs/GEN_AUTONOME_PLAN.md`) : binaire
  `gemma4_gen_auto` texte→texte (tokenizer ZML natif, chat template Zig, prefill-par-decode,
  early-stop EOS, détok stdout). Gates : A0 ids==HF bit-exact ; A1 **48/48 autonome complet**
  (75-94 tok/s) ; A2 critère N/N FAIL publié → requalifié PASS différentiel (autonome ≥
  replay, même bifurcation marge 0.006 au step ~590 — décision Régis) ; A3 early-stop
  « Paris ». Non-régression E1+replay PASS, non-vacuité template PASS. Pièges neufs :
  tokens de tour `<|turn>`/`<turn|>` (EOT=106, lookup `<end_of_turn>`→unk silencieux) ;
  **repli CPU silencieux sans `--@zml//platforms:cuda=true`** (garde dure ajoutée) ;
  tolérance cos/sin ULP linéaire en position.
- [x] **Transfert G2.3 → TurboQuant / alambic** — **FAIT le 10 juil 2026** : notes livrées aux
  deux repos consommateurs (`turboquant/transfert_g23_zml.md` : baseline kv_store bf16, anti-cible
  softcap, banc rejouable ~40 s/run + pointeur CLAUDE.md ;
  `alambic/docs/decisions/2026-07-10-transfert-g23-precision-seuils.md` : régime bf16 student,
  budget bruit teacher ~1e-4 KL p50 dans les seuils, méthode de l'enveloppe — décision Régis
  requise avant le tag `prereg-v1`, consignée au PLANNING alambic). Cross-tasks soldées.

### Garde-fous courants

- **Contention VRAM 3090 (vécu 11 juil)** : un service Ollama local (~22/24 Go) peut occuper la
  carte — **vérifier `nvidia-smi --query-compute-apps` avant tout run GPU**. Symptôme si oublié :
  OOM dès la matérialisation (`CreateBuffersForAsyncHostToDevice … 6.00MiB`) suivi d'un crash
  `General protection exception` dans `io.zig deinit` — ce crash est un bug d'error-path UPSTREAM
  ZML (double-free post-OOM), pas notre code ; l'OOM est la vraie erreur. Libération réversible :
  `ollama stop <modèle>` (keep_alive recharge à la demande côté service). Depuis le 11 juil,
  `gemma4_gen_auto` intègre cette vérification (garde `error.GpuBusy`, cf item [x] check VRAM) —
  le réflexe `nvidia-smi` manuel reste NÉCESSAIRE pour tous les autres runners GPU. **Depuis L3
  (12 juil)** : le seuil de garde de `gemma4_gen_auto` est désormais **20 GiB** (pic mesuré
  16,27 GiB post-L3, table `embed_tokens_per_layer` en résidence device — cf item [x] L3 in-graph
  et `VRAM_CHECK_DESIGN.md` errata).
- **Piège `deploy_to_3090.sh`** : exige `ZML_REMOTE=user@gpu-host ZML_DST=/data/rqz_workspace/zml/examples/rqz`
  en env — les défauts sont des placeholders (`user@gpu-host`) → échec de résolution DNS ; avec la
  sortie redirigée, le deploy rate SILENCIEUSEMENT et on teste l'ancien binaire (vécu en
  non-vacuité, 11 juil).
- **GPU = flag de BUILD obligatoire** : `--@zml//platforms:cuda=true` sur chaque `bazel.sh run`
  GPU (sinon libpjrt_cuda hors runfiles → repli CPU ; refusé en dur par `gemma4_gen_auto`
  (`error.CudaRequired`), mais les AUTRES runners GPU du repo n'ont pas cette garde).
- **Piège workspace ZML** : patch local 1 ligne `@setEvalBranchQuota(100_000)` dans `pjrt.zig`
  (`structSize`, commenté `local patch rqz`) — **à réappliquer si le workspace ZML de la 3090 est
  resynchronisé upstream**. Requis dès qu'un `@typeName` de type (modèle, runner) devient assez long
  pour dépasser le quota comptime 1000 (`indexOf` sur `@typeName`) — un piège général, pas propre à
  une famille de runners donnée.
- Critère « 1020/1020 » exigible en fp32 seulement — en bf16, HF lui-même ne le tient pas (G2.0) ;
  le critère bf16 est l'enveloppe chiffrée de `docs/G2_BF16_FIDELITY.md` §7.1.
- Leçons méthodo toujours actives : argmax greedy trop robuste (comparer les LOGITS) ; un audit
  multi-agents ne remplace pas le compilateur ; oracle = source de vérité (voir garde-fous
  historiques en bas de fichier).

---

## État 2 juin 2026 (P5.7.5-prep — contrat de précision verrouillé, gate docs-only avant moteur 35 couches)

**Gate courant `P5.7.5-prep` ✅** — décision Régis : **oracle HYBRIDE** (fp32 sauf `embed_tokens_per_layer`
bf16). Contrat figé dans `docs/P5_7_5_precision_contract.md`. Périmètre = docs seulement, **aucun runner**.
- **Pourquoi hybride** : modèle texte full fp32 = 18,51 Go résidents / pic chargement ≈ 27,8 Go > VM 23 Go.
  `embed_tokens_per_layer` `[262144,8960]` = 9,40 Go (50,7 % des params résidents, bf16 sur disque → upcast
  ne récupère rien). Hybride = 13,82 Go ; **bit-identique au full fp32** (gather exact + ×√256=16 puissance
  de 2 exacte), rigueur fp32 préservée sur les 35 couches.
- **Seuils** : PASS `max_abs ≤ 1e-2` **ET** `mean_abs ≤ 1e-4` ; WARN `1e-2 < max_abs ≤ 1e-1` (→ investiguer
  câblage : distribution, localisation par couche, points fixes, suspects YOCO/dispatch/MLP-width) ; FAIL
  `> 1e-1` / NaN-Inf / mismatch shape ou distribution. Drift attendu = matmul Eigen-vs-BLAS accumulé sur
  35 couches (~1e-3..1e-2), concentré (mean_abs petit). Distinguer drift vs bug = §6 du contrat.
- **Dette pour P5.7.5 (phase moteur)** : `scripts/38_p5_7_5_prefill_oracle.py` est **full bf16** → à régénérer
  en hybride ; fixture `p5_7_5_prefill.safetensors` **périmé** ; `expected_zml_max_abs_le` du manifest
  (2e-3) à aligner sur le PASS contractuel (1e-2). Options 2 (moteur ZML bf16) / 3 (tol 1e-1 critère premier)
  rejetées. **Interdit** : démarrer le moteur 35 couches tant que ce contrat n'est pas committé (✅ fait).

## Etat 31 mai 2026 (P5.2.E.mask PASS — ZML sliding mask réel S=8/window=3, bit-exact)

P-1 ✅ · P2 ✅ · P3 ✅ · P4-prep ✅ · P4.3 ✅ · P4.4.0 ✅ · P4.4.1 ✅
**P4.4.2 ✅** gates : A ✅ B ✅ C ✅ D ✅ E ✅ F ✅ G ✅ H ✅ I ✅ **J ✅**
**P5.0 ✅ · P5.1 ✅ · P5.2.A ✅ · P5.2.B ✅ · P5.2.C ✅ complet (C.0/C.1/C.2/C.3)**
**P5.2.D ✅ COMPLET (branche V réparée)** : bug `v_norm` (RMSNorm `with_scale=False` non appliqué à V) découvert le 30 mai en préparant E, corrigé en 3 sous-gates.
  - **D.0b ✅** oracle K/V corrigé (V RMSNormed sans scale) — tag `p5.2-d0b-v-norm-oracle-pass`
  - **D.2b ✅** ZML v_norm sans scale — scan global 2.384e-7 (marge ~420 000×) — tag `p5.2-d2b-zml-v-norm-pass`
  - **D.5 ✅** KV slot corrigé (`value = v_after_norm`) — K_slot 5.36e-7, V_slot 4.17e-6, sanity 0.777 — tag `p5.2-d5-kv-slot-mock-pass` (ancien V-brut `p5.2-d5-zml-kv-slot-mock-pass` superseded)
**P5.2.E ✅ COMPLET** (pilote = layer 15 reader → KV layer 13) — chaîne d'attention ZML `QK → mask → softmax → context` validée bout-en-bout vs oracles PyTorch indépendants :
  - **E.0 ✅** oracle PyTorch attention `Q15×K13ᵀ → mask → softmax → V13 → context` — scaling 1.0, GQA repeat_kv 8, masque additif, V normé (0.7768), sliding≡causal prouvé à S=4<512, Σprobs−1=1.19e-7, futur=0 strict — fixture `p5_2_e0_attention_oracle_layer15_kv13.pt` (md5 `f88ea58d…` M1≡3090) — tag `p5.2-e0-pytorch-attention-oracle-pass`
  - **E.1 ✅** ZML QK scores only — `splitAxis(Q GQA) → dot(.hd) → merge(.h,.hq) → transpose` (scaling 1.0, PAS de `1/√hd`), forward `[1,8,4,4]` ✓, scan global **max_abs 2.384e-6** (marge ~42 000× vs 1e-4) vs `scores_raw`. Helper `sdpa`/`attention` écartés (n'exposent pas les scores bruts) → dot manuel. Vérif adversariale : 2 expressions GQA (repeat_kv PyTorch vs splitAxis ZML) convergent. Runner `gemma4_qk_scores.zig` — tag `p5.2-e1-zml-qk-scores-pass`
  - **E.mask ✅** sliding mask **réel** S=8/window=3 (cas mordant) — helper natif `zml.nn.causalAttnMask(.{.q=8,.k=8}, .f32, 3)` + `add(broad)`. Grille ZML == oracle, masked **43/64** (mask) **86/128** (scores), visible **bit-exact** (0.0), struct_mismatch=0. Convention triple-validée (transformers `sliding_window_overlay` = helper ZML = table Régis). Preuve que la fenêtre mord : causal pur masquerait 28, sliding 43 (+15 anciennes). Garde `qlen>=window` = ce qui dégénérait E.0/E.1. Runner `gemma4_sliding_mask.zig` — tag `p5.2-emask-sliding-mask-pass`
  - **E.softmax ✅** ZML softmax only — `scores_masked.softmax(.k)` fp32 (conv. sdpa), forward `[1,8,4,4]` ✓, Σprobs−1=**1.19e-7**, futur masqué=**0** strict, NaN/Inf=false, 3 fixed-points **bit-exact** (max_diff 0.0), scan global **max_abs 2.98e-8** (marge ~3300× vs 1e-4, < jitter QK E.1 car softmax borne [0,1]). finfo.min géré (sub max → exp=0). **Piège build** : nom long `gemma4_attention_softmax` (24c) débordait le quota comptime 1000 branches de `pjrt.zig structSize` (`@setEvalBranchQuota` dans `main` n'atteint pas cette Sema) → renommé runner **`gemma4_softmax.zig`** (14c). Script `24_` (23 pris). — tag `p5.2-esoftmax-zml-pass`
  - **E.context ✅** ZML context dot — `probs.splitAxis(.h){.h=1,.hq=8}.dot(v_final,.k).merge(.h,.hq).transpose` (GQA par split des têtes Q, miroir E.1 ; V=`v_final` RMSNorm no-scale D.0b), forward `[1,8,4,256]` ✓, NaN/Inf=false, max|context|=6.26, 3 fixed-points (dont **h=7** → GQA correcte) **bit-exact**, scan global **max_abs 0.0 / mean_abs 0.0 bit-exact** (réduction .k=4 courte → PJRT-CPU≡BLAS). Garde anti-régression V auto-contenue : RMS(v_final,hd)≈1 (dev 4.77e-7). Design figé par **workflow 4-agents adversarial** (split PROBS pas V). Runner `gemma4_context.zig` (14c). Script `25_`. — tag `p5.2-econtext-zml-pass`
**P5.2.F ✅** ZML o_proj (projection sortie attention, layer 15) — `context.transpose({.b,.q,.h,.hd}).merge({.m={.h,.hd}}).dot(o_proj_weight,.m)` → `[1,4,1536]`. o_proj TEXTE = `nn.Linear(2048,1536,bias=False)` **sans clipping** (Gemma4ClippableLinear = attention VISION, fausse alerte écartée). Transpose(1,2) manquant dans context E.0 géré. Concat têtes h-major confirmé (einsum oracle 5.72e-6). Forward `[1,4,1536]`, 2 fixed-points max_diff 2.50e-6, scan global **max_abs 2.29e-5** (réduction .m=2048, cohérent q_proj C.1 ; résidu non-nul → réfute echo oracle). Runner `gemma4_oproj.zig`. Script `26_`. Oracle `nn.Linear` lit `...layers.15.self_attn.o_proj.weight` [1536,2048]. — tag `p5.2-f-oproj-zml-pass`
**P5.2.G ✅** ZML post_attention_layernorm + résiduel (layer 15) — `out = residual + rmsNorm(attn_output,.d).mul(pa_norm_weight)`. Ground truth `Gemma4TextDecoderLayer.forward` L1395-1406 (sandwich norm). post_attn_ln = `Gemma4RMSNorm(1536,eps=1e-6,with_scale)` pattern Llama (`*weight`, weight non-uniforme mean 0.914). Oracle = **module réel Gemma4RMSNorm** (pas ré-dérivation). residual = stand-in `hidden_input` C.0 (vrai résiduel pré-input_layernorm non modélisé par le pilote → valide l'OP, pas la sémantique e2e). Forward `[1,4,1536]`, 2 fixed-points 2.38e-7, scan global **max_abs 9.54e-7** (marge ~100 000×). Runner `gemma4_attn_resid.zig`. Script `27_`. — tag `p5.2-g-attn-resid-zml-pass`
**P5.2.H ✅** ZML MLP feed-forward (layer 15) — `out = residual + post_ff_norm(down(gelu(gate(x))*up(x)))`, `x=pre_ff_norm(residual)`. **Découverte : layer 15 (reader, KV-shared) + use_double_wide_mlp → intermediate=12288** (pas 6144 ; layers 0-14=6144, 15-34=12288). `gelu_pytorch_tanh` = `Tensor.gelu` ZML (confirmé 0.0). gating `gelu(gate)*up` (pas gelu(gate*up)). Oracle = modules réels Gemma4RMSNorm+ACT2FN. Forward `[1,4,1536]`, max|out|=103.6, 2 fixed-points 1.43e-5, scan global **max_abs 5.34e-5** (marge ~1.9×, réduction .f=12288). Runner `gemma4_mlp.zig`, script `28_`. — tag `p5.2-h-mlp-zml-pass`
**P5.6 ✅** ZML full_attention Q-rope manuelle partielle (layer 14) — **RISQUE DU PROJET LEVÉ**. Découverte : full attn = **head_dim 512** (q_proj [4096,1536], pas 256), partial_rotary 0.25 (128/512 dims tournent), theta=1e6, scaling=1.0. `zml.nn.rope` ne couvre pas proportional → **RoPE manuelle** : cos/sin oracle (512-wide, 384 identité), `rotate_half` via `split(.hd,{256,256})`/`negate`/`concatenate`, `q*cos+rh*sin`. Sanity oracle : manuelle == apply_rotary_pos_emb à 0.0. Scan global **max_abs 7.99e-6** (marge ~12×). Runner `gemma4_full_qrope.zig`, script `29_`. Le reste du chemin full = identique sliding (E/F) avec dims 512. — tag `p5.6-full-qrope-zml-pass`
**P5.4 ✅** ZML embedding gather + scale √1536 (slice vocab 4096) — `weight.gather(.{.voc=ids}).scale(√1536)`. Op gather validée (P4.4 avait validé le scale, pas le gather). **bit-exact** (max_abs 0.0). Runner `gemma4_embed.zig`, script `30_`. — tag `p5.4-embed-zml-pass`
**P5.5 ✅** ZML head (final norm + lm_head tied + softcap) — `rmsNorm(hidden).mul(norm_w)` → `dot(lm_head_slice,.d)` → `scale(1/30).tanh().scale(30)`. Op neuve softcap (Tensor.tanh). lm_head tied=embed_tokens, slice vocab 4096. Forward `[1,4,4096]`, max|logits|=29.6 (borné ±30), 2 fixed-points 1.86e-5, scan **max_abs 5.44e-5**. Runner `gemma4_head.zig`, script `31_`. — tag `p5.5-head-zml-pass`
**P5.3 ✅** ZML **couche décodeur sliding COMPLÈTE** (layer 13 producer) vs module réel `Gemma4TextDecoderLayer` — compose input_ln + attention(QKV/norm/rope/QK/mask/softmax/context/o_proj) + résiduel + MLP(6144) + résiduel + bloc PLE per-layer (gate/proj/norm + per_layer_input) + layer_scalar(0.0884). Gestion tags (rename .s→.q/.k, GQA splits) OK du 1er essai. Forward `[1,4,1536]`, 2 fixed-points 6.68e-6, scan **max_abs 6.72e-5** (marge ~7.4×). Runner `gemma4_layer.zig`, script `32_`. — tag `p5.3-layer-zml-pass`
**P5.6.K ✅** ZML full_attention K-rope manuelle (layer 14) — ferme le gap K-full-rope de l'audit closeout (même technique que P5.6 sur K). scan 2.68e-7. tag `p5.6k-full-krope-zml-pass`.
**P5.6.closeout ✅ (2 juin)** — audit complétude (workflow 3 agents) : **matrice composant→runner/tag/preuve/tolérance** (`docs/P5_6_closeout.md`), 0 gap, tolérances ALL_JUSTIFIED, 0 faux invariant, tags superseded documentés. **Base saine pour P5.7.**
**P5.7.0 ✅** loader manifest only — `scripts/34_p5_7_0_loader_manifest.py` produit `fixtures/p5_7_0_loader_manifest.json` sans compute ni chargement payload. Résumé : 600 clés disque attendues, 540 tenseurs runtime à charger, 60 clés K/V reader disk-only ignorées par YOCO, `v_norm` documenté comme op sans poids (`with_scale=False`). Validation checkpoint optionnelle via `--require-weights`; sautée localement car `weights/model.safetensors` absent.
**P5.7.0 ✅ (Codex/ChatGPT, vérifié Claude 2 juin)** — loader manifest 35 couches (`scripts/34_p5_7_0_loader_manifest.py`, `fixtures/p5_7_0_loader_manifest.json`). **Vérification Claude contre le checkpoint réel sur 3090** (ce que Codex avait sauté) : `--require-weights` PASS, **600 clés / 0 manquante / 0 mismatch** ; reverse-check **600 clés `model.language_model.*` = 600 attendues, 0 oubli, 0 extra** (1411 autres = tours vision/audio hors scope). Shapes archi correctes (full head_dim 512, MLP double-wide 12288 readers, YOCO producers 0-14 / readers 15-34 → target 13 sliding / 14 full, K/V readers disque-only ignorés). Manifest committé mis à jour SKIPPED→PASS.
Tag courant : `p5.6k-full-krope-zml-pass`. **🎯 COUCHE DÉCODEUR E2E + TOUTES OPS DISTINCTES VALIDÉES + AUDIT CLOSEOUT + P5.7.0 manifest.** Reste = P5.7 runtime (P5.7.1→.8). Idéal contexte frais. **🎯 TOUTES LES OPS DISTINCTES DU FORWARD VALIDÉES EN ZML.** Reste = intégration (35 couches + KV cache + PLE). Voir ROADMAP. **MODE AUTONOME** (re-priorisé par risque) : ~~H~~ ~~P5.6~~ faits → **P5.4 (embedding+scale)** → **P5.5 (final norm+lm_head+softcap)** → P5.3 (assemblage couche e2e) → P5.7 (multi-couches). Après P5.4/5.5 : toutes ops distinctes validées. Plan = `docs/ROADMAP_to_full_forward.md`.

> **Gemma 4 E2B PLE minimal ZML validated end-to-end.**

Synthèse numérique P4.4.2 :
- A→G : tous bit-exact PJRT CPU vs numpy fp32 (`max_diff = 0.0`).
- H (rmsNorm + weight) : 1.49e-8 vs numpy fp32 — 1 ULP fp32 sur 2 blocks, bit-exact sur 2 autres.
- I (fusion add) : 1.49e-8 — heritage exact de H, l'add n'ajoute aucun drift.
- J (scale 1/√2 + comparaison fixture) :
  - 4 blocks max_diff : **1.79e-7** vs fixture fp32 (`ple_reference_final.npy`).
  - Scan global 35840 valeurs : max_abs **1.526e-5** (flat_index 10756), mean_abs **1.85e-7**.
  - Tolérance 1e-4, marge ~6500×.
  - Résidu confirmé = matmul PJRT-CPU Eigen-like vs PyTorch BLAS (P4.3 observait 1.53e-5 numpy vs PyTorch, on retrouve à l'octet près).

Le matmul Gate E est la seule source de divergence ; tout le reste de la chaîne PLE reproduit PyTorch à <2e-7. Le PLE-only ZML minimal est **validé end-to-end** pour gemma-4-E2B-it sur l'input `'ZML test prompt'`.

**Next session: P5 (YOCO / Shared KV) — débloqué, non démarré.**

### Connaissance capitalisée pour P5 et au-delà

1. **Piège ZML #1 — reshape perd les tags** : `reshape(.{...})` retourne shape anonyme. Re-tagger via `.withTags(.{ ... })` avant toute op qui cible un axe par tag (rmsNorm, mul/add cross-tagged).
2. **Piège ZML #2 — mul/add ne broadcastent pas implicitement** : `weight.broad(other.shape())` obligatoire (pattern Llama `model.zig:391`).
3. **Choix RMSNorm verrouillé** : pattern Llama `normalized.mul(weight)`. Variante Qwen3.5 `normalized.mul(1+weight)` interdite pour Gemma 4.
4. **Numérique attendu** : matmul PJRT-CPU vs PyTorch BLAS introduit ~1.5e-5 résidu. Pour valider une couche entière vs référence PyTorch, viser tolérance 1e-4 ; pour valider la couche vs numpy fp32 reproduit localement, viser bit-exact ou 1 ULP.
5. **Piège Gemma4 #1 — `with_scale=False` ≠ pas de normalisation** : `v_norm = Gemma4RMSNorm(head_dim, eps, with_scale=False)` normalise V (division RMS) **sans** poids appris. L'absence de `v_norm.weight` au checkpoint = « pas de poids », PAS « pas de norm ». V est RMSNormé sans scale ; K et Q sont RMSNormés AVEC scale (`with_scale=True`). En ZML : V = `zml.nn.rmsNorm(v_4d, .hd, eps)` **sans** `.mul(weight)`. (bug D.0→D.0b, 30 mai)
6. **Principe méthodo — l'oracle doit être indépendant du code testé** : le bug v_norm a survécu à un PASS « end-to-end » parce que l'oracle PyTorch ET l'implémentation ZML partageaient la même hypothèse fausse → ils s'accordaient à ~5e-6 (fausse confiance). Un oracle ne révèle un bug que s'il dérive de la **source de vérité** (`modeling_gemma4.py`), jamais d'une hypothèse ré-encodée à la main.
7. **Faits attention Gemma4 (pour P5.2.E)** : `Gemma4TextAttention.scaling = 1.0` (PAS √head_dim — la norm passe par q_norm), **pas de softcap d'attention** (seulement `final_logit_softcapping` en P7), GQA via `repeat_kv` (`num_key_value_groups = 8`), masque **additif**, softmax fp32.

## Planning historique (état 31 mai — tout est clos depuis, voir « Planning courant » en tête)

### Haute priorité

- [x] **P4.4.2 — Mini-runner ZML PLE-only**
  - Charger `fixtures/ple_fixture.safetensors` via `zml.safetensors.TensorRegistry`.
  - Reproduire le pipeline PLE :
    `lookup × √1536`, `lookup × √256`, projection `× 1/√1536`, reshape, RMSNorm Gemma 4 pure `* weight` ε=1e-6, fusion `/√2`.
  - Comparer à `ple_reference_final` (chargé depuis le même safetensors).
  - Gate fp32 : `max_abs ≤ 1e-4` + fixed point `[0,0,0,:4]` aligné.
  - Sans YOCO, sans attention, sans KV-cache.

### Medium

- [x] **P5 — Shared KV / YOCO** (`num_kv_shared_layers=20`) : inspecter forward Transformers, tracer shapes et cache lifecycle.
- [x] **P6 — Attention hybride** : pattern `layer_types` 4×sliding + 1×full × 7 (full aux couches 4, 9, 14, 19, 24, 29, 34), p-RoPE.

### Backlog

- [x] **P7 — Logits** : `final_logit_softcapping = 30.0`, top-k overlap, flip-rate temp=0.
- [B] Intégrer `05` et `06` dans `04_run_all.sh`.
- [x] Tester d'autres `input_ids` que `'ZML test prompt'` (fait : gén longue 1020 tokens + prompts custom script 49).

## Garde-fous (historiques — pièges ZML/Gemma4 toujours valides)

- Ne PAS écrire `gemma4.zig` complet avant que P4.4.2 mini-runner PLE-only passe.
- Ne PAS ouvrir P5 (YOCO) tant que P4.4.2 n'est pas fermé.
- Référence à viser en P4.4.2 : **tensor `ple_reference_final` dans `fixtures/ple_fixture.safetensors`** (fp32, vérité math depuis P3) — pas la version bf16 du `.pt` Transformers.
- `fixtures/ple_fixture.safetensors` est gitignored mais régénérable via `scripts/08_export_safetensors_fixture.py` (depuis les `.npy` versionnés).
- Piège RmsNorm : `zml.nn.rmsNorm` est neutre. **NE PAS** réutiliser le wrapper `RmsNorm` de `examples/llm/models/qwen3_5/model.zig` (variante `1+weight`). Suivre le pattern Llama (`examples/llm/models/llama/model.zig:391`) : `normalized.mul(weight.broad(x.shape()))` sans `.add(normalized)`.
- **Piège ZML #1 — reshape perd les tags** : `tensor.reshape(.{1,4,35,256})` retourne un `Tensor({1,4,35,256, f32})` anonyme. Pour cibler un axe par tag (`rmsNorm(x, .d, eps)`, `add` avec un tenseur taggué, etc.), il faut chaîner `.withTags(.{ .b, .s, .l, .d })` immédiatement après le reshape. Observé Gate H, valable pour Gate I et au-delà.
- **Piège ZML #2 — `mul`/`add` ne broadcastent pas implicitement** : `normalized {.b,.s,.l,.d}.mul(weight {.d})` panique `mul expects tensor shapes to match`. Il faut expliciter le broadcast : `weight.broad(normalized.shape())`. Pareil pour `add`. Pattern Llama exact : `normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()))`. Le `.convert(dtype)` est utile en mixed-precision ; ici tout est fp32, on l'omet.
- Compute : 3090 pour Python (`/data/gemma4-zml-probe`, venv `/data/venvs/gemma4-probe`). ZML est dans `/data/rqz_workspace/zml` sur la 3090. Pour le portage Zig, machine = 3090 (Bazel + accès `examples/`).
- **Oracle = source de vérité, pas hypothèse** : avant de coder un oracle PyTorch d'une couche, LIRE le `forward` de référence dans `modeling_gemma4.py` (scaling, softcap, norms, masque). Ne jamais inférer « pas de poids ⇒ pas d'op » (cf bug v_norm D.0→D.0b). Un oracle qui partage une hypothèse avec le code ZML testé donne un PASS trompeur.

## Mémoire associée

