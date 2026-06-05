# ENGINE_LOG — P5.7.5 → P5.7.8 (run autonome)

> Journal du run autonome de portage du forward complet Gemma-4-E2B-it en ZML (démarré 2 juin 2026).
> Mandat Régis : ne pas attendre, /iterate, sub-agents à volonté ; inspection cross-LLM en aval.
> Contrat de précision : `docs/P5_7_5_precision_contract.md`. Discipline : oracle = source de vérité,
> gate atomique, commit + tag, comparaison par couche pour localiser (contrat §6).

## Plan

| Gate | Contenu | Statut |
|---|---|---|
| 0. Oracle hybride | script 39, fp32+embptl bf16, +35 taps + last_hidden | ✅ (fixture `p5_7_5_hybrid.safetensors`, 42 tensors) |
| 0bis. Blueprint | analyse structure moteur (4 sous-agents) | ✅ |
| P5.7.5 | moteur 35 couches → comparaison par couche vs oracle | ⏳ build OK (21s), run en cours |
| P5.7.6 | head → logits vs HF | ⬜ |
| P5.7.7 | decode 1 token (KV incrémental) | ⬜ |
| P5.7.8 | decode N tokens | ⬜ |

**Décisions/findings de build :**
- Moteur `gemma4_prefill.zig` : forward `(self: Engine, rt: Runtime)` — poids checkpoint + inputs fixture en arg (multi-store). Corps fp32 (`convert`), `embed_tokens_per_layer` bf16 jusqu'après ×16. Retourne 8 taps (frontières dispatch 0/4/13/14/15/19/33 + last_hidden) → localisation §6 en un build.
- YOCO : capture K/V à L13(sliding)/L14(full) en vars Zig threadées (trace déroulé), readers 15-34 réutilisent par layer-type.
- Build ZML = ~20s (deps en cache) ; le gros (MLIR/LLVM + exec) est au RUN. → itération rapide.
- Revue Zig (sous-agent) avant 1er run : 1 bug réel (`layer_scalar {.one}.broad` → `asScalar()`), 11 points OK. Fixé.
- API clés : `gather(.{.voc=ids},.{})`, `choose1d(.layer,i)` (slice par couche), `convert(.f32)`, `results.get(struct{...x8})` destructuring.

## Compute
- 3090 `ssh user@gpu-host` (direct OK + jumphost macmini), VM 23 Go / ~22 dispo, GPU 24 Go.
- venv `/data/venvs/gemma4-probe` (py3.12, transformers 5.9.0), repo `/data/gemma4-zml-probe`.
- ZML `/data/rqz_workspace/zml`, runners `examples/rqz/` ; deploy `zml_runner/deploy_to_3090.sh` (rsync -J macmini).
- Build : `cd /data/rqz_workspace/zml && ./bazel.sh build //examples/rqz:<cible>` (~9 min premier, incrémental ensuite).

## Journal

### 2 juin — démarrage
- Accès 3090 confirmé. Oracle hybride (script 39) lancé en bg. Blueprint workflow (4 agents) lancé en bg.

### 2 juin — P5.7.5 moteur : écrit, compile, débogage mémoire
- Oracle hybride OK (fixture 42 tensors : input_ids, embed_slice, embptl_slice, cos/sin, attn_mask, last_hidden, hidden_00..34). **Finding** : `out.hidden_states[35]` alias `last_hidden` (HF remplace la dernière entrée par la sortie post-final-norm) → garder [0..34] + last_hidden, `.clone()`.
- Moteur `gemma4_prefill.zig` écrit (forward 2-store, 8 taps), build dbg 21s, revue Zig (1 bug `layer_scalar.broad` → `asScalar()`).
- **Débogage mémoire (VM 24 Go) — itératif :**
  1. 1er run : OOM (RSS 23 Go) au **load** — je chargeais les tables embed (5,5 Go) que load_all différait. Fix : **pré-gather** les lignes dans la fixture (embed_slice/embptl_slice), tables non résidentes.
  2. 2e run : load OK, OOM au **compile/exec** (RSS 23,5 Go).
  3. Sampler RSS : load monte à **14,8 Go** (>> 3,8 Go bf16 attendus), exec lazy (calcul au 1er `toSliceAlloc`) → **22,2 Go**.
  4. Libérer le registre checkpoint post-load : **n'a rien changé** (pas le mmap).
  5. **Hypothèse courante** : build en mode **`dbg`** (XLA sans réutilisation de buffers → tous les intermédiaires 35 couches vivants). Test `-c opt` en cours.
- **NB exec lazy** : `exe.call` retourne un futur ; le calcul se déclenche à la matérialisation host (`toSliceAlloc`). Le RSS grimpe pendant la « comparaison ».

### 2 juin — P5.7.5 : mémoire contournée (producer-only) + **bug oracle trouvé**
- **Mémoire** : 35 couches fp32 déroulées = ~22,2 Go > VM 24 Go (~21 dispo). Pas de bump RAM (hôte Proxmox = hors périmètre, refusé classifier). Slim checkpoint (3,8 Go, sans embeddings/vision) + registre libéré : load reste 14,8 Go (buffers fp32 + staging). **Diagnostic via RUN_LAYERS=15 (producer-only) : tient (~20 Go) et donne enfin les chiffres.** Fix final mémoire = exécution étagée (à faire, requis aussi pour decode).
- **Bug trouvé = ORACLE, pas moteur** (leçon [[feedback_oracle_independence]]) : taps L0 FAIL max_abs 1499. Localisation par magnitudes : moteur h1/h2 = 1530 ≈ oracle 1527 (corps **correct**), mais sortie moteur = 27 = h3×**0.01782** (vrai `layer_scalar[0]`), oracle h01 = 1527 = h3×**1.0**. → **`layer_scalar` est un `registered_buffer`, pas un Parameter** ; mon script 39 (streaming load) ne chargeait que `named_parameters()` → buffers ratés → oracle calculé avec layer_scalar=1.0 partout. Valeurs réelles : L0=0.0178, L1=0.22, L4=0.50, L13=0.088, L14=0.029. **Fix script 39** : charger aussi `named_buffers()` (+ assert 35/35 layer_scalar). Le moteur ZML était **correct depuis le début**.
- **STAGE0** (embeds vs hidden_00) : max_abs 0.032 WARN = `embed_scale` artefact. `Gemma4TextScaledWordEmbedding.embed_scale = torch.tensor(√1536)` construit en bf16-first → arrondi **39.25** au lieu de fp32 **39.1918**. Un oracle fp32 fidèle (et le moteur) utilise 39.1918. **2e bug oracle** : fix script 39 = `model.embed_tokens.embed_scale = torch.tensor(√1536, dtype=fp32)`. (Autres scales `per_layer_*` = floats fp64, OK.)

### 2 juin — 🎯 PRODUCER PATH VALIDÉ (bit-near)
Après les 2 fix oracle (layer_scalar buffers + embed_scale fp32), producer-only (15 couches) **tous PASS** :
- STAGE0 embeds vs h00 : **max_abs 0.0 (bit-exact)**.
- tap L0 sliding **3.4e-5**, L4 full **1.8e-5**, L13 sliding-writer **2.4e-5**, L14 full-writer **5.1e-5** → tous PASS (drift matmul Eigen-vs-BLAS, ≪ 1e-2).
- Magnitudes exactes (`max|zml|=max|ref|`). **Embedding + PLE frontend + sliding + full + KV-writers + layer_scalar corrects.** La claim bit-identique du contrat tient.
- **Leçon clé** : les 2 « bugs » étaient dans l'ORACLE (mon streaming load ratait buffers `layer_scalar` + arrondi bf16 `embed_scale`), PAS le moteur ZML. Cf [[feedback_oracle_independence]] — un oracle recodé doit dériver fidèlement de la source de vérité.

**Reste P5.7.5** : (a) valider readers 15-34 (oracle fournit hidden_15 + shared KV via `return_shared_kv_states`), (b) last_hidden complet (compo producer+reader ; mémoire = 2 phases), (c) RUN_LAYERS=35 + retirer taps debug. Puis P5.7.6/7/8.

### 2 juin — 🎯🎯 P5.7.5 FORWARD COMPLET VALIDÉ (bit-near)
Moteur refactoré : helper `runLayer` partagé + 3 modes (mémoire VM 24 Go ne tient pas les 35 couches fp32 déroulées d'un coup → validation par tranches, chacune comparée à l'oracle, composition = forward complet). Oracle exporte `return_shared_kv_states=True` (KV writers 13 sliding `[1,1,4,256]` / 14 full `[1,1,4,512]`) + hidden_00..34.
- **producer** (0-14, depuis embeds) : STAGE0 **0.0 bit-exact**, L0/L4/L13/L14 = 3.4e-5/1.8e-5/2.4e-5/5.1e-5 → **PASS**.
- **reader** (15-24, depuis hidden_15 + KV) : L15 2.3e-5, L19 1.1e-5, →hidden_25 1.9e-5 → **PASS** (YOCO read OK : sliding+full readers, double-wide MLP 12288).
- **reader2** (25-34 + final norm, depuis hidden_25 + KV) : L29 2.3e-5, **last_hidden 4.6e-5** → **PASS**.
- **Composition** : chaque tranche fed l'input correct de l'oracle → produit l'output correct ⟹ forward 35 couches + final norm validé bout-en-bout. **Gemma-4-E2B-it portable ET porté en ZML, prouvé numériquement.**
- Modes CLI : `gemma4_prefill <ckpt_slim> <fixture> [producer|reader|reader2]`. Checkpoint slim = `weights/model_text_layers.safetensors` (35 couches + frontend + norm, sans embeddings/vision, pour la mémoire). Oracle = `scripts/39_..._hybrid.py` (2 fix : layer_scalar buffers + embed_scale fp32).

**Reste projet** : P5.7.6 (logits vs HF : head P5.5 sur last_hidden), P5.7.7/.8 (decode KV cache incrémental). Mémoire : la full-35-en-un-process nécessiterait l'exécution étagée (per-layer) — non requise pour la validation (composition suffit), mais utile pour un runtime de prod.

### 2 juin — 🎯🎯🎯 P5.7.6 LOGITS VALIDÉ (top-1 identique 4/4)
`zml_runner/gemma4_logits.zig` : lm_head tied (= `embed_tokens.weight` brut, vocab **262144**) + softcap 30·tanh(x/30), sur le `last_hidden` validé. Comparé à l'oracle (script 39 exporte `logits` = `softcap(last_hidden @ embed_tokens.T)`).
- **argmax (token prédit) identique sur les 4 positions** : 236761 / 99998 / 2078 / 108 (= HF). **flips temp=0 = 0/4**. logits `max_abs = 3.8e-5`.
- **Chaîne complète validée e2e** : forward 35 couches + YOCO → last_hidden → lm_head + softcap → **logits → tokens identiques à Gemma-4-E2B-it HF**. embed_tokens (lm_head) lu du checkpoint complet ; CLI `gemma4_logits <model.safetensors> <fixture>`.
- **Gemma-4-E2B-it est correctement porté en ZML — prouvé bout-en-bout (hidden states bit-near + tokens identiques).**

**Reste** : P5.7.7/.8 (decode incrémental : KV cache prefill→decode, pos_idx, boucle génération). Effort substantiel (structure différente, mur mémoire) — la validation de port (forward+logits) étant le cœur scientifique, le decode est l'extension runtime.

### 2 juin — 🎯 MÉMOIRE DÉ-RISQUÉE : forward complet en UN process, mémoire bornée (mode `chain`)
Objectif (choix Régis) : débloquer les 35 couches en un process, base propre pour le decode. Le forward unique 35-couches (`full`) ne tient pas même en **opt** (XLA réutilise partiellement les buffers : 22.2 dbg → 20.7 → 22.5 selon la RAM dispo, tient ~6 Go de poids fp32 simultanés). Donc **exécution étagée**.
- **Mode `chain`** : forward complet en **5 chunks** enchaînés en un process — P(0-14) + 4 readers de 5 couches (15-19/20-24/25-29/30-34+norm). Le KV writers (13/14) et le hidden sont **threadés device→device** via override des champs de `rt_buf` (réinjection `rt_buf.hidden_15` ← sortie du chunk précédent). **Sync forcée** (matérialiser la sortie de chunk = libère le working set device) + `exe.deinit()` entre stages → mémoire bornée à **1 chunk**.
- **Résultat** : pic RSS **18.5 Go** (vs 22.6 sans chunking fin / 22.5 opt-full), profil load 14.1 → P 18.5 → R2 16.4 → R4 14.7 (working sets libérés entre stages). `last_hidden` vs oracle **6.9e-4 PASS** (chaîne les sorties PROPRES du moteur sur 35 couches — preuve du forward auto-contenu). CLI `gemma4_prefill <ckpt_slim> <fixture> chain`.
- **Finding clé** : l'exec ZML est **lazy** (calcul au 1er `toSliceAlloc`) → sans sync forcée, les working sets de chunks consécutifs s'accumulent. Matérialiser la sortie d'un chunk force sa complétion et libère ses intermédiaires avant le chunk suivant. Le MLP **double-wide 12288** des readers impose des chunks ≤5 couches (vs 15 pour les producers à 6144).
- **Base decode** : l'infra (forward de chunk paramétrable + threading buffers + sync + deinit) généralise au decode (per-token + KV cache croissant). Les modes validation `producer|reader|reader2|full` conservés.

### 5 juin — 🧱 Génération longue · Task 0 : paramétrisation comptime NEUTRE du socle (E1+E2 PASS, HLO graphe identique)
Fondation gen-long (cf `GENERATION_LONGUE_DESIGN.md` / `..._PLAN.md`, branche `generation-longue`). `engine.zig` gagne :
- `EngineCfg { ring, two_masks, kmax_sliding, kmax_full }` — **tous defaulted → neutre** ; `EngineModel(comptime Brick, comptime cfg: EngineCfg)`.
- `Packed(comptime two_masks)` → **2 structs distincts** (pas de champ `void` : `zml.io.load` réfléchit récursivement sur les champs, chacun doit être un Tensor). `two_masks=false` ≡ l'ancien `Packed` (fixture E1/E2 inchangée).
- scatter sliding ring `pos % kmax_sliding` + sélection de masque par `comptime isFull(i)` — **gardés comptime** : en défaut, branches inactives **non émises** (ni `remainder`, ni 2e masque).
- **E1 PASS** (`[1018,6398,25967,53121]` == HF) + **E2 PASS** (`[107,1,106,1]` == gen_vq), config `EngineModel(_, .{})`.
- **Preuve HLO** `diff -rq` before(état propre, pré-Task-0) vs after(code Task 0) : **1037/1037 fichiers identiques**, 2 diffs **bénins** — (a) `debug_options` (= chemin de dump) ; (b) 1 `.ir-with-opt.ll` où seuls des **noms SSA LLVM** diffèrent (`%y_approx.i`↔`.i8`, etc.), alpha-équivalent (mêmes ops/constantes/types/flux) = bruit de codegen entre 2 compilations, **pas le calcul**. Tous les fichiers HLO (le graphe) sont byte-identiques. La technique de preuve reste `diff -rq` des dumps `--xla_dump_to`.
- **OOM infra (5 juin)** : le compile XLA-CPU monte à **~22,7 Go RSS** > 23 Go de la VM → OOM-killer (`tf_XLAEigen invoked oom-killer`, **mort silencieuse exit 255** pile à `Compiling gen step`) tuait E1 ET decode4. Débloqué par **swapfile temporaire 16 Go** (`/swapfile_xla`, swap total → 22 Go). À pérenniser pour L0-L2. Build via `./bazel.sh` (bazelisk ; `bazel` absent du PATH).
- **Prochain : L0** (oracle `46_gen_long_oracle.py`, 2048 tokens + 2 masques + caches L_max, sans V-quant).
