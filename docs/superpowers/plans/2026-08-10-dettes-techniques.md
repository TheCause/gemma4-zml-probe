# Plan d'implémentation — Solde des dettes techniques actionnables

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal :** Solder les dettes techniques actionnables du repo (README bilingue, K7, D11,
PLANNING périmé) et, sur GO explicite de Régis, les deux dettes coûteuses (D1/D2 couverture
GPU des warpers, K8 mesure à froid) — sans toucher au graphe ni aux claims d'équivalence.

**Architecture :** Aucun changement de graphe. Tout est host-side (Zig `zml_runner/`), doc
(`README.md`, `PLANNING.md`, `docs/`), ou mesure sur la VM 3090. Chaque correctif suit la
convention du repo : refus/comportement **VU** avant et après, gate committé, clôture
documentaire selon la checklist §5.4 de `docs/DOCUMENTATION.md`.

**Tech stack :** Zig 0.16-dev (build via `zml_runner/build_3090.sh` — JAMAIS `-c opt` seul,
cf `docs/MODE_BUILD_AUDIT.md`), Bazel + ZML sur la VM 3090 (`ssh ia@192.168.1.163`,
workspace `/data/rqz_workspace/zml`, repo `/data/gemma4-zml-probe/`), Python venv
`/data/venvs/gemma4-probe`.

---

## Contexte pour une session neuve (lis ceci d'abord)

Ce repo est un portage ZML bit-exact de Gemma 4 (E2B + 12B) prouvé gate par gate (~70 gates
taggés). Règles non négociables du repo :

1. **Le graphe ne bouge pas** sans témoin HLO : tout chantier qui touche `zml_runner/` doit
   prouver `md5 HLO identique` au témoin pré-code (pattern DC0/GC0/S2-G) — ce plan ne touche
   que du code host, mais le gate de non-régression reste dû.
2. **Un refus/une claim se VOIT échouer** : jamais « le test devrait passer » — on exerce le
   cas, on archive la sortie (`docs/evidence/`).
3. **Mode de build prouvé** : toute mesure part de `zml_runner/build_3090.sh` et vérifie la
   bannière `BUILD: mode=ReleaseFast` dans le log (leçon : deux mesures « opt » concordantes
   étaient toutes deux du debug).
4. **Interdit D10 gardé** : `ALLOC-LOOP: alloc=0` par step, compteur toujours actif — tout
   run le re-vérifie gratuitement.
5. **VRAM** : `nvidia-smi --query-compute-apps` avant tout run GPU ; un Ollama résident peut
   occuper la carte (`ollama stop <modèle>`, réversible).
6. **Anonymisation** : aucun chemin perso (`/Users/regis`, hostnames) dans un commit — grep
   avant push (`git grep -nE 'Users/regis|macmini|192\.168' -- ':!docs/superpowers'`).
7. **Branche + PR** : jamais de commit direct sur `main`. Branche `dettes-techniques-aout`,
   merge en `--no-ff` après GO Régis.

Sources de vérité à lire avant d'exécuter : `PLANNING.md` (tête), `docs/KVDUMP_RESULTS.md`
§6 (dettes K), `docs/D10_RESULTS.md` §6 (dettes DA), `docs/MODE_BUILD_AUDIT.md`,
`docs/SAMPLING_RESULTS.md` (gates S2, dettes D1/D2).

**Invocations réelles (build + runner 12B)** — reprises du plan exécuté
`docs/superpowers/plans/2026-08-09-kv-cache-dump-restore.md` (lignes ~57-68) :

```bash
# Build (sur la VM) — les DEUX flags de mode, jamais -c opt seul :
cd /data/rqz_workspace/zml && ./bazel.sh build -c opt \
  --@rules_zig//zig/settings:mode=release_fast --@zml//platforms:cuda=true \
  //examples/rqz:gemma4_g12auto //examples/rqz:gemma4_g12a4k
# Runner (sur la VM) — chemins relatifs au workspace : cd /data/rqz_workspace/zml d'abord
B1=./bazel-bin/examples/rqz/gemma4_g12auto     # variante 1280
B4=./bazel-bin/examples/rqz/gemma4_g12a4k      # variante 4k
W=/data/gemma4-zml-probe/weights_12b
$B1 $W/model.safetensors $W/tokenizer.json <flags...>
```

Alternative équivalente depuis M1 : `ZML_REMOTE=ia@192.168.1.163
ZML_WS=/data/rqz_workspace/zml ./zml_runner/build_3090.sh` (script = source unique des
flags ; cibles via la variable `TARGETS`, pas d'argument positionnel).

## Inventaire des dettes (état au 10 août 2026)

### A — Traitées par CE plan (aucune décision nouvelle requise)

| Dette | Quoi | Tâche |
|---|---|---|
| **README bilingue** | Corps en anglais, 1ᵉʳ bandeau en français (« Portée de la claim == HF ») ; dette signalée par Régis le 10 août : « choisir UNE langue et s'y tenir » | Task 3 (langue tranchée en Task 0) |
| **K7** | Le refus « fichier tronqué » (`KvDumpTruncated`, 7 sites de `return` dans `zml_runner/kvdump.zig` ; la 8ᵉ occurrence l.49 est la déclaration de l'error set) n'émet aucun `log.err` avant l'erreur — cosmétique mais contraire au standard « refus bruyant » du repo | Task 2 |
| **D11 (résidu)** | `PLANNING.md` garde une case ouverte « `70_u8_corrupt.py` doit passer `--gen-config <fichier>` » alors qu'un correctif du 29 juil est coché plus haut — état réel à instruire, puis solder (code OU case) | Task 1 |
| **PLANNING périmé** | 2 cases 🔴 « NOUVEAU FRONT » du 30 juil sont soldées ou bornées depuis : l'audit mode ambigu est FAIT (`docs/MODE_BUILD_AUDIT.md`, 174 claims, paniers A/B) ; DA-1 est BORNÉE (`D10_RESULTS.md` §6 : < ~450 mallocs C/step, LD_PRELOAD différentiel du 30 juil) | Task 1 |

### B — Traitées SUR GO EXPLICITE de Régis (coût GPU / accès root)

| Dette | Quoi | Tâche |
|---|---|---|
| **D1** | `applyTopP` (`zml_runner/sampling.zig:189`) n'a aucune couverture GPU — seule couverture : fixture host `S2-U` (`scripts/72_sampling_fixture.py`) | Task 4 |
| **D2** | `applyTemperature` (`sampling.zig:112`) non exercé de bout en bout | Task 4 |
| **K8** | Le `0,898 s` de DC7(ii) est une lecture À CHAUD (cache de pages). À froid le gain resterait ≥ ×130 — non mesuré, `drop_caches` exige root sur la VM | Task 5 |

### C — HORS périmètre (écrites, pas tues — ne PAS les traiter ici)

| Dette | Pourquoi hors périmètre |
|---|---|
| **K1** (PRNG non sérialisé → sampling armé + dump = refus) | Décision Régis ACTÉE le 9 août — la « traiter » serait la rouvrir |
| **K2 / DA-4 résiduelle** (8k non exercé) | Décision Régis ACTÉE (gates 1280+4k seulement) |
| **K3 / périmètre E2B** (logits hors graphe, pas de `suppress_tokens`) | Chantier propre ; la claim est écrite comme fausse pour E2B, c'est voulu |
| **K4** (`--repl` + dump/load refusés) | Sémantique multi-tour inexistante — futur chantier « résident à reprise » |
| **K5** (reprise avec prompt neuf) | C'est un prefill partiel : son propre chantier avec sa propre spec |
| **K6** (compression bf16 du dump) | « On ne dégrade pas un état exact pour du disque » — décision de design |
| **DA-6** (`std.Io.Threaded` sur gpa non wrappé) | Structurelle, contrepartie documentée (`D10_RESULTS.md` §6) : compteurs mono-thread corrects, bornée par AL-RSS |
| **Phase 1 repetition penalty** (SUSPENDUE) | Plan dédié déjà écrit et revu : `docs/superpowers/plans/2026-07-27-sampling-repetition-penalty.md` (spec rév. 4 du 27 juil) — se relance tel quel, pas ici |
| **Triton paged attention** (option) | Chantier majeur (bump ZML + refonte cache YOCO) |
| **Options B/C anonymisation** (purge historique `main` ; sortir les plans superpowers du repo public) | Décision Régis en attente depuis le 26 juil — irréversible, ne rien faire sans lui |
| **10 branches distantes mergées** (`batching`, `repl-mode`…) | Leur suppression en masse a été bloquée par le classificateur de permissions — action manuelle Régis (`git push origin --delete <branche>`) |

---

### Task 0 : Cadrage avec Régis + état des lieux

**Files :** aucun (lecture seule).

- [ ] **Step 1 : Vérifier l'état du repo**

Run : `git -C ~/dev/gemma4-zml-probe status --porcelain && git log --oneline -3`
Expected : arbre propre (hors ce plan), HEAD = `bfaeb32` ou plus récent sur `main`.
Si l'arbre n'est pas propre : STOP, montrer à Régis (leçon : un travail non committé est
invisible — ne pas l'écraser).

- [ ] **Step 2 : Poser les 3 décisions à Régis, AVANT tout code**

1. **Langue du README** — recommandation : **anglais intégral** (le corps est déjà ~90 %
   anglais, README = vitrine GitHub publique, la doc détaillée reste en français et le README
   l'annonce déjà « in French »). Alternative : français intégral (cohérent avec `docs/`,
   mais ~300 lignes à traduire et audience GitHub réduite). NEEDS_DECISION.
2. **GO/NO-GO Task 4 (D1/D2)** — coût : ~1 h de session dont ~2 runs GPU 12B + 1 build.
3. **GO/NO-GO Task 5 (K8)** — exige un accès root sur la VM (ou via l'hôte Proxmox `prox`)
   pour `sync && echo 3 > /proc/sys/vm/drop_caches`. Coût : ~30 min, 2 runs restore.

- [ ] **Step 3 : Créer la branche**

```bash
git -C ~/dev/gemma4-zml-probe switch -c dettes-techniques-aout
```

---

### Task 1 : Instruire D11 + rafraîchir PLANNING.md

**Files :**
- Lire : `scripts/70_u8_corrupt.py`, `docs/GENERATION_CONFIG_RESULTS.md`
- Modifier : `PLANNING.md` (section « Sampling phase 2 » et cases 🔴 du 30 juil)

- [ ] **Step 1 : Établir l'état réel de D11**

Run : `grep -n "gen.config\|gen_config" scripts/70_u8_corrupt.py`
Deux issues possibles :
- Le script passe déjà `--gen-config` (ou dépose `generation_config.json` à côté de son
  checkpoint — c'est le correctif coché du 29 juil, `PLANNING.md:99`) → D11 est SOLDÉE,
  la case ouverte `PLANNING.md:143` est un doublon périmé.
- Il ne le fait pas → appliquer le correctif minimal : ajouter l'argument
  `--gen-config <dq>/generation_config.json` (un chemin de **fichier**, pas un répertoire)
  à l'invocation du runner dans le script, et **exercer le script** sur la VM pour VOIR
  qu'il passe (c'est un gate historique : le laisser mort en silence est exactement la
  dette).

- [ ] **Step 2 : Rafraîchir PLANNING.md — 3 corrections factuelles**

Dans `PLANNING.md`, section « ⏳ Sampling phase 2 » (lignes ~127-146) :
1. Case 🔴 « mesures passées en mode ambigu » (`:127`) → passer `[x]` avec :
   « SOLDÉ (30 juil) : audit complet `docs/MODE_BUILD_AUDIT.md` — 174 claims, panier A
   re-mesuré par D10, panier B marqué sans re-mesure (décision documentée §5). »
2. Case 🔴 « dettes DA-1/DA-6 » (`:131`) → passer `[x]` pour DA-1 avec :
   « DA-1 BORNÉE (30 juil, `D10_RESULTS.md` §6) : LD_PRELOAD différentiel,
   < ~450 mallocs C/step. DA-6 reste ouverte (structurelle, contrepartie documentée). »
   La 8k (DA-4 résiduelle) reste écrite comme dette assumée (= K2).
3. Case D11 (`:143`) → verdict du Step 1 (soldée + référence, ou correctif appliqué).

Ajouter en tête de PLANNING une section courte « Dettes ouvertes (10 août 2026) » qui
pointe vers les 3 tables de ce plan — une seule source de vérité, plus de doublons épars.

- [ ] **Step 3 : Commit**

```bash
git add PLANNING.md scripts/70_u8_corrupt.py
git commit -m "dettes(PLANNING+D11) : fronts du 30 juil soldés/bornés, D11 instruit, inventaire dettes unifié"
```

---

### Task 2 : K7 — refus « tronqué » bruyant dans kvdump.zig

**Files :**
- Modifier : `zml_runner/kvdump.zig` (7 sites `return ReadError.KvDumpTruncated` : lignes
  101, 108, 109, 178, 179, 199, 205 — relire le fichier d'abord, les lignes ont pu bouger)

Convention visée : même standard que les autres refus DC5 — un `log.err` nommant le fichier,
ce qui était attendu et ce qui a été lu, AVANT de retourner l'erreur.

- [ ] **Step 1 : VOIR le refus actuel (silencieux)**

Les dumps de gates ont été supprimés de la VM le 10 août (il reste ~7,6 Mo de petits
fichiers). Régénérer un dump 1280 (~840 Mo, sur `/data`, jamais `/`), le tronquer,
l'offrir au runner :

```bash
# sur la VM, avec $B1 et $W définis comme en tête de plan :
$B1 $W/model.safetensors $W/tokenizer.json --prompt "test K7" --max-tokens 8 \
  --dump-cache /data/gemma4-zml-probe/kvdump/k7.kvdump
head -c 100000 /data/gemma4-zml-probe/kvdump/k7.kvdump > /data/gemma4-zml-probe/kvdump/k7_trunc.kvdump
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /data/gemma4-zml-probe/kvdump/k7_trunc.kvdump \
  --max-tokens 8 > /tmp/k7_avant.out.log 2> /tmp/k7_avant.err.log
```

(Flags exacts : vérifier dans `docs/KVDUMP_RESULTS.md` §2/§4 ou `--help` — ne pas deviner.)
**Archiver la sortie** : l'erreur `KvDumpTruncated` sort sans message contextuel.
C'est l'état AVANT.

- [ ] **Step 2 : Implémenter le helper + les 7 sites**

Dans `kvdump.zig`, ajouter (adapter le nom du logger au scope existant du fichier) :

```zig
fn failTruncated(path: []const u8, what: []const u8, want: u64, got: u64) ReadError {
    std.log.err("kvdump: fichier tronqué '{s}' — {s} : attendu {d} octets, lu {d}", .{ path, what, want, got });
    return ReadError.KvDumpTruncated;
}
```

et remplacer chaque `return ReadError.KvDumpTruncated;` par un appel avec le contexte du
site (`"préfixe header"`, `"header JSON"`, `"tenseur <nom>"`, …). Contrainte : ces chemins
sont hors boucle de step, donc AUCUN risque D10 — mais ne pas allouer quand même (format
direct, pas de `allocPrint`).

- [ ] **Step 3 : Rebuild au mode prouvé + re-VOIR le refus**

Après avoir déployé le `kvdump.zig` modifié sur la VM (`zml_runner/deploy_to_3090.sh`,
avec `ZML_REMOTE`/`ZML_DST` renseignés — les défauts sont des placeholders qui échouent
en silence, piège documenté au PLANNING) :

```bash
# depuis M1, à la racine du repo :
ZML_REMOTE=ia@192.168.1.163 ZML_WS=/data/rqz_workspace/zml ./zml_runner/build_3090.sh
```

(le script est la source unique des deux flags de mode ; ses cibles par défaut incluent
`gemma4_g12auto` — pas d'argument positionnel). Vérifier la bannière `BUILD:
mode=ReleaseFast` au log du run suivant.
Relancer le `--load-cache` tronqué du Step 1. Expected : le message `kvdump: fichier
tronqué…` apparaît AVANT l'erreur, avec chemin + octets. Archiver AVANT/APRÈS dans
`docs/evidence/kvdump/k7_refus_bruyant.txt`.

- [ ] **Step 4 : Non-régression — un load SAIN passe encore**

Run `--load-cache` sur le dump complet du Step 1 : la continuation démarre, bannière
`BUILD: mode=ReleaseFast` et `ALLOC-LOOP: alloc=0` présents au log (l'interdit D10 se
re-vérifie gratuitement). Exercer le cas sain est obligatoire (leçon launchd : un refus qui
marche ne prouve pas que le nominal marche encore).

- [ ] **Step 5 : Commit**

```bash
git add zml_runner/kvdump.zig docs/evidence/kvdump/k7_refus_bruyant.txt
git commit -m "dettes(K7) : refus 'tronqué' bruyant — log.err contextuel aux 7 sites, vu AVANT/APRÈS, load sain re-exercé"
```

---

### Task 3 : README — une seule langue

**Files :**
- Modifier : `README.md`

Pré-requis : décision de langue prise en Task 0 (recommandation : anglais).

- [ ] **Step 1 : Inventorier le français résiduel**

Run : `grep -nE "« |ç|é|è|à |’" README.md`
Expected (si cible = anglais) : essentiellement le 1ᵉʳ bandeau (« ⚠ Portée de la claim
"== HF" », lignes ~3-9) + éventuels fragments épars. Lister CHAQUE occurrence — pas
d'échantillonnage.

- [ ] **Step 2 : Traduire sans affaiblir**

Traduire le bandeau en anglais en conservant EXACTEMENT la portée de la claim (c'est un
texte de nuance issu du gate GC11 — le sens juridique de chaque mot compte) :

```markdown
> **⚠ Scope of the "== HF" claim** (nuance pass, `generation_config` work, 29 Jul 2026).
> Throughout this document, "== HF" means **same argmax on the raw logits** — a stricter
> criterion than comparing two `generate()` calls, but **not** the same statement. Until
> 29 Jul the port did **not** apply `generation_config.json` (`suppress_tokens`, multiple
> EOS): the reading "reproduces what `generate()` would produce" was **false**. It became
> true **for the 12B in free-running mode** and remains **false for the E2B runners**.
> Details and figures: `docs/GENERATION_CONFIG_RESULTS.md` · `docs/FINDING_GENERATION_CONFIG.md`.
```

Garder les liens vers la doc française avec la mention existante « (in French) ».

- [ ] **Step 3 : GC11 — FAIL attendu, puis marqueur bilingue, puis gate re-mordant**

`scripts/gc11_claim_scope.sh` exige la **présence** du marqueur littéral français
`MARQUEUR="argmax sur les logits bruts"` (l.~20) dans tout document de `CIBLES` contenant
« == HF » — et `README.md` est en `CIBLES` (l.~22). Après traduction, le README devient
« NU » au sens du gate : **le FAIL est l'état AVANT attendu** — l'exercer et l'archiver
(c'est le gate qui mord, pas un bug).

Puis étendre le marqueur en alternative bilingue (OU logique, deux `grep -qF` : le
français existant OU `"same argmax on the raw logits"`), à portée strictement constante.
Re-exercer le gate : PASS sur le repo. Contre-preuve : `--self-test` du script (et si le
self-test n'a qu'un canary français, ajouter un canary nu **anglais** pour voir le gate
mordre aussi dans la nouvelle langue).

- [ ] **Step 4 : Contre-preuve + relecture là où l'humain regarde**

Run : `grep -cE "« |ç|é|è|à |’" README.md` → Expected : 0 (hors noms propres/liens).
Puis OUVRIR le rendu (aperçu markdown) et relire en entier — le README est l'endroit où
l'humain regarde (leçon du 9 août : vérifier LÀ, pas dans le diff).

- [ ] **Step 5 : Commit**

```bash
git add README.md scripts/gc11_claim_scope.sh
git commit -m "dettes(README) : une seule langue (EN) — bandeau claim-scope traduit à portée constante, GC11 vert"
```

---

### Task 4 : D1/D2 — couverture GPU de applyTopP et applyTemperature (GO Régis requis)

**Files :**
- Lire d'abord : `zml_runner/sampling.zig` (112, 122, 189), `zml_runner/gemma4_g12auto.zig`
  (~1054 et ~2873 : les 2 sites d'armement), `docs/SAMPLING_RESULTS.md` (gates S2-U,
  S2-PONT et le ⚠ « ne pas mal lire les compteurs d'un run ARMÉ »), `scripts/72_sampling_fixture.py`
- Créer : `scripts/75_d1d2_gpu_bridge.py` (dépouillement), extension pont dans
  `gemma4_g12auto.zig`
- Preuves : `docs/evidence/kvdump/../d1d2/` (créer `docs/evidence/d1d2/`)

**Méthode (pattern S2-PONT, étendu au régime armé)** : la seule façon insensible à la
bistabilité est de comparer DEUX implémentations sur le MÊME vecteur de logits, au MÊME
step, dans le MÊME processus. S2-PONT l'a fait pour `applyTopK` en régime neutre ; D1/D2 =
le refaire avec `top_p` et `temperature` actifs, contre un sélecteur de référence
indépendant (réimplémentation naïve documentée face aux pièges HF : tri **ascendant** de
`top_p`, ex æquo conservés par `top_k`, `torch.sort` instable au-delà de n=128 — cf
`docs/SAMPLING_RESULTS.md`).

- [ ] **Step 1 : Écrire la mini-spec (½ page) et la faire valider par Régis**

Elle fixe, AVANT tout code, les 3 gates et leurs prédictions falsifiables :
- **G-D1** : sur ≥ 300 steps GPU réels armés (`top_k=64, top_p=0.95, T=1.0`), l'ensemble
  des survivants de `applyTopP` == celui du sélecteur de référence sur le même vecteur,
  0 désaccord. **Antécédent non vide exigé** : compter les steps où top_p a réellement
  retranché ≥ 1 token après top_k (sinon le gate est passé À VIDE — leçon vacuité de
  l'antécédent) ; si 0 sur le prompt choisi, changer de prompt/température.
- **G-D2** : avec `T=0.7`, les logits post-`applyTemperature` == référence à tolérance
  bit-exacte (division f32 pure), ET le run de bout en bout produit une trajectoire qui
  diffère du run `T=1.0` (non-vacuité : la température a un effet observable).
- **G-D0** : md5 HLO identique au témoin pré-code (le pont est host-side, le graphe ne
  bouge pas) + `ALLOC-LOOP: alloc=0` inchangé — le pont doit compter/comparer SANS allouer
  dans la boucle (pré-réserver, pattern C5 de D10).

- [ ] **Step 2 : Capturer le témoin HLO AVANT la première ligne de code**

Même procédure que DC0/GC0 — commande inline (le gold est le **pré-opt**, cf
`docs/superpowers/plans/2026-07-27-sampling-repetition-penalty.md` l.104/154) :

```bash
# sur la VM, AVANT la première ligne de code du pont :
XLA_FLAGS="--xla_dump_to=/data/gemma4-zml-probe/d1d2_hlo_witness" \
  $B1 $W/model.safetensors $W/tokenizer.json --prompt "witness" --max-tokens 4
md5sum /data/gemma4-zml-probe/d1d2_hlo_witness/module_0001.*.before_optimizations.txt
```

Md5 archivé dans `docs/evidence/d1d2/hlo_witness.md5`.

- [ ] **Step 3 : Implémenter le pont armé** (extension du mécanisme S2-PONT existant dans
  `gemma4_g12auto.zig` — réutiliser son échafaudage, ne pas dupliquer), rebuild via
  `build_3090.sh`, run GPU, dépouiller avec `scripts/75_d1d2_gpu_bridge.py`.
  (75 est le premier numéro libre — le VÉRIFIER par `ls scripts/` : la collision de numéro
  a déjà été payée une fois, script 71.)

- [ ] **Step 4 : Juger les 3 gates, archiver, tagger**

Expected : G-D0/G-D1/G-D2 PASS → tags `gate/d1-gpu-pass`, `gate/d2-gpu-pass`. En cas de
FAIL : c'est peut-être un vrai bug de `applyTopP` jamais exercé sur GPU — le traiter en
finding, pas en obstacle.

- [ ] **Step 5 : Mettre à jour `docs/SAMPLING_RESULTS.md`** (D1/D2 passent de « déclarée,
  pas tue » à soldées, avec chiffres) **+ commit + retirer D1/D2 de l'inventaire du
  PLANNING**.

```bash
git add zml_runner/gemma4_g12auto.zig scripts/75_d1d2_gpu_bridge.py docs/SAMPLING_RESULTS.md docs/evidence/d1d2/ PLANNING.md
git commit -m "dettes(D1/D2) : applyTopP+applyTemperature couverts sur GPU — pont armé in-process, 3 gates verts, graphe intouché"
```

---

### Task 5 : K8 — le ×500 mesuré à froid (GO Régis + root requis)

**Files :**
- Modifier : `docs/KVDUMP_RESULTS.md` (§3 perf + §6 dette K8)
- Preuves : `docs/evidence/kvdump/k8_cold_read.txt`

- [ ] **Step 1 : Pré-enregistrer la prédiction, AVANT de mesurer**

Écrire dans `docs/evidence/kvdump/k8_prediction.md` (et committer) : « restore 4k à froid
prédit entre 3 s et 14,9 s (2,62 GiB à ~0,2-1 GiB/s disque VM) ⇒ **gain ≥ ×30 sur toute
la plage** (449,485/14,98 s = ×30 exactement : la claim C-D publiée reste vraie à froid
tant que restore ≤ 14,98 s) ; le "≥ ×130" écrit au §6 de
`KVDUMP_RESULTS.md` ne tient que si le restore à froid ≤ ~3,5 s et **devra être requalifié
sinon** ; ce qui tue la claim : un restore à froid > 34,6 s (= gain < ×13). »
(Leçon : falsifiable AUSSI en ingénierie — prédiction ET ce qui la tue, committées AVANT
de mesurer.)

- [ ] **Step 2 : Régénérer le dump 4k** (supprimé le 10 août) : re-run DC7(i) — fixture
  oracle 3 927 positions, `--dump-cache /data/gemma4-zml-probe/kvdump/k8_4k.kvdump`
  (~2,62 GiB ; vérifier l'espace : `df -h /data`, la VM avait 136 G libres).

- [ ] **Step 3 : Mesurer à froid**

```bash
# en root sur la VM (ou depuis l'hôte prox si sudo absent) :
sync && echo 3 > /proc/sys/vm/drop_caches
```

puis run `--load-cache` immédiat, chrono KVLOAD-PERF (fenêtre post-compile, comme DC7).
Faire 1 mesure à froid + 1 à chaud de contrôle (elle doit retrouver ~0,9 s — sinon
l'instrument a changé, STOP et diff l'instrument avant toute requalification).

- [ ] **Step 4 : Publier + nettoyer**

Mettre à jour `docs/KVDUMP_RESULTS.md` : le chiffre à froid À CÔTÉ du 0,898 s (jamais en
remplacement — les deux régimes sont vrais), K8 marquée soldée. Supprimer le dump K8 de la
VM, `df` AVANT/APRÈS (leçon : supprimer n'est pas libérer). Commit.

```bash
git add docs/KVDUMP_RESULTS.md docs/evidence/kvdump/
git commit -m "dettes(K8) : restore 4k mesuré à froid — prédiction pré-enregistrée, chaud re-contrôlé, dump nettoyé (df avant/après)"
```

---

### Task 6 : Clôture

- [ ] **Step 1 : Checklist de clôture §5.4 de `docs/DOCUMENTATION.md`** — la dérouler
  telle quelle (README, DOCUMENTATION, PLANNING, mémoire). Les dettes soldées disparaissent
  des tables « ouvertes » mais restent dans les docs de résultats avec leur date de solde.
- [ ] **Step 2 : Grep anonymisation** (commande en tête de plan) → 0 occurrence nouvelle.
- [ ] **Step 3 : Push branche + PR vers `main`**, titre « dettes : README EN, K7 bruyant,
  D11/PLANNING soldés [+ D1/D2, K8 si GO] ». Merge `--no-ff` sur GO Régis uniquement.
- [ ] **Step 4 : Mettre à jour la mémoire** (`project_gemma4_zml_probe.md` + PLANNING) :
  état des dettes après solde, avec dates.
