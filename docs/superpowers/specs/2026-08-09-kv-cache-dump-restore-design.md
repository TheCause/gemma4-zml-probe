# Spec — Dump/restore du KV-cache 12B (sauvegarder l'état, le réimplanter à la demande)

> **Date** : 2026-08-09 · **Niveau de travail** : standard (décision Régis, 9 août) ·
> **Statut** : **rév. 3** — deuxième passe de revue (8 findings sur les corrections, dont 4
> majeurs) : la fenêtre `KVLOAD-PERF` incluait la COMPILE (contradiction avec DC7(ii)) →
> load en 2 phases, validations avant compile / lecture+chrono après ; la fixture oracle
> fabriquée butait sur la garde `positions[0] == ids.len` (`:1767-1770`) → paramètre
> `positions0` ; les gardes DUPLIQUÉES de `run()` (`:1788`, `:1793`) n'étaient pas traitées.
> Première passe : rév. 2 — 19 findings dont
> 3 BLOQUANTS, tous traités) : le dump polluait la fenêtre ALLOC-LOOP telle que loguée (DC6
> auto-saboté) ; tout `--load-cache` mourait à la garde « --prompt requis » `:1626-1630` ;
> DC7 visait 3 900 tokens en mode libre alors que l'arrêt EOS l'aurait coupé bien avant
> (antécédent irréalisable → bascule sur `--oracle`, qui désactive l'EOS par décision Régis
> du 28 juil) ; `--prompt-ids` n'existe pas (refus à antécédent irréalisable retiré) ;
> l'instrument de DC7 excluait la lecture des 2,6 GiB de la mesure (biais vers le PASS).
> Rédigée sur ancrages code au HEAD `dbf6c21`, AVANT toute mesure.
> Les claims §2bis sont pré-enregistrées : ce fichier est committé avant la première ligne de
> code et avant le premier run. Le `git log` fait foi.
> **Demande Régis (9 août, verbatim)** : « est-ce qu'il est possible de sauvegarder le contenu
> d'un KV-Cache et le "ré-implanter" à la demande ? » puis « Ouvre le chantier dump/restore du
> KV-cache, fais la planification et rédige les contrats directement dans le projet ».
> **Exécution prévue** : session dédiée (Opus), plan
> `docs/superpowers/plans/2026-08-09-kv-cache-dump-restore.md`. En cas d'écart plan/spec,
> **LA SPEC FAIT FOI**.
>
> **✅ DEUX DÉCISIONS TRANCHÉES — GO Régis, 9 août 2026 (« Go pour les propositions par
> défaut ») :**
> 1. **Sampling armé + dump = refus bruyant** (`error.DumpWithSamplingArmed`). La
>    sérialisation de l'état du PRNG Xoshiro256 (32 octets) est une **dette documentée** —
>    elle ajouterait une claim d'équivalence stochastique qu'aucun gate simple ne prouve.
> 2. **Gates sur 1280 + 4k seulement** ; le code est le même `comptime` pour `g12a8k` mais
>    aucun gate ne l'exerce (**dette écrite**, comme DA-4).
>
> La session d'implémentation applique ces décisions telles quelles — plus rien n'est ouvert.

---

## 1. Contexte et faits établis (tous ancrés, HEAD `dbf6c21`)

### 1.1 Ce qu'est l'état d'une génération en cours

Le runner 12B (`gemma4_g12auto.zig`, variantes comptime `L_MAX` ∈ {1280, 4096, 8192}) tient
l'état d'une génération dans **exactement six choses** :

| # | État | Où il vit | Ancrage |
|---|---|---|---|
| E1 | 4 caches KV `sl_k/sl_v/fl_k/fl_v` (f32) | device, buffers donnés/rebindés chaque step | `:2163-2169` (création), `:1533-1537` (donation) |
| E2 | `step` — position absolue du prochain call | host, compteur de boucle | `:2220` |
| E3 | `fed` — le token à feeder au prochain step | host | `:2219`, `:2426` |
| E4 | la séquence des tokens déjà feedés (prompt + générés) | host, `ids` + `generated` | `:2146`, `:2172` |
| E5 | la politique `GenCfg` (suppress/EOS) | re-dérivée des fichiers à chaque lancement | `gencfg.zig` |
| E6 | l'état du PRNG si sampling armé | host, `scfg.prng` (Xoshiro256) | `sampling.zig:62` |

E5 se reconstruit à l'identique au prochain lancement (même `generation_config.json`).
E1-E4 sont **le** contenu à persister. E6 : décision ouverte n°1 (proposé : hors périmètre v1,
refus bruyant).

**Fait structurel qui rend le restore exact** : le cache sliding est **LINÉAIRE** `.k=L_MAX`
(R10, commentaire `:9-13`) — le scatter écrit à la position absolue `step`, pas dans un ring.
Un cache dumpé à la position N se réimplante donc sans transposition, tant que `L_MAX` est le
même. C'est aussi pour ça que le chantier **exige** l'égalité de variante (§4.5).

### 1.2 Les deux moitiés existent déjà à moitié

- **Restore (h2d)** : les caches naissent host-side (`HostInputs.init`, `[]u8` memset 0,
  `:505-521`) puis montent par `zml.Buffer.fromBytes` (`:2164-2168`). Réimplanter = remplir ces
  mêmes slices depuis un fichier au lieu de `@memset(0)`. **Le graphe ne voit aucune différence.**
- **Dump (d2h)** : `toSlice` est le mécanisme D2H prouvé du projet (chemin B, D10/C2) ; les
  caches finaux sont vivants dans `cache_buf` après la boucle, jusqu'aux `deinit` `:2436-2439`.
- **Écriture safetensors depuis Zig** : `writeIdsSafetensors` (`:1547`) écrit déjà le format
  (header JSON + u64 LE + données brutes) — à généraliser, pas à inventer.

### 1.3 Ce que le dump/restore vaut économiquement (fondement de C-D)

Le prefill de ce moteur est **par-decode** (1 position = 1 step). Mesures existantes :
**8,7 tok/s** à 4k (gate D2, `docs/CACHE_DONATION_RESULTS.md`), 9,6-9,7 tok/s à 1280
(`docs/MODE_BUILD_AUDIT.md`). Re-préfixer 4041 positions ≈ **464 s**. Relire ~2,6 GiB depuis
le NVMe de la VM et les monter en h2d se compte en secondes. Le restore doit gagner
**au moins un ordre de grandeur** — sinon le chantier n'a pas d'objet (C-D le rend falsifiable).

### 1.4 Le piège connu qui contraint les gates : la bistabilité inter-process

`docs/FINDING_NONDETERMINISME_TRAJECTOIRE.md` : deux runs du **même binaire** peuvent diverger
(autotune XLA-GPU) aux positions de marge fine (@47, marge 0,004587). Conséquence de design :

- l'équivalence **bit-exacte** ne peut être exigée qu'**intra-process** (même exécutable
  compilé/autotuné) → gate DC2 ;
- l'équivalence **inter-process** (le vrai cas d'usage : dumper, quitter, relancer) se juge
  **aux positions dont la marge dépasse le seuil hérité 1,873e-3** (2× le bruit U7, méthode de
  la spec generation-config §2bis) → gate DC3.

---

## 2. Objectif et critères de succès

1. `--dump-cache <fichier>` : en fin de `generateOnce` (tout mode sauf exclusions §3), l'état
   E1-E4 est écrit dans **un** fichier safetensors auto-décrivant (manifest §4.4), relisible
   par Python (`safetensors` standard) ET par le runner.
2. `--load-cache <fichier>` : le runner repart de cet état **sans re-prefill** et continue —
   génération libre (`--max-tokens`) ou teacher-forcée (`--oracle`).
3. **Le graphe ne bouge pas** : HLO pré-optimisation byte-identique au témoin (DC0).
4. **L'interdit D10 tient** : zéro allocation par step — dump et restore vivent HORS de la
   fenêtre ALLOC-LOOP (DC6).
5. **Aucun restore silencieusement faux** : toute discordance (variante, checkpoint, checksum,
   shape) est un **refus bruyant**, chacun VU échouer (DC5).
6. **L'équivalence est prouvée** aux deux niveaux que la bistabilité autorise (DC2 intra-process
   bit-exact, DC3 inter-process à marge), et le **mordant** est prouvé (DC4 : un cache zéroté
   change la sortie — sinon tous les gates d'équivalence sont vides).

## 2bis. Claims falsifiables — prédictions PRÉ-ENREGISTRÉES

> Toute valeur mesurée qui contredit une prédiction est publiée telle quelle ; une
> requalification exige une décision Régis écrite.

### C-A — « Le round-trip fichier est sans perte »
- **Conviction : certain** (c'est de la copie d'octets).
- **Prédiction** : dump → load → les 4 slices host sont **bit-identiques** (xxh64 égaux) ; les
  champs du manifest survivent au round-trip.
- **Ce qui tue C-A** : un octet qui diffère. Porté par **DC1** (host pur, sans GPU).

### C-B — « Intra-process, restaurer = n'avoir jamais quitté »
- **Conviction : certain** si le contrat de donation est respecté (le host ne relit jamais un
  buffer donné — le dump lit les buffers de **sortie** du dernier step, jamais un buffer donné).
- **Prédiction** : dans un même process, la continuation depuis un cache restauré produit des
  top-5 (ids **et** vals) **bit-identiques** à la continuation de référence, sur la totalité
  des steps comparés.
- **Ce qui tue C-B** : un seul bit d'écart. (Même exécutable ⇒ insensible à l'autotune **par
  construction** — l'argument de S2-PONT.) Porté par **DC2**.

### C-C — « Inter-process, l'état restauré est fidèle — et les seules divergences possibles sont celles de la bistabilité connue »
- **Conviction : probable** (borne connue : la bistabilité inter-process est un fait mesuré —
  `FINDING_NONDETERMINISME_TRAJECTOIRE.md` : l'autotune peut faire flipper un argmax à marge
  fine entre deux process, et **en mode libre un seul flip casque toute la suite**. Aucun
  critère de trajectoire complète n'est donc énonçable inter-process ; le critère porte sur la
  PREMIÈRE divergence, seule attribuable).
- **Prédiction** : un process neuf qui restaure le dump du selftest DC2 et continue en libre :
  (i) au **premier step**, argmax identique à la référence intra-process (sa marge est publiée) ;
  (ii) sur 32 tokens, **la première divergence, si elle existe, est à une marge ≤ 1,873e-3**
  (tie de bistabilité, publié comme tel avec sa marge).
- **Ce qui tue C-C** : une première divergence à marge **> 1,873e-3** — c'est la signature d'un
  état corrompu, pas d'un tie (DC4 le contre-prouve : un cache faux diverge gras et tôt).
  Porté par **DC3**.

### C-D — « Le restore gagne ≥ ×30 sur le re-calcul du préfixe à 4k »
- **Conviction : probable.**
- **Prédiction** : à 4k, atteindre l'état @~3 927 positions (prompt témoin ~27 ids + 3 900
  générés en `--oracle`) coûte ≈ 3927/8,7 ≈ **451 s** de calcul (dans ce moteur, prefill et
  génération ont le même coût par position — prefill par-decode) ; le restore du même état,
  **lecture des ~2,6 GiB incluse mais COMPILE EXCLUE** (instrument `KVLOAD-PERF:`, du début
  de la lecture des tenseurs — post-compile — au 1er token ; la compile est exclue des DEUX
  côtés de la comparaison, sinon elle est inéquitable), coûte **< 15 s** ⇒ gain **≥ ×30**.
- **Ce qui tue C-D** : gain mesuré **< ×5**. (Entre ×5 et ×30 : publié comme écart, chantier
  non tué — la valeur d'usage reste.) Porté par **DC7**.

### C-E — « Le contenu du cache est bien ce qui porte la mémoire » (non-vacuité du dispositif)
- **Conviction : certain.**
- **Prédiction** : un restore depuis un fichier au manifest **valide** mais aux 4 tenseurs
  **zérotés** produit une continuation qui **diverge de la référence DC2 dans les 4 premiers
  tokens générés**.
- **Ce qui tue C-E** : une continuation identique (⇒ le cache ne porte rien ⇒ tous les gates
  d'équivalence de ce chantier sont vides — arrêt immédiat, diagnostic). Porté par **DC4**.

### Grandeurs prédites AVANT mesure (arithmétique dérivée des constantes, pas devinée)

| Grandeur | Valeur prédite | Fondement |
|---|---|---|
| octets tensoriels du dump 1280 | **880 803 840 o** (~840 MiB) | 2×(40·8·1280·256·4) + 2×(8·1·1280·512·4), constantes `:69-85` |
| octets tensoriels du dump 4k | **2 818 572 288 o** (~2,62 GiB) | même formule, L_MAX=4096 |
| `sl_k` seul, 4k | **1 342 177 280 o** | 40·8·4096·256·4 |
| temps de calcul pour atteindre l'état @3 927 pos, 4k | **~451 s** | 8,7 tok/s mesuré (D2) — c'est la grandeur que DC7(i) mesure réellement |
| ALLOC-LOOP avec `--dump-cache` ou `--load-cache` | **identique au run nu** | dump/restore hors fenêtre |

Si une seule de ces valeurs tombe à côté, le résultat est publié avec l'écart et le chantier
s'arrête pour diagnostic.

---

## 3. Non-objectifs (et où vit la dette)

- **E2B** (`gen_auto`, `w4auto`, `bbatch`) — même politique de périmètre que generation-config
  et sampling : 12B seul. Dette au PLANNING.
- **`--repl`** : `--load-cache` + `--repl` = `error.LoadCacheReplUnsupported` (le REPL remet
  l'état à zéro par prompt ; la composition demande une sémantique multi-tour qui n'existe pas).
  Dette écrite — c'est le chemin naturel d'un futur « multi-tour à reprise ».
- **Sampling armé** (décision ouverte n°1 ; proposé : `--dump-cache` + seed ⇒ refus bruyant ;
  `--load-cache` + seed ⇒ **autorisé** — le PRNG repart de la seed CLI, c'est un choix assumé
  et logué, pas une équivalence).
- **8k** (décision ouverte n°2 ; proposé : même code comptime, aucun gate — dette).
- **Compression du dump** (le f32 se comprimerait ~×2 en bf16 — mais on ne dégrade pas un état
  exact pour du disque ; dette si le poids devient un problème).
- **`--window-vacuity` et selftests** : chemins séparés, jamais dumpés (ils `return` avant
  `generateOnce` ou n'en sortent pas un état de génération).
- **Multi-tour / continuation de prompt neuf sur cache restauré** (« restore puis NOUVEAU
  prompt ») : hors périmètre v1 — la v1 continue la génération ou teacher-force ; ajouter des
  tokens de prompt post-restore est un prefill partiel qui mérite son propre chantier.

---

## 4. Design

### 4.1 Format du fichier : safetensors, un seul fichier, auto-décrivant

Généralisation de `writeIdsSafetensors` (`:1545`). Clés :

| Clé | dtype | shape | Contenu |
|---|---|---|---|
| `sl_k`, `sl_v` | F32 | `[40,1,8,L_MAX,256]` | caches sliding (shapes **du binaire qui dumpe**) |
| `fl_k`, `fl_v` | F32 | `[8,1,1,L_MAX,512]` | caches full |
| `ids_fed` | I32 | `[step_next]` | **tous** les tokens feedés (prompt + générés), dans l'ordre |

`__metadata__` (strings, format safetensors standard) :

```
format        = "g12-kvdump-v1"
l_max         = "4096"
step_next     = "<step+1 au moment du dump>"
fed_next      = "<le token sélectionné au dernier step exécuté, pas encore feedé>"
stop_reason   = "eot|max_tokens|l_max|oracle"
ckpt_bytes    = "<taille en octets de model.safetensors>"
ckpt_hdr_xxh64= "<xxh64 du header JSON du checkpoint (8+len octets)>"
gencfg_path   = "<chemin résolu logué par GENCFG:>"
build_mode    = "<la bannière BUILD: mode=… du binaire>"
sampling      = "off" | "T=<t>,top_k=<k>,top_p=<p>"   (trace informative — un restore avec
                d'autres warpers diverge légitimement, mais l'écart doit être VISIBLE)
sl_k_xxh64, sl_v_xxh64, fl_k_xxh64, fl_v_xxh64, ids_fed_xxh64 = "<checksums par tenseur>"
```

**Pourquoi safetensors** : relisible côté Python sans outillage (inspection, gates,
fabrication du mutant DC4/DC5), même famille que toutes les fixtures du projet, et le writer
Zig existe déjà en miniature. **Pourquoi un manifest embarqué et pas un JSON à côté** : deux
fichiers peuvent se désapparier (copié l'un sans l'autre) ; le format safetensors a un champ
`__metadata__` fait pour ça.

**Fingerprint checkpoint — contenu, pas chemin** : `ckpt_bytes` + xxh64 du header JSON du
safetensors de poids (lecture de `8 + header_len` octets, **jamais** des 24 Go). Un même
checkpoint accessible par un autre chemin/symlink reste valide ; un checkpoint différent de
même taille est arrêté par le hash du header (noms/shapes/offsets de tous les tenseurs).

### 4.2 Sémantique du dump

- **Point unique** : dans `generateOnce`, **après** la sortie de boucle et la mesure `elapsed`,
  **avant** les `cache_buf.*.deinit()` (`:2436-2439`). L'état dumpé = celui qui suivrait le
  dernier step exécuté : `step_next = step + 1`, `fed_next = <dernier tok sélectionné>`,
  `ids_fed = ids ++ generated[0..len-1]` (le dernier généré n'a pas été feedé — invariant :
  `ids_fed.len == step_next`, **vérifié par assertion au dump**, refus si faux).
- **d2h sans coût RSS neuf** : `toSlice` des 4 buffers **vers les slices `host.cache_*`
  existantes** (bonnes tailles par construction), puis écriture fichier, puis **re-`@memset(0)`**
  des 4 slices (le contrat « cache ZÉROS par génération » `:2162` reste vrai pour l'appel
  suivant — REPL). Zéro allocation nouvelle de 2,6 GiB, le plafond B10 ne bouge pas.
- Le dump s'exécute **quel que soit `stop_reason`** (y compris arrêt EOS : reprendre après un
  EOS est légitime — c'est l'appelant qui décide).
- **Log obligatoire, une ligne, format imposé** (ce que les gates greppent — boucle de
  formatage manuelle, jamais `{any}`) :

```
KVDUMP: <chemin> l_max=<L_MAX> step_next=<n> fed_next=<id> ids=<n> octets=<total> xxh64_ok
KVLOAD: <chemin> l_max=<L_MAX> step_next=<n> fed_next=<id> ids=<n> (reprise sans prefill)
```

### 4.3 Sémantique du restore

Dans `run()` : après le chargement de `HostInputs` et **avant** la compile (fail-fast §4.5
d'abord, qui ne demande que le header du fichier) :

**En DEUX phases** (rév. 3 — la fenêtre de mesure C-D exclut la compile) :
1. **Phase manifest, AVANT compile** (fail-fast) : lire le header, exécuter les validations
   de forme §4.5 (format, variante, shapes, fingerprint checkpoint, invariants du manifest,
   garde de place, WARN gencfg) — coût : quelques Ko lus ;
2. **Phase tenseurs, APRÈS compile** : chrono `KVLOAD-PERF` démarré ICI ; lire les 4 tenseurs
   **directement dans `host.cache_*`** (remplace le `@memset(0)` de fait) ; vérifier les
   xxh64 **après lecture** (l'ordre lecture→hash→comparaison, pas l'inverse) ;
3. reconstruire `ids_fed` et le passer à `generateOnce` comme `ids`, avec un
   `resume: ?struct { step_next: usize, fed_next: i64 }` non-nul. ⚠ Les pré-checks DUPLIQUÉS
   de `run()` (`ids.len + limit > L_MAX` `:1788`, `ids.len >= SLIDING_WINDOW` `:1793`,
   `positions[0] == ids.len` du mode oracle `:1767-1770`) reçoivent `ids = ids_fed` AVANT de
   s'exécuter ; la garde SLIDING_WINDOW est conditionnée au mode non-load (comme `:2158`),
   les deux autres restent actives telles quelles.

Dans `generateOnce` avec `resume` non-nul :
- `step` démarre à `step_next`, `fed` à `fed_next` — la boucle entre **directement en phase de
  génération** (`in_gen_phase = step + 1 >= ids.len` est vrai par l'invariant §4.2) ;
- la garde prompt `ids.len >= SLIDING_WINDOW` (`:2158`) est **remplacée** en mode resume par
  `step_next + limit <= L_MAX` (un état repris peut légitimement dépasser la fenêtre — le cache
  la porte déjà ; la garde historique protégeait le **prefill**, qui n'a pas lieu) ;
- **rien d'autre ne change** : mêmes `call_args`, même exécutable, mêmes compteurs (D10).

Modes de reprise v1 : `--load-cache` + `--max-tokens N` (libre) et `--load-cache` +
`--oracle <fixture>` (génération autonome **bornée**, arrêt EOS désactivé + verdict A1 —
⚠ `--oracle` n'est PAS du teacher-forcing, leçon F6 de la spec generation-config). `--prompt`
avec `--load-cache` = `error.LoadCacheWithPrompt` (non-objectif §3, refus plutôt que sémantique
implicite ; `--prompt-ids` n'existe pas dans ce runner — vérifié, seule occurrence un
commentaire `:1764`). ⚠ La garde existante « `--prompt` requis hors `--repl` » (`:1626-1630`)
doit apprendre `--load-cache` (sinon aucune reprise ne démarre — finding bloquant de revue).

### 4.4 Détokenisation de la reprise

`generated` ne contient que les tokens de LA reprise ; le texte affiché est celui de la
continuation seule, préfixé d'une ligne loguée `KVLOAD: contexte de <n> tokens (non réaffiché)`.
Reconstituer le texte complet = `scripts/48_detokenize.py` sur `ids_fed` + ids de sortie
(outillage existant, clé `ids` déjà supportée depuis GC6).

### 4.5 Validations au restore — chaque refus est bruyant et TESTÉ

| Cas | Erreur | Vu échouer par |
|---|---|---|
| `format` ≠ `g12-kvdump-v1` | `error.KvDumpBadFormat` | DC5 |
| `l_max` manifest ≠ `L_MAX` binaire | `error.KvDumpVariantMismatch` | DC5(a) |
| shapes/dtype d'un tenseur ≠ shapes compilées | `error.KvDumpShapeMismatch` | DC5 |
| `ckpt_bytes` ou `ckpt_hdr_xxh64` ≠ checkpoint courant | `error.KvDumpCheckpointMismatch` | DC5(b) |
| xxh64 d'un tenseur ≠ manifest | `error.KvDumpChecksumMismatch` | DC1-mutant |
| fichier tronqué / clé absente | erreur de lecture qualifiée | DC5(c) |
| `ids_fed.len ≠ step_next` | `error.KvDumpInconsistentState` | DC5(f) — manifest forgé (script 71 `--set-meta`) |
| `step_next + limit > L_MAX` | `error.SequenceTooLong` (garde §4.3) | DC5(g) — `--max-tokens` volontairement trop grand |
| `--load-cache` + `--repl` | `error.LoadCacheReplUnsupported` | DC5 |
| `--dump-cache` + `--repl` | `error.DumpCacheReplUnsupported` | DC5 (un flag silencieusement inopérant est un mensonge) |
| `--load-cache` + `--prompt` | `error.LoadCacheWithPrompt` | DC5 |
| `--dump-cache` + sampling armé (si décision n°1 = refus) | `error.DumpWithSamplingArmed` | DC5 |
| `gencfg_path` du manifest ≠ chemin résolu courant | **WARN logué, pas un refus** | lecture du log DC3 |

(Le dernier cas est un warn : la politique est re-dérivée des fichiers courants — c'est le
comportement voulu, mais l'écart doit être visible.)

### 4.6 Ce que le design NE fait PAS

- Aucun octet dans `engine.zig`, aucun tenseur nouveau, aucun op : le HLO est byte-identique
  (DC0). Dump = d2h après la boucle ; restore = mêmes `fromBytes` sur des octets différents.
- Aucune allocation dans la boucle de steps : tout le travail vit avant/après (DC6).
- Pas de `realpath` multi-hop sur le checkpoint : le fingerprint est par contenu (§4.1), le
  problème des symlinks HF ne se pose pas.

---

## 5. Gates — chacun avec ce qui le ferait échouer

> Conventions d'exécution : celles du plan D10 (build `-c opt` + `release_fast`, bannière
> `BUILD: mode=ReleaseFast` greppée sinon INEXÉCUTABLE, capture `> out.log 2> err.log`,
> FAIL ⇒ STOP sans requalification à chaud). Tags : `gate/dc<N>-pass`.

| Gate | Prouve | Critère PASS | Ce qui le fait FAIL |
|---|---|---|---|
| **DC0** graphe intact | C du §2.3 | md5 HLO before_optimizations == témoin capturé AVANT la 1ʳᵉ ligne de code ; `git diff engine.zig` vide | 1 octet de HLO ; 1 ligne engine.zig |
| **DC1** round-trip host | C-A | `--selftest-kvdump-io <dir>` (host-only, early-return avant GPU/poids) : write→read→xxh64 identiques sur tenseurs synthétiques + manifest round-trip + **mutant intégré** (flip d'1 octet ⇒ `KvDumpChecksumMismatch` VU) ; complété par le mutant Python sur un dump RÉEL (script 71 `mutate-flip` sur `dc2.kvdump`) | un hash qui diffère ; un des deux mutants qui passe |
| **DC2** équivalence intra-process | C-B | `--selftest-kvdump-eq` : TROIS appels `generateOnce` dans le même process — (1) gen k=16 + dump ; (2) gen k+m=48 depuis zéro = référence ; (3) restore du fichier + gen m=32 ; top-5 ids+vals des steps k..k+m de (2) **bit-identiques 32/32** à ceux de (3). Pré-condition auto-vérifiée : les k premiers tokens de (1) et (2) identiques (déterminisme intra-process). Si un appel s'arrête avant sa borne (`stop_reason != max_tokens`, EOS précoce) ⇒ **INEXÉCUTABLE**, pas FAIL — c'est la pré-condition qui couvre le risque EOS, aucun précédent de mode libre n'est invoqué (rév. 3 : la tenue « ≥ 1150 tokens » citée en rév. 2 venait de runs ORACLE, où l'EOS est désactivé — elle ne témoignait de rien) | 1 bit d'écart entre (2) et (3) |
| **DC3** équivalence inter-process | C-C | process neuf : restore du fichier produit par DC2 + continuation libre 32 tokens ; PASS = argmax du 1er step identique à la référence DC2 (marge publiée) ET première divergence éventuelle à marge ≤ 1,873e-3 (publiée comme tie) ; `n_match/32` publié à titre informatif | 1ʳᵉ divergence à marge > seuil ; ou argmax du 1er step différent hors tie |
| **DC4** le mordant | C-E | restore d'un fichier zéroté (fabriqué par script Python, manifest valide) : divergence vs référence DC2 dans les **4 premiers** tokens | continuation identique (⇒ dispositif vide, STOP diagnostic) |
| **DC5** refus bruyants | §4.5 | (a) dump 1280 chargé par binaire 4k ⇒ `VariantMismatch` ; (b) `ckpt_hdr_xxh64` altéré ⇒ `CheckpointMismatch` ; (c) fichier tronqué à 1000 octets ⇒ erreur qualifiée, pas un crash ; (d) `format` altéré ⇒ `BadFormat` ; (e) shape altérée dans le header ⇒ `ShapeMismatch` ; (f) `step_next` forgé ≠ `ids_fed.len` ⇒ `InconsistentState` ; (g) `--max-tokens` > place restante ⇒ `SequenceTooLong` ; + les 4 refus de flags (§4.5) | un refus qui ne se déclenche pas, ou un crash non qualifié |
| **DC6** interdit D10 intact | §2.4 | `ALLOC-LOOP:` d'un run `--dump-cache` ET d'un run `--load-cache` == valeurs du run nu (mêmes compteurs). ⚠ Les deltas ALLOC-LOOP sont **figés dans des locales immédiatement après la boucle, AVANT le bloc de dump** (finding bloquant de revue : le dump alloue via l'allocateur compté, la ligne de log est émise après lui — sans ce gel, DC6 échouerait par construction en incriminant la boucle à tort). Volet RSS : runs de **≥ 220 tokens** via `--oracle` sur fixture fabriquée (à 32 tokens, `RSS-DELTA` émet `INEXECUTABLE` — `:2492-2494` — le critère serait invérifiable) ; `RSS-DELTA` dans le plafond B10 | +1 alloc dans la fenêtre ; RSS au-delà du plafond ; ou `RSS-DELTA: INEXECUTABLE` (run mal dimensionné = INEXÉCUTABLE, pas PASS) |
| **DC7** le gain, mesuré | C-D | à 4k : (i) `--oracle` sur fixture fabriquée de 3 900 ids (⚠ PAS le mode libre : l'arrêt EOS couperait bien avant — en `--oracle` l'EOS est désactivé, décision Régis 28 juil) + `--dump-cache` → temps de calcul publié (ligne `PERF :`, compile exclue) ; le verdict A1 du run peut sortir en erreur (fixture factice) : le dump et `PERF :` sont émis AVANT lui, c'est accepté et noté ; (ii) process neuf : `--load-cache` + `--max-tokens 1` → **`KVLOAD-PERF:`** = chrono du début de la lecture des tenseurs (phase 2 du load, POST-compile) au 1er token — lecture des 2,6 GiB INCLUSE (un chrono qui l'exclut serait biaisé vers le PASS), compile EXCLUE (elle est exclue du côté (i) aussi : l'inclure d'un seul côté serait inéquitable) ; gain ≥ ×30 attendu, < ×5 = kill | gain < ×5 ; ou temps non mesurés (INEXÉCUTABLE) |

**Non-vacuité transversale** : DC2/DC3 ne sont déclarés PASS que si DC4 a produit sa
divergence — un dispositif d'équivalence dont on n'a pas vu le contraire mordre n'a rien
prouvé (leçon `feedback_invariant_tue_le_controle`).

---

## 6. Livrables

1. `zml_runner/kvdump.zig` — module : writer/reader safetensors généralisé, manifest,
   validations, xxh64. **Std-only** (pas de zml), testable host — DC1 l'exerce via un mode
   `--selftest-kvdump-io` host-only du binaire (early-return avant tout GPU/poids, patron
   `--selftest-draw`).
2. `gemma4_g12auto.zig` — flags `--dump-cache`/`--load-cache`, point de dump §4.2, chemin
   resume §4.3, usage string, logs `KVDUMP:`/`KVLOAD:`.
3. `scripts/71_kvdump_inspect.py` — inspection/fabrication côté Python (lit le manifest,
   vérifie les xxh64, fabrique les mutants DC4/DC5). Réutilise `safetensors` + `xxhash`
   (dispo : `pip show xxhash` à vérifier en Task 0, sinon `zlib.crc32` en repli — auquel cas
   le manifest porte `crc32` et non `xxh64`, décidé en Task 0, PAS improvisé plus tard).
4. `docs/KVDUMP_RESULTS.md` — verdicts des 8 gates, claims jugées, dettes.
5. PLANNING.md, fiche mémoire, PR.
