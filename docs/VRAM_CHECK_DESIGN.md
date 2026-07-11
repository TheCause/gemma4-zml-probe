# Design — Check VRAM au lancement de `gemma4_gen_auto`

> Item [B] du PLANNING, motivé par l'incident du 11 juil 2026 : Ollama occupait ~22/24 Go de la
> 3090 → OOM dès la matérialisation (`CreateBuffersForAsyncHostToDevice … 6.00MiB`) suivi d'un
> crash `General protection exception` dans `io.zig deinit` (double-free post-OOM, bug d'error-path
> UPSTREAM ZML). L'OOM est la vraie erreur ; le but ici est un message explicite AVANT, pas de
> patcher ZML.
>
> Décisions de cadrage (Régis, 11 juil 2026) : portée = `gemma4_gen_auto` seul ; comportement =
> fail-fast + échappatoire ; mécanisme = spawn `nvidia-smi`.

## 1. Comportement

Le binaire interroge `nvidia-smi` **avant tout travail GPU** (poids, `Platform.init`) mais
**après** les branches host-only qui font early-return sans jamais toucher le GPU
(`--selftest-inputs`, `--ids-only`) — la garde ne doit pas bloquer un mode qui n'a pas besoin de
la carte.

> **Errata L3 (12 juil 2026)** : `--selftest-gather` était host-only à l'origine ; depuis L3
> (`L3_INGRAPH_DESIGN.md` gate SG), ce mode compile un mini-graphe GPU (gather in-graph) — il est
> **reclassé GPU** et la garde VRAM s'y applique désormais comme à un run normal (il requiert un
> `--prompt` factice pour atteindre le dispatch, placé en aval de la garde).

Si la VRAM libre du GPU 0 est sous le seuil
requis, il sort en `error.GpuBusy` avec un message qui :

- chiffre l'écart : `VRAM libre X GiB < Y GiB requis` ;
- liste les process compute occupants (PID, nom, mémoire) ;
- suggère la libération en deux temps : `ollama ps` puis `ollama stop <modèle>` (nvidia-smi ne
  donne que le nom du process, pas celui du modèle — réversible, keep_alive recharge à la demande).

Nouveau flag `--force-vram` pour outrepasser en connaissance de cause.

Exemple de sortie attendue (illustratif — les chiffres réels mesurés sont au §6) :

```
$ gemma4_gen_auto --prompt "..."
error: GPU occupé — VRAM libre 1.8 GiB < 10 GiB requis
  PID 41337  /usr/local/bin/ollama  21.9 GiB
Libérer d'abord : ollama ps && ollama stop <modèle>
(ou --force-vram pour tenter quand même)
```

## 2. Détection

Deux invocations `nvidia-smi` :

| Requête | Usage |
|---|---|
| `--query-gpu=memory.free --format=csv,noheader,nounits` | MiB libres ; première ligne = GPU 0 (VM 3090 mono-GPU) |
| `--query-compute-apps=pid,process_name,used_memory --format=csv,noheader` | liste des occupants, pour le message uniquement |

Le parsing CSV vit dans des fonctions **pures** (`parseFreeMiB`, `parseComputeApps`), testables
indépendamment du spawn.

## 3. Seuil

Constante `MIN_FREE_VRAM_GIB = 10`, commentée : mesure G2.1 = **8,5 GiB réels** pour ce modèle
(poids déjà bf16 sur device, cf `docs/G2_BF16_FIDELITY.md`). Le commentaire doit chiffrer la marge
honnêtement : le binaire initialise CUDA avec BFC `memory_fraction 0.90, preallocate=true`, donc à
10 GiB libres la réserve utilisable est ~9 GiB → marge réelle ~0,5 GiB au-dessus des 8,5 mesurés.
Pas de flag de réglage (YAGNI) ; la constante est déclarée avec les autres invariants du runner.

> **Errata L3 (12 juil 2026)** : ce seuil de 10 GiB est celui du chantier `vram-check` initial
> (pré-L3). Le chantier L3 in-graph (`L3_INGRAPH_DESIGN.md`) ajoute la table
> `embed_tokens_per_layer` en résidence device pour le gather in-graph → pic mesuré **16,27 GiB**
> (16 658 MiB, run `preallocate=false` post-L3). Le seuil est **porté à `MIN_FREE_VRAM_GIB = 20`**
> (`ceil(16,27 / 0,90) + 1`, même formule qu'ici), cf `L3_INGRAPH_DESIGN.md` § Résultats. V1/V3
> ci-dessous ont été re-runs avec ce nouveau seuil (§6).

## 4. Cas limites — garde best-effort, jamais bloquante à tort

| Cas | Comportement |
|---|---|
| `nvidia-smi` introuvable | `log.warn` + on continue (machine sans GPU ; l'OOM réel reste le filet) |
| `nvidia-smi` en échec / sortie illisible | `log.warn` + on continue (outil cassé ≠ GPU occupé) |
| `nvidia-smi` qui pend | hors scope (pas de timeout de spawn — coûteux sous l'API `Io` 0.16 pour un cas jamais observé ; Ctrl-C utilisateur) |
| modes host-only (`--selftest-inputs`, `--ids-only`) | check jamais atteint (placé après leurs early-returns, cf §1) |
| `--selftest-gather` | **reclassé GPU depuis L3** (errata §1) — la garde s'applique comme à un run normal |
| `--allow-cpu` actif | check ACTIF quand même (errata revue 11 juil : --allow-cpu ne force pas le CPU, l'init .cuda est tentée d'abord — sur machine sans GPU le best-effort couvre déjà le cas) |
| `--force-vram` actif | check sauté avec log explicite |
| VRAM libre ≥ seuil mais process présents | on continue silencieusement (seul le seuil décide) |

## 5. Risque identifié

L'API child-process de Zig 0.16-dev (`std.process` sous le nouveau `Io` threadé) a bougé — les
équivalents 0.16 sont documentés au repo pour les fichiers (`std.Io.Dir.cwd()` etc.,
cf PLANNING P5.2.A), pas pour le spawn. **À vérifier en tête d'implémentation** ; si le spawn
direct s'avère pénible, repli : un unique `sh -c "nvidia-smi … ; nvidia-smi …"` et parsing de la
sortie concaténée.

## 6. Validation (style gates du repo, sur la 3090)

| Gate | État initial | Attendu |
|---|---|---|
| **V1 — la garde mord** | Ollama chargé (~22 Go) | message explicite + exit propre `error.GpuBusy`, pas d'OOM ni de crash double-free |
| **V2 — échappatoire** | idem + `--force-vram` | check sauté (l'OOM assumé peut suivre, c'est le contrat) |
| **V3 — non-régression** | GPU libre | génération normale, sortie identique à avant (check neutre) |

**Résultats (11 juil 2026, 3090)** — exécution subagent-driven, logs locaux (`logs/` est
gitignored dans ce repo — les logs de gates restent hors git, la doc porte les résultats) :

| Gate | État initial mesuré | Verdict |
|---|---|---|
| V1 | Ollama résident (llama-server, 22 588 MiB), 1 780 MiB libres | **PASS** — `GPU occupé — VRAM libre 1.7 GiB < 10 GiB requis`, process listé, suggestion `ollama ps`/`ollama stop`, exit `error.GpuBusy` propre, zéro OOM / zéro crash — `logs/vram_check_v1.log` |
| V2 | idem + `--force-vram` | **PASS** — warn « garde VRAM sautée », backend cuda atteint, OOM assumé (36 MiB à la matérialisation) + crash error-path upstream connu (cosmétique), exit 134 — `logs/vram_check_v2.log` |
| V3 | GPU libéré (24 373 MiB libres) | **PASS** — garde totalement silencieuse, « Paris », early-stop EOT, 51,0 tok/s, exit 0 — `logs/vram_check_v3.log` |

Test Zig du parsing CSV si la toolchain Bazel du repo le permet sans cérémonie, sinon couvert par
V1-V3.

**Errata L3 (12 juil 2026) — re-run au seuil final 20 GiB** (cf §3, gate VG de
`L3_INGRAPH_DESIGN.md` § Résultats) :

| Gate | État initial mesuré | Verdict |
|---|---|---|
| V1 | Ollama résident, 22 588 MiB | **PASS** — `GPU occupé — VRAM libre 1.7 GiB < 20 GiB requis`, GpuBusy propre — `logs/l3_vg_v1.log` |
| V3 | GPU libre | **PASS** — garde silencieuse, « Paris », exit 0 — `logs/l3_vg_v3.log` |

## 7. Périmètre exact

- `zml_runner/gemma4_gen_auto.zig` : check + flag `--force-vram` + fonctions de parsing.
- `zml_runner/BUILD.bazel` : inchangé (pas de nouvelle dépendance).
- Doc : `docs/DOCUMENTATION.md` (§ usage `gemma4_gen_auto`), `PLANNING.md` (solder l'item [B]).
- **Rien dans les autres runners** (le helper reste factorisé proprement dans `gen_auto` pour être
  extrait plus tard si besoin).
