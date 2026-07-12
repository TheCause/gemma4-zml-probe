# Protocole du banc batché — **pré-enregistré**

> **Committé AVANT le premier run de mesure** (discipline G2.3 §5 : le protocole se fige avant
> de voir les résultats, sinon la requalification se négocie après coup — précédent A2).
> Toute exécution qui s'écarte de ce document publie la **déviation** dans
> `docs/BATCHING_RESULTS.md`.
> Spec : `docs/superpowers/specs/2026-07-12-batching-flash-attn-design.md` §3.3 / §6 B4.

## 1. Matériel et custody

- VM 3090 (24 GiB), driver 580.159.03, ZML `adee932e` (rev capturée en T0 — voir
  `fixtures/batch_manifest.json`), patch quota `pjrt/pjrt.zig:26`.
- **Un build unique pour tout le sweep** : le moteur est shape-polymorphe (gate T0), donc
  **un seul binaire `gemma4_bbatch` sert tous les B**. Son **sha256 est consigné une fois** et
  **vérifié avant chaque run** — un hash qui change invalide le sweep (doctrine G2.3 §7.1).
- **GPU vierge** exigé : `nvidia-smi --query-compute-apps` avant chaque point ; toute contention
  (Ollama ~22 Go) invalide le point. La garde de contention de bbatch le refuse déjà, mais la
  vérification est faite explicitement par le script.

## 2. Points de mesure

- **B ∈ {1, 2, 4, 8, 16, …}** — liste **bornée et doublante** (pas de balayage linéaire :
  chaque B est un graphe tracé/compilé, ~17 s + RAM de compile).
- **Arrêt par projection, jamais par crash** : avant le point 2B, projeter
  `pic(2B) ≈ pic(B) + B × Δ` (Δ ≈ 38 Mo de cache f32/lane + marge activations).
  Si la projection dépasse **~22 GiB**, le sweep s'arrête et rapporte le dernier B tenu.
  Motif : l'error-path OOM upstream crashe en General protection exception (`io.zig` deinit,
  `PLANNING.md`) — on ne « pousse pas jusqu'au mur ».

## 3. Charge par point

| B | Prompts | Fidélité |
|---|---|---|
| 1 (bras apparié) | fixture 49 (gen_auto est mono-prompt) | oracle 48/48 |
| 2 | `fixtures/bench_prompts_b2.txt` (2 prompts **distincts**) | oracle **par lane**, 48/48 |
| 4 | `fixtures/bench_prompts_b4.txt` (4 prompts **distincts**) | oracle **par lane**, 48/48 |
| ≥ 8 | `--replicate` sur le jeu B=4 | **spot-check** : la lane 0 doit == sa fixture |

Tous les prompts du jeu ont la **même longueur tokenisée** (19 tokens, template chat inclus) —
contrainte V1 imposée par la position partagée du moteur (spec §2.2). Jeux **versionnés**.

## 4. Mesure de perf — non-régression B4

- **Comparateur = runs frais appariés** `gemma4_gen_auto` (B=1) vs `gemma4_bbatch` (B=1), lancés
  **dans la même fenêtre de session**, sur la **même charge** (fixture 49, 48 tokens).
  **Jamais** les chiffres publiés seuls (110-113 tok/s) : le bruit inter-compiles est de 2 à 16 %
  (« piège 15 »), un `≥` nu contre un chiffre historique produirait un FAIL ambigu.
- **3 runs par bras**, statistique = **médiane**.
- **Critère PASS pré-enregistré** :
  `médiane(bbatch B=1) ≥ 0,95 × médiane(gen_auto B=1)` en tok/s de **génération**
  (budget bruit **−5 %**, qui couvre le ~2 % « grand effet » du piège 15 avec marge).
- **Confound K** : bbatch et gen_auto rapatrient tous deux K=5 par step (le K paramétrable du
  mode sampling est hors gates). K est consigné au manifest. Seuil « marginal » pré-défini :
  si le verdict se joue à **moins de 5 points de pourcentage**, re-run avec K explicitement
  apparié, qui **remplace** le verdict initial ; les deux mesures sont publiées.
- Convention de comptage identique à L3 (s0 produit par le dernier call de prefill mais compté
  dans `generated`) — sinon les chiffres ne sont pas comparables aux 110-113 tok/s de référence.
- Prefill et génération **mesurés séparément** ; tok/s **agrégé** ET **par lane** rapportés.

## 5. Mesure VRAM — le plafond est l'output

- **`--no-prealloc` obligatoire** : sous préallocation BFC (0,90 × libre), `nvidia-smi` ne montre
  que la **réserve**, pas le besoin réel (« piège 14 »). `mem_probe` mesure la **RSS host**,
  pas la VRAM — ne pas s'en servir.
- Échantillonnage `nvidia-smi --query-compute-apps` **pendant** le run (compile + prefill + gén),
  pic scopé au PID du runner.
- **Charge du run VRAM** : run long (999 tokens) — vérifier `ids.len + 999 ≤ L_MAX = 1024`
  (avec des prompts à 19 tokens : 19 + 999 = 1018 ✓).
- Le pic mesuré par B est publié tel quel ; la garde de `gemma4_gen_auto` (20 GiB, calibrée B=1)
  reste **intouchée**.

## 6. Ce qui est publié

Table `B → {tok/s agrégé, tok/s par lane, pic VRAM (MiB), compile (s), verdict fidélité}`
dans `docs/BATCHING_RESULTS.md`, **plafond B rapporté**, manifest de custody mis à jour
(`fixtures/batch_manifest.json`). Un point FAIL ou un plafond plus bas qu'espéré se publie
au même titre qu'un PASS.
