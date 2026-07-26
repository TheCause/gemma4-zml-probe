# U_12B_RESULTS — W4-J2 : Gemma 4 12B Unified w4a16 sur la 3090

**Chantier** : W4-J2 (2e jalon de la brique poids 4-bit, spec
[`2026-07-18-w4-poids-4bit-12b-design.md`](superpowers/specs/2026-07-18-w4-poids-4bit-12b-design.md),
plan [`2026-07-24-w4-j2-12b-unified.md`](superpowers/plans/2026-07-24-w4-j2-12b-unified.md) + 3 amendements).
**Exécution** : 24-25 juil 2026, branche `w4-j2-12b`. **Statut : CLOS — 11 gates U0→U10 PASS.**

**Résultat central** : `google/gemma-4-12B-it-qat-w4a16-ct` (checkpoint QAT officiel,
`Gemma4UnifiedForConditionalGeneration`, 48 couches, ~24 Go bf16 = infaisable seul sur 24 Go
de VRAM) **décode sur la RTX 3090 en ZML** : 1150 tokens stables à **9,0 tok/s**, pic VRAM
réel **16 680 MiB**, et la sortie est **== HF-fp32-même-checkpoint STRICT** — 48/48 (U8
re-fondé) et 1150/1150 (U9-iv) au teacher-forcing, **zéro requalification**. Le moteur E2B
est préservé par preuve HLO byte-identique (U1).

---

## 1. L'histoire épistémique — l'événement principal du chantier

Ce chantier a failli se clore sur des verdicts « requalifiés différentiels ». Il se clôt sur
des verdicts **stricts**, parce que le problème était dans l'instrument, pas dans le runner.

- **Le glissement** : entre J1 (E2B) et J2, l'oracle HF est passé de **fp32** (scripts 46/56 :
  « poids fp32, oracle = source de vérité ») à **bf16** (script 69 — un 12B fp32 monolithique
  = ~48 Go, au-dessus de la RAM de l'hôte oracle). Quantum bf16 = 0,125 sur des logits ~25 →
  l'oracle fabrique des ties artificiels sur ~1 % des steps (48 quasi-ties mesurés sur 1150).
- **Les symptômes** : U8 brut 42/48 (« tie bf16 exact à gen=39 », requalifié différentiel,
  décision ratifiée — commit « plan(j2): U8 REQUALIFICATION DIFFÉRENTIELLE RATIFIÉE », 25 juil) ; puis U9-iv brut 1139/1150 — 11 « divergences », toutes des paires
  top1/top2 inversées à 0-2 ULP bf16, et un menu de requalification dont le critère glissait
  déjà (« marge ≤ 1e-3 » → « ≤ 2 ULP »).
- **Le recadrage** (Régis, 25 juil au soir) : « fausse route sur les gates précédentes […]
  le plus juste, le moins corrompu ». Diagnostic : la requalification différentielle répétée
  ne peut pas distinguer le bruit d'oracle d'un petit bug systématique — les deux ne flippent
  que des quasi-ties. Chaque requalification acceptée rendait la suivante plus facile.
- **La réparation** (Amendement 3) : `69_u8_gen_oracle.py --compute-fp32` — poids **stockés**
  bf16, chaque module à paramètres directs (626) converti en fp32 **le temps de son forward**
  (hooks, bf16→fp32→bf16 sans perte). Le « fp32 = 48-52 Go infaisable » confondait stockage et
  arithmétique. Parité restaurée avec les oracles J1 et avec le runner (stockage w4, calcul f32).
- **Le verdict** : U8 **48/48 STRICT** (à gen=39 la marge fp32 réelle est 0,026 — le runner
  avait raison ; la « cascade 40-47 » n'existait pas), U9-iv **1150/1150 STRICT** (marge min
  0,0279 @ gen=1043 — aucun vrai tie sur tout le run ; les 11 « divergences » bf16 : marges
  fp32 réelles 0,028-0,24, toutes des artefacts de quantum). Bonus : l'oracle fp32 est ~5×
  plus rapide que le bf16 émulé (prefill 1176 : 58 s vs 284 s).

Leçon capitalisée (mémoire) : *l'instrument dégradé fabrique des requalifications en cascade —
2e requalification du même type = STOP, diff l'instrument ; stockage ≠ arithmétique.*
La requalification U8 d'origine reste dans l'historique (commits « plan(j2)/gate(j2) U8
requalification », 25 juil après-midi), **SUPERSEDED**
sans réécriture.

## 2. Gates — verdicts

| Gate | Objet | Verdict / chiffres | Tag |
|---|---|---|---|
| U0 | Contrat 12B vérifié sur pièce | 328 linears packés ; **hétérogène** : sliding GQA 16Q/8KV tête 256, full **MQA 1×512 K=V sans v_proj** (erratum spec §4.2), motif %6, softcap 30, vocab 262144 ; manifest auto-vérifié exit-code | `gate/j2-u0-pass` |
| U1 | Moteur paramétré `Geom` comptime | HLO E2B **byte-identique (md5)** avant/après + 2 témoins re-PASS — le refactor n'a rien changé au moteur E2B | `gate/j2-u1-pass` |
| U1b/U2 | Export dq streaming + embed | Dequant 3 shapes **bit-exact** ; census 328 modules asserté + voie officielle ct ; embed gather + scale bf16 62.0 **bit-exact** (D12) | `gate/j2-u2-pass` |
| U3 | QKV sliding L0 (hooks réels) | max_abs ≤ 1e-4 S=8 ; tripwire S=1040 : écart ULP `zml.nn.rope` structurel, s'annule dans l'output — **ratifié** | `gate/j2-u3-pass` |
| U4 | GQA groupe 2 + masque 1024 | Oracle `repeat_kv` réel ; masque mordant prouvé | `gate/j2-u4-pass` |
| U5 | Couche full L5 : K=V, p-RoPE, MQA | Discriminabilité **câblée et mesurée sur la sortie d'attention : ×24 (S=8) / ×19 (S=1040)** ≥ 10× seuil ; fait de checkpoint découvert par le STOP câblé : q_norm/k_norm des 8 couches full **UNIFORMES** (scalaires broadcastés, k_norm L5 = 0.0605, probable artefact QAT) ; étage (c) attention complète ≤ 1e-4 aux deux S | `gate/j2-u5-pass` |
| U6 | Couche 0 + chaîne L0→L5 par `runLayerGen`/`Geom.g12` | max_abs ≤ 1e-3 (resserrable, jamais élargissable) ; branche moteur K=V réellement exercée (D4, placeholder v_proj `[1]` prouvé non consommé par compile) ; distribution des 48 layer_scalar consignée | `gate/j2-u6-pass` |
| U7 | Prefill 48 couches + softcap | **RESSERRÉ 26 juil à l'oracle fp32 (68 `--compute-fp32`)** : max_abs = **9.365e-4** ≤ seuil originel §3-U7 **1e-2 restauré et câblé** (marge ~10×) ; top-5 ensemble+ordre EXACTS, zéro tie utilisé ; marge top1-top2 12.559394 (précision fp32) ; softcap mordant identique (26.3532 vs 26.3531). L'ancien 0.376/0.5 vs oracle bf16 (Amendement 2) est SUPERSEDED — c'était le quantum de l'oracle, pas le bruit du moteur | `gate/j2-u7-pass` (+ resserrage fp32) |
| U8 | Décode court GPU vs HF-même-checkpoint | Brut vs oracle bf16 : 42/48 (requalification du 25 juil, SUPERSEDED) → **re-fondé oracle fp32 : 48/48 STRICT** ; contre-test câblé : gate_proj L24 ×100 → `A1Mismatch`, divergence logits ×200 le bruit | `gate/j2-u8-pass` |
| U9 | Décode long 1150 | (i) 1150 tok stables, 0 NaN, texte cohérent ; (ii) morsure fenêtre **in-process même binaire** : logits bit-identiques q ≤ 1023, première divergence **exactement à q=1024** (max_abs 6.3e-2) ; (iv) teacher-forcing **fp32 1150/1150 STRICT** | `gate/j2-u9-pass` |
| U10 | Observations perf/VRAM | Voir §3 | `gate/j2-u10-pass` |

Non-vacuité exercée dans les deux sens : U0 (exit code), U1 (md5), U5 (STOP câblé — il a
réellement attrapé le fait q_norm/k_norm uniformes et forcé l'amendement de la mesure),
U8 (contre-test A1Mismatch), U9-ii (divergence à q=1024 exactement, pas ailleurs), et côté
oracle : 48 quasi-ties sur le run U9 dont 37 matchent en bf16, 1150/1150 en fp32.

## 3. U10 — observations (mesures 3090, protocole G3 `--no-prealloc`)

| Mesure | Valeur | Note |
|---|---|---|
| Débit génération | **9,0 tok/s** (1150 tok en 127,2 s) | identique avec/sans prealloc ; 12,2 tok/s sur micro-run (4 tok) |
| Débit prefill (step-par-step) | 8,8 tok/s (27 steps en 3,07 s) | |
| Pic VRAM réel (décode 1150) | **16 680 MiB** | poller nvidia-smi 1 Hz, `pgrep -x` (nom exact) |
| Pic VRAM réel (décode 4) | 16 696 MiB | **≈ identique → allocation statique**, indépendante de la longueur |
| Projection du plan | ~10-12 Go | **dépassée de ~5-6 Go** : poids ~9,6 + cache f32 L_MAX=1280 ~0,9 ; le delta = arène XLA du graphe step (transients de dequant W4 par step compris) — non projetée |
| Référence | 24 Go bf16 poids seuls | le 12B bf16 ne RENTRE PAS sur la 3090 ; w4a16 laisse ~7,3 Go de marge |
| Oracle M4 (contexte) | bf16 : 28,6 s/token (U8), prefill 1176 = 284 s ; **fp32 : prefill 1176 = 58 s** | le bf16 CPU était émulé — l'instrument fp32 est meilleur ET plus rapide |

L'écart de projection est consigné comme observation (gate observationnel — pas de seuil) :
la projection ne comptait que poids+cache+« transients » non chiffrés ; le poste manquant est
l'arène d'exécution XLA du mono-graphe step (dequant W4 de 328 linears + buffers d'attention
sur cache 1280). Piste de réduction connue (non exercée) : scinder dequant/step ou libérer
les buffers de prefill.

## 4. Décisions en cours de route (toutes datées dans le plan)

- **D6** : oracle full-decode = HF CPU **M4** (export dq rsyncé), VM = repli — la voie GPU-HF
  est morte (compressed-tensors décompresse tout → OOM, Amendement 1).
- **D12** : embed scale — reproduction HF directe (round-trip bf16 62.0), bit-exact au gate U2.
- **U3/U5 amendements ratifiés** : écart ULP rope structurel (s'annule en aval) ; mesure de
  discriminabilité déplacée sur la sortie d'attention après découverte des k_norm uniformes.
- **Amendement 2** : U7 max_abs requalifié en garde-fou 0.5 vs oracle bf16-réel (le seuil 1e-2
  était un « contrôle qui ne peut pas réussir » contre du bf16) ; top-5+marges = discriminant.
- **Amendement 3** (l'événement du chantier, §1) : oracle **fp32** = instrument officiel des
  gates argmax J2 ; U8 re-fondé 48/48 strict ; U9-iv 1150/1150 strict.
- **Resserrage U7 (26 juil, post-merge — la « voie ouverte » de l'Amendement 3 refermée)** :
  fixture fp32 (68 `--compute-fp32`, hôte M4 — la VM n'a pas la RAM), gate re-jugé :
  **max_abs 9.365e-4** (vs 0.376 à l'ère bf16, ×400) ; seuil originel §3-U7 **1e-2 restauré**
  dans le runner (mesure d'abord au garde-fou 0.5, resserrage au vu de l'oracle, re-run au
  seuil câblé — doctrine U6). Les TROIS requalifications bf16 de J2 (U7 max_abs, U8 42/48,
  U9 marges) se sont toutes résolues au même remède : l'instrument fp32.

## 5. Findings transférables

1. **Stockage ≠ arithmétique** : un oracle « infaisable en fp32 » ne l'était que pour le
   *chargement monolithique* — 626 hooks par-module (fp32 le temps du forward) donnent la
   précision fp32 dans l'empreinte RAM bf16, et 5× plus vite que le bf16 CPU émulé.
2. **Fait de checkpoint QAT** : les q_norm/k_norm des 8 couches full du 12B sont des scalaires
   broadcastés (uniformes). Candidat transfert TurboQuant/alambic (comme la scale-invariance
   des linears E2B en J1).
3. **Le 12B Unified est hétérogène** : sliding GQA 16/8×256 vs full MQA 1×512 **K=V sans
   v_proj** — toute hypothèse « carte uniforme » était fausse (erratum spec §4.2).
4. **Signature d'un instrument au quantum trop gros** : échecs répétés « de peu », tous en
   paires top1/top2 inversées, jamais de vraie dérive. Le remède est l'instrument, pas la
   requalification.

## 6. Artefacts

- Rapports oracle (locaux, `logs/` non versionné) : `u8_tf_fp32.json` (48/48),
  `u9_tf_fp32.json` (1150/1150, marges par position), `u9_tf.json` (bf16, comparaison),
  `u9_gen_1150.log`, `u9_window_vacuity.log`, `u10_rerun.log`/`u10_short.log` (VM `/tmp`).
- Scripts : 62-70 (venv g12b M4/VM) ; runners : `gemma4_g12gate.zig` (gates unitaires),
  `gemma4_g12auto.zig` (décode autonome, flags `--dump-top5 --out-ids --window-vacuity
  --no-prealloc --oracle`).
- Commits clés : contrat `gate/j2-u0-pass` → … → `gate/j2-u10-pass` ; instrument fp32
  « feat(j2): oracle --compute-fp32 » ; Amendement 3 « plan(j2): AMENDEMENT 3 RATIFIÉ ».
  (Les sujets de commit et les tags font référence — la branche a été réécrite pour
  anonymisation le 25 juil au soir, les SHA antérieurs ne résolvent plus.)
