#!/usr/bin/env bash
# g2_3_sweep.sh — orchestrateur du sweep G2.3 sur la 3090 (docs/G2_3_OP_SENSITIVITY.md §8).
#
# UN BINAIRE UNIQUE pour tout le sweep (§8.1) : build au départ, sha256 vérifié avant CHAQUE run
# (refus si le binaire change en cours de sweep). Par config : run → md5 du dump → check HLO
# (script 53) → analyse (script 52) → purge du dump (§8.2). Survivent à la purge : D0 (jamais
# purgé), le manifest, les npz de métriques (écrits par le 52), les rapports HLO json, le log
# des md5 — et le dump complet si KEEP=1 (run combiné D*, §7.2 spec).
#
# Usage (sur la VM 3090) :
#   bash scripts/g2_3_sweep.sh [configs]
#     [configs] = configs séparées par des ESPACES ; chaque config = "none" ou une liste de
#     familles séparées par des VIRGULES **SANS ESPACES** ("qkv_proj,mlp") — fromSpecList
#     (engine.zig) ne trim PAS : "mlp, ple" → error.UnknownFamily, hard-fail VOULU.
#     Défaut = none (D0, en PREMIER : référence de tous les suivants) + les 11 familles one-hot
#     HORS kv_store — protocole §3 : kv_store exige la fixture variante cache-bf16, la lancer
#     avec la fixture standard f32 donnerait un VACUOUS garanti. Run kv_store explicite :
#       FIXTURE=/data/gemma4-zml-probe/gen_long_kvbf16.safetensors bash scripts/g2_3_sweep.sh "kv_store"
#
#   Variables d'environnement :
#     RUN_PREFIX (défaut g2_3)   namespace des sorties — stabilité S49 : RUN_PREFIX=g2_3_s49 (§7)
#     FIXTURE    (défaut $DATA/gen_long.safetensors — RACINE du repo côté VM, cf script 46,
#                PAS fixtures/) ; surchargée pour S49 (gen_custom) et kv_store (gen_long_kvbf16)
#     REF_A      (défaut $DATA/g2_logits_a_f32.npy — .npy memmap du script 50, custody par le 52)
#                REF_A=none → mode S49/diagnostic : le 52 tourne vs D0 seul et SANS --envelope
#                (pas de buckets — pas d'enveloppe S49, §7) ; les gates 53/anti-câblage restent
#     KEEP=1     conserver dump logits + dump HLO après analyse (run combiné D*)
#     REPO_REV   rev git du repo probe — À PASSER DEPUIS M1 : le /data/gemma4-zml-probe de la VM
#                n'est PAS un clone git (transport rsync, §8.3), le fallback local vaudrait "n/a"
#     ZML_WS, DATA, OUT_ROOT : défauts ci-dessous (OUT_ROOT = racine des dumps ; surchargeable
#                pour les tests à blanc hors VM)
#
#   Exemples :
#     bash scripts/g2_3_sweep.sh                             # sweep par défaut (none + 11)
#     bash scripts/g2_3_sweep.sh "mlp"                       # re-run déterminisme (Task 12.1)
#     KEEP=1 bash scripts/g2_3_sweep.sh "qkv_proj,mlp,ple"   # run combiné D* (conservé)
#     RUN_PREFIX=g2_3_s49 FIXTURE=/data/gemma4-zml-probe/gen_custom.safetensors REF_A=none \
#       bash scripts/g2_3_sweep.sh "none fam1,fam2 fam1 fam2"   # stabilité S49 (§7)
#
# PRÉREQUIS (vérifiés au départ quand c'est possible) : seuils de sanité calibrés
# (`52_g2_3_analyze.py --calibrate-sanity`, §5.1, AVANT tout run) ; scripts 52/53 + fixtures
# rsyncés depuis M1 (§8.3) ; swap actif + patch pjrt.zig (cf scripts/smoke.sh).
#
# CONTRAT DE FIABILITÉ (§8.2) : le succès d'un run = code de sortie 0 du runner **ET** fraîcheur
# du fichier logits (mtime postérieur à un marqueur posé avant le lancement) — JAMAIS sa seule
# existence : le runner ne crée le fichier qu'APRÈS la compile XLA, donc un run échoué au
# compile laisse l'ancien dump intact (fichier présent, run pourtant raté).
set -euo pipefail

# Filet pour les échecs NON guardés (les étapes attendues ont leurs messages fatal() dédiés).
trap 'echo "[g2_3_sweep] ABORT ligne $LINENO : « $BASH_COMMAND » a échoué — sweep interrompu, rien n a été purgé pour le run en cours" >&2' ERR

fatal() { echo "[g2_3_sweep] FATAL: $*" >&2; exit 1; }

ZML_WS=${ZML_WS:-/data/rqz_workspace/zml}
DATA=${DATA:-/data/gemma4-zml-probe}
OUT_ROOT=${OUT_ROOT:-/data}
FIXTURE=${FIXTURE:-$DATA/gen_long.safetensors}
RUN_PREFIX=${RUN_PREFIX:-g2_3}
REF_A=${REF_A:-$DATA/g2_logits_a_f32.npy}
KEEP=${KEEP:-0}
MIN_FREE_GB=6   # 1 dump ≈ 1,07 Go + dump HLO texte + marge (§8.2)

MODEL=$DATA/weights/model.safetensors
MANIFEST=$DATA/fixtures/g2_3_manifest.json
EXPECTED=$DATA/fixtures/g2_3_expected_converts.json
ENVELOPE=$DATA/fixtures/g2_envelope_manifest.json
ANALYZE=$DATA/scripts/52_g2_3_analyze.py
HLOCHECK=$DATA/scripts/53_g2_3_hlo_check.py
BIN=$ZML_WS/bazel-bin/examples/rqz/gemma4_g23_sweep
LOGS=$DATA/logs
D0_LOGITS=$OUT_ROOT/${RUN_PREFIX}_logits_none.bin
D0_HLO=$OUT_ROOT/${RUN_PREFIX}_hlo_none

# 'none' en PREMIER (D0 = référence, jamais purgé) ; kv_store EXCLU du défaut (cf en-tête).
FAMILIES=${1:-"none qkv_proj qk_scores pv_ctx o_proj mlp ple head norms softmax rope softcap"}

mkdir -p "$LOGS" "$DATA/fixtures"

# ---- préflights : échouer en 1 ms plutôt qu'après un build/run de plusieurs minutes
[ -f "$MODEL" ] || fatal "checkpoint absent : $MODEL"
[ -f "$FIXTURE" ] || fatal "fixture absente : $FIXTURE (générer côté M1 — script 46/49 — puis rsync, §8.3)"
{ [ -f "$ANALYZE" ] && [ -f "$HLOCHECK" ]; } \
  || fatal "scripts 52/53 absents sous $DATA/scripts — la VM n'est PAS un clone git : rsync depuis M1 (§8.3)"
[ -f "$EXPECTED" ] || fatal "table des converts attendus absente : $EXPECTED — rsync depuis M1 (§8.3)"
if [ "$REF_A" != "none" ]; then
  [ -f "$REF_A" ] || fatal "bras A absent : $REF_A (régénérer via script 50) — ou REF_A=none pour le mode S49"
  [ -f "$ENVELOPE" ] || fatal "enveloppe absente : $ENVELOPE — rsync depuis M1 (§8.3)"
fi
# Une VALEUR de seuil, pas la clé seule : '"sanity_thresholds": null' (calibration avortée après
# l'enregistrement de la custody) passerait un grep sur la clé et n'échouerait qu'au 52 — APRÈS
# le build et le run D0 (nuit GPU perdue).
grep -q '"entropy_mean_min"' "$MANIFEST" 2>/dev/null \
  || fatal "seuils de sanité absents/incomplets dans $MANIFEST — lancer d'abord \`52_g2_3_analyze.py --calibrate-sanity\` (§5.1, AVANT tout run du sweep)"
# Footgun S49 : REF_A=none avec le namespace par défaut pré-wiperait puis écraserait le dump et
# le HLO de D0-S46 (la custody du 52 rattraperait en exit 2, mais après la nuit GPU, et D0-S46
# serait à régénérer).
if [ "$REF_A" = "none" ] && [ "$RUN_PREFIX" = "g2_3" ]; then
  fatal "REF_A=none exige un RUN_PREFIX dédié (ex: RUN_PREFIX=g2_3_s49) — protection de D0-S46 (§7)"
fi

# ---- 0. build UNIQUE + provenance (§8.1)
echo "[g2_3_sweep] $(date +%Y-%m-%dT%H:%M:%S) build unique gemma4_g23_sweep (cuda) ..."
(cd "$ZML_WS" && ./bazel.sh build //examples/rqz:gemma4_g23_sweep --@zml//platforms:cuda=true) \
  || fatal "build Bazel en échec — swap actif ? patch pjrt.zig (@setEvalBranchQuota) présent ? (cf scripts/smoke.sh)"
[ -x "$BIN" ] || fatal "binaire absent après build : $BIN"
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
ZML_REV=$(git -C "$ZML_WS" rev-parse HEAD 2>/dev/null || echo "n/a")
REPO_REV=${REPO_REV:-$(git -C "$DATA" rev-parse HEAD 2>/dev/null || echo "n/a")}
echo "[g2_3_sweep] BIN_SHA=$BIN_SHA ZML_REV=$ZML_REV REPO_REV=$REPO_REV"
echo "[g2_3_sweep] configs: $FAMILIES"
echo "[g2_3_sweep] RUN_PREFIX=$RUN_PREFIX FIXTURE=$FIXTURE REF_A=$REF_A KEEP=$KEEP"

# Pas de mélange de binaires inter-invocations : si cette invocation NE régénère PAS D0 (pas de
# 'none' dans les configs — re-run déterminisme, run combiné KEEP=1...), le D0 réutilisé doit
# venir du MÊME binaire — sinon les runs nouveau-binaire seraient comparés à un D0 ancien-binaire
# (le 53 finirait en INVALID, mais après la nuit GPU). BIN_SHA n'est une baseline QUE par
# invocation ; la provenance de D0 vit dans le manifest (entrée ${RUN_PREFIX}_none du 52).
has_none=0
# shellcheck disable=SC2086
for f in $FAMILIES; do if [ "$f" = "none" ]; then has_none=1; fi; done
if [ "$has_none" -eq 0 ]; then
  d0_sha=$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m.get("runs",{}).get(sys.argv[2],{}).get("provenance",{}).get("bin_sha",""))' \
    "$MANIFEST" "${RUN_PREFIX}_none" 2>/dev/null || echo "")
  [ -n "$d0_sha" ] || fatal "D0 (${RUN_PREFIX}_none) absent du manifest — lancer d'abord la config 'none' (référence du sweep)"
  [ "$d0_sha" = "$BIN_SHA" ] \
    || fatal "le binaire a changé depuis D0 (manifest bin_sha=$d0_sha ≠ courant $BIN_SHA) — relancer 'none' d'abord (D0 doit venir du même binaire que les runs)"
fi

# shellcheck disable=SC2086  # split volontaire : FAMILIES = configs séparées par des espaces
for fam in $FAMILIES; do
  # binaire INCHANGÉ avant CHAQUE run (§8.1 : provenance du sweep entier = un seul binaire)
  [ "$(sha256sum "$BIN" | cut -d' ' -f1)" = "$BIN_SHA" ] \
    || fatal "binaire modifié en cours de sweep (sha256 ≠ $BIN_SHA) — provenance rompue : re-builder et relancer le sweep entier"
  # espace disque avant chaque run (§8.2)
  free_gb=$(df --output=avail -BG "$OUT_ROOT" | tail -1 | tr -dc '0-9')
  [[ "$free_gb" =~ ^[0-9]+$ ]] || fatal "sortie df illisible pour $OUT_ROOT (free_gb='$free_gb')"
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || fatal "espace insuffisant sur $OUT_ROOT (${free_gb}G < ${MIN_FREE_GB}G) — purger avant de relancer"

  safe=${fam//,/+}                                 # nom de fichier des configs combinées
  hlo_dir=$OUT_ROOT/${RUN_PREFIX}_hlo_$safe
  logits=$OUT_ROOT/${RUN_PREFIX}_logits_$safe.bin
  run_log=$LOGS/${RUN_PREFIX}_run_$safe.log
  hlo_report=$LOGS/hlo_${RUN_PREFIX}_$safe.json
  marker=$LOGS/.${RUN_PREFIX}_t0_$safe

  if [ "$fam" != "none" ]; then
    [ -f "$D0_LOGITS" ] || fatal "D0 absent ($D0_LOGITS) — lancer d'abord la config 'none' (référence du sweep)"
    [ -d "$D0_HLO" ] || fatal "dump HLO de D0 absent ($D0_HLO) — lancer d'abord la config 'none'"
  fi

  # Pré-wipe des sorties du run COURANT : un dossier HLO résiduel (KEEP=1 antérieur, re-run)
  # mélangerait les modules de DEUX compiles → diff/comptage du 53 faussés en silence. Re-runner
  # 'none' régénère donc la référence (déterminisme : même md5 attendu, custody du 52 vérifie).
  rm -rf "$hlo_dir"
  rm -f "$logits"

  # Marqueur de fraîcheur posé AVANT le run (contrat §8.2, cf en-tête) : à la fin on exige
  # logits présent ET mtime > marqueur — l'existence seule ne prouve rien.
  touch "$marker"
  t_start=$(date +%Y-%m-%dT%H:%M:%S)
  echo "=== [$t_start] RUN '$fam' → $logits (BIN_SHA=$BIN_SHA)" | tee "$run_log"

  # Dumps HLO en TEXTE seulement : PAS de --xla_dump_hlo_as_proto — les .pb embarquent un id
  # de compilation unique → faux « differs » au diff vs D0 (caveat du script 53, qui ne
  # consomme que les *before_optimizations* texte).
  rc=0
  XLA_FLAGS="--xla_dump_to=$hlo_dir" "$BIN" \
    "$MODEL" "$FIXTURE" "$logits" "$fam" \
    2>&1 | tee -a "$run_log" || rc=$?
  t_end=$(date +%Y-%m-%dT%H:%M:%S)
  echo "=== [$t_end] FIN run '$fam' (rc=$rc, début $t_start)" | tee -a "$run_log"
  [ "$rc" -eq 0 ] || fatal "runner en échec sur '$fam' (rc=$rc) — voir $run_log (compile XLA ? OOM/swap exit 255 ? famille inconnue — espaces dans la liste ?)"
  { [ -f "$logits" ] && [ "$logits" -nt "$marker" ]; } \
    || fatal "logits absent ou PÉRIMÉ ($logits, marqueur $marker) : runner sorti 0 sans dump frais — run invalide, ne pas analyser"
  rm -f "$marker"

  # md5 de CHAQUE dump AVANT purge (déterminisme Task 12.1) — format md5sum pur, ne pas décorer.
  md5sum "$logits" | tee -a "$LOGS/${RUN_PREFIX}_md5.log"

  if [ "$fam" = "none" ]; then
    # D0 = référence seule : custody (md5 A si fournie + D0) + sanité via --register-reference —
    # pas d'auto-analyse vs soi-même, pas de check HLO vs soi-même (52, docstring mode).
    rc=0
    python3 "$ANALYZE" --run-logits "$logits" --run-name "${RUN_PREFIX}_none" \
      --ref-a "$REF_A" --ref-d0 "$logits" --register-reference \
      --manifest "$MANIFEST" \
      --bin-sha "$BIN_SHA" --zml-rev "$ZML_REV" --repo-rev "$REPO_REV" || rc=$?
    if [ "$rc" -eq 2 ]; then
      fatal "custody REFUSÉE (52, exit 2) : références corrompues — RÉGÉNÉRER (script 50 pour A/B, run 'none' pour D0), jamais réutiliser (§2)"
    fi
    [ "$rc" -eq 0 ] || fatal "enregistrement de D0 (52 --register-reference) en échec (rc=$rc)"
    continue   # D0 JAMAIS purgé : dump + HLO servent de référence à tous les runs suivants
  fi

  # Anti-câblage-croisé (53) : exit 0 même si verdict INVALID/IDENTICAL (le verdict EST le
  # résultat, c'est le 52 qui le consigne) ; exit ≠ 0 = erreur d'usage/structure → abort.
  rc=0
  python3 "$HLOCHECK" --run-dir "$hlo_dir" --d0-dir "$D0_HLO" \
    --family "$fam" --expected "$EXPECTED" --out "$hlo_report" || rc=$?
  [ "$rc" -eq 0 ] || fatal "check HLO (53) en échec sur '$fam' (rc=$rc = erreur de structure, PAS un verdict) — dump pré-opt absent ? famille inconnue ?"

  # Analyse (52) — mode complet (custody A + vs D0 + vs A + buckets vs enveloppe) ; en S49
  # (REF_A=none) le 52 exige run-logits/run-name/ref-d0/manifest mais PAS --envelope (il le
  # refuse implicitement : pas de buckets sans enveloppe, verdict diagnostic-only §7) —
  # --expected-converts/--hlo-report restent fournis : la gate §5.3 vaut aussi en S49.
  args_52=( --run-logits "$logits" --run-name "${RUN_PREFIX}_$safe"
            --ref-a "$REF_A" --ref-d0 "$D0_LOGITS"
            --expected-converts "$EXPECTED" --hlo-report "$hlo_report"
            --manifest "$MANIFEST"
            --bin-sha "$BIN_SHA" --zml-rev "$ZML_REV" --repo-rev "$REPO_REV" )
  if [ "$REF_A" != "none" ]; then
    args_52+=( --envelope "$ENVELOPE" )
  fi
  rc=0
  python3 "$ANALYZE" "${args_52[@]}" || rc=$?
  if [ "$rc" -eq 2 ]; then
    fatal "custody REFUSÉE (52, exit 2) : références corrompues — RÉGÉNÉRER (script 50 pour A/B, run 'none' pour D0), jamais réutiliser (§2)"
  fi
  [ "$rc" -eq 0 ] || fatal "analyse (52) en échec sur '$fam' (rc=$rc) — verdict NON consigné, corriger avant de poursuivre"

  # Purge au fil de l'eau (§8.2) — le npz de métriques (écrit par le 52 à côté du manifest),
  # le rapport HLO json et le md5 log survivent ; KEEP=1 conserve dump + HLO (run D*).
  if [ "$KEEP" != "1" ]; then
    rm -f "$logits"
    rm -rf "$hlo_dir"
    echo "[g2_3_sweep] purge '$fam' : dump + HLO supprimés (md5 consigné dans ${RUN_PREFIX}_md5.log)"
  fi
done

echo "[g2_3_sweep] SWEEP DONE $(date +%Y-%m-%dT%H:%M:%S) — manifest: $MANIFEST ; md5: $LOGS/${RUN_PREFIX}_md5.log"
