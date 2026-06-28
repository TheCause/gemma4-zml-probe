#!/usr/bin/env bash
# smoke.sh — Test de fumée : compile (sans exécuter) les runners ZML clés sur la 3090.
#
# But (R5) : vérifier rapidement que la toolchain ZML + les sources compilent, SANS avoir besoin des
# weights ni des fixtures ni d'un run (le mur mémoire n'est pas touché en build-only). C'est la
# vérification minimale de reproductibilité après un changement de source (ex: les livrables R1/R2).
#
# Inclut les runners clés : non-régression (E1/E2), gen-long (gchunk/vacuity/ring/auto), GPU (G1/bench).
#   - gemma4_engine_e1   (socle mono, config par défaut — la base de preuve HLO)
#   - gemma4_engine_e2   (brique TurboQuant)
#   - gemma4_gchunk      (L1a chunké + instrumentation mémoire R1)
#   - gemma4_gchunk_vacuity (contre-test non-vacuité R2)
#
# Usage (sur la 3090, depuis le workspace ZML) :
#   bash /data/gemma4-zml-probe/scripts/smoke.sh
#   SMOKE_TARGETS="gemma4_engine_e1 gemma4_gchunk_vacuity" bash scripts/smoke.sh   # sous-ensemble
set -uo pipefail

ZML_WS="${ZML_WS:-/data/rqz_workspace/zml}"
BAZEL="./bazel.sh"
TARGETS=${SMOKE_TARGETS:-"gemma4_engine_e1 gemma4_engine_e2 gemma4_gchunk gemma4_gchunk_vacuity gemma4_gchunk_ring gemma4_gchunk_auto gemma4_gen_long_gpu gemma4_bench"}

cd "$ZML_WS" || { echo "ERR: workspace ZML introuvable: $ZML_WS"; exit 1; }
# Prérequis swap (OOM compile, cf GENERATION_LONGUE_PLAN conventions) :
if swapon --show 2>/dev/null | grep -q .; then :; else
  echo "WARN: aucun swap actif — le compile XLA-CPU peut OOM-killer (exit 255). Vérifier /swapfile_xla."
fi

ok=0; fail=0
for t in $TARGETS; do
  printf "  [build %-26s] " "//examples/rqz:$t"
  if $BAZEL build "//examples/rqz:$t" >/tmp/smoke_$t.log 2>&1; then
    echo "OK"; ok=$((ok+1))
  else
    echo "FAIL (cf /tmp/smoke_$t.log)"; fail=$((fail+1))
  fi
done
echo "==============================="
echo "BUILD OK=$ok  FAIL=$fail"
[ $fail -eq 0 ] && echo "Smoke OK — sources + toolchain compilent (runners non exécutés)."
exit $fail
