#!/usr/bin/env bash
# sweep_perf.sh — Caractérisation du trade-off perf/mémoire du decode chunké (R4, cf analyse point 10).
#
# Le decode chunké a 2 leviers comptime antagonistes (cf docs/GENERATION_LONGUE_CHUNKING_DESIGN.md §5) :
#   - CHUNK       : couches/stage. Plus grand = moins de stages = moins de syncs host/step MAIS pic
#                  compilation/stage plus gros (plus de poids f32 coexistant).
#   - SYNC_EVERY  : fréquence de sync (toSliceAlloc) entre stages. Défaut 1 = sync après chaque stage
#                  (mémoire bornée). Plus grand = moins de round-trips host MAIS working sets qui
#                  s'accumulent (le risque que le design §5 point 2 a flaggé comme « inconnue centrale »).
#
# CHUNK est NON-MONOTONE (design §5 point 2) : trop petit = pic/stage faible MAIS plus d'exe résidents
# → peut AGGRAVER le pic dominant. Ce sweep cherche l'optimum (pic <23 Go ET run rapide).
#
# Méthode : pour chaque (CHUNK, SYNC_EVERY) on patch les 2 consts comptime du runner, on build+run
# sur la 3090, on capture : temps total, RSS post-compile (go/no-go), RSS/swap finaux, match count.
# On restore le fichier après chaque config (le runner de référence doit rester CHUNK=5/SYNC_EVERY=1).
#
# ATTENTION : ce script PATCH gemma4_gchunk.zig en place (sur la 3090). Il le restore en finally.
# Les configs sont hardcodées (petite grille) — édite CONFIGS pour balayer plus large.
#
# Usage (sur la 3090, depuis le workspace ZML) :
#   cd /data/rqz_workspace/zml
#   bash /data/gemma4-zml-probe/scripts/sweep_perf.sh <model.safetensors> <gen_long.safetensors> [max_steps]
# Exemple : bash .../sweep_perf.sh weights/model.safetensors gen_long.safetensors 64   # grille courte
set -euo pipefail

CKPT="${1:-/data/gemma4-zml-probe/weights/model.safetensors}"
FIXTURE="${2:-/data/gemma4-zml-probe/gen_long.safetensors}"
MAXSTEPS="${3:-}"   # optionnel : cappe le run (ex: 64) pour itérer vite sur la grille

# Grille (CHUNK SYNC_EVERY). Garder petite : chaque config = 1 build (~min) + 1 run.
# CHUNK doit diviser 15 (pour ne pas couper un producer/reader au milieu) → 3,5,15. 7 ne divise pas 15.
CONFIGS=(
  "5 1"     # référence L1a (baseline)
  "3 1"     # +petit CHUNK : moins de pic/stage, +d'exe résidents (test non-monotonie)
  "7 1"     # +grand CHUNK : -de syncs, +gros pic/stage (7 ne divise pas 15 → stage mixte, à observer)
  "15 1"    # CHUNK=15 : 1 stage producteur + ... (très gros pic, borne haute)
  "5 2"     # SYNC_EVERY=2 : -de syncs, working sets s'accumulent (l'inconnue §5.2)
  "5 7"     # SYNC_EVERY=7 : sync seulement au dernier stage = max mémoire, min round-trips
)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER="$ROOT/zml_runner/gemma4_gchunk.zig"
ZML_WS="${ZML_WS:-/data/rqz_workspace/zml}"
BAZEL="./bazel.sh"
# les sources sont déployées dans examples/rqz/ (cf deploy_to_3090.sh) — on patch la copie déployée
DEPLOYED="${ZML_WS}/examples/rqz/gemma4_gchunk.zig"
[ -f "$DEPLOYED" ] || { echo "ERR: runner déployé introuvable: $DEPLOYED (lance deploy_to_3090.sh d'abord)"; exit 1; }

# Restaure toujours la copie déployée à la fin (le git local reste intact : on patch $DEPLOYED, pas $RUNNER).
cleanup() { cp "$RUNNER" "$DEPLOYED" 2>/dev/null || true; }
trap cleanup EXIT

run_args="$CKPT $FIXTURE"
[ -n "$MAXSTEPS" ] && : # gchunk ne supporte pas max_steps natif (run = len expected) ; pour un run court
                        # on tronque `expected` côté fixture. Ici on run complet — voir note ci-dessous.

printf "%-12s %-10s %-14s %-14s %-14s %-10s\n" "CHUNK" "SYNC_EVERY" "RSS_postcomp_GiB" "swap_final_GiB" "RSS_final_GiB" "match"
echo "----------------------------------------------------------------------------------------"

for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; CHUNK=$1; SE=$2
  # patch les 2 consts comptime (lignes `const CHUNK: usize = ...` / `const SYNC_EVERY: usize = ...`)
  sed -E -i.bak \
    -e "s/^const CHUNK: usize = [0-9]+;/const CHUNK: usize = ${CHUNK};/" \
    -e "s/^const SYNC_EVERY: usize = [0-9]+;/const SYNC_EVERY: usize = ${SE};/" \
    "$DEPLOYED"
  rm -f "${DEPLOYED}.bak"

  echo ">>> build CHUNK=$CHUNK SYNC_EVERY=$SE ..."
  ( cd "$ZML_WS" && $BAZEL build //examples/rqz:gemma4_gchunk >/dev/null 2>&1 ) || {
    echo "    BUILD FAIL (CHUNK=$CHUNK SE=$SE) — stage mixte (7 ne divise pas 15) ?"; 
    printf "%-12s %-10s %-14s %-14s %-14s %-10s\n" "$CHUNK" "$SE" "BUILD_FAIL" "-" "-" "-"
    cp "$RUNNER" "$DEPLOYED"; continue
  }

  LOG=$(mktemp)
  t0=$(date +%s)
  ( cd "$ZML_WS" && bazel-bin/examples/rqz/gemma4_gchunk $run_args ) >"$LOG" 2>&1 || true
  t1=$(date +%s)
  dt=$((t1 - t0))

  # parse la sortie instrumentée (mem_probe logs : "[mem] tag: RSS=... KiB (... GiB) swap=... KiB (... GiB)")
  postcomp=$(grep -m1 '\[mem\] post-compile' "$LOG" | grep -oE 'RSS=[0-9]+ KiB \(~?[0-9.]+ GiB\)' | grep -oE '\(~?[0-9.]+ GiB' | tr -d '(~ ' || echo "?")
  final=$(grep -m1 '\[mem\] post-run' "$LOG" | grep -oE 'RSS=[0-9]+ KiB \(~?[0-9.]+ GiB\)' | grep -oE '\(~?[0-9.]+ GiB' | tr -d '(~ ' || echo "?")
  swapf=$(grep -m1 '\[mem\] post-run' "$LOG" | grep -oE 'swap=[0-9]+ KiB \(~?[0-9.]+ GiB\)' | grep -oE '\(~?[0-9.]+ GiB' | tr -d '(~ ' || echo "?")
  match=$(grep -oE 'L1a CHUNKÉ : [0-9]+/[0-9]+ tokens match' "$LOG" | grep -oE '[0-9]+/[0-9]+' || echo "?")
  verdict=$(grep -m1 'L1a CHUNKÉ PASS\|divergence vs expected' "$LOG" | head -c 60 || echo "?")

  printf "%-12s %-10s %-14s %-14s %-14s %-10s  (%ds, %s)\n" "$CHUNK" "$SE" "${postcomp}GiB" "${swapf}GiB" "${final}GiB" "$match" "$dt" "$verdict"
  cp "$RUNNER" "$DEPLOYED"   # restore pour la prochaine config
done

echo "----------------------------------------------------------------------------------------"
echo "Lecture : RSS_postcomp = pic go/no-go (doit <23 GiB). match doit être N/N (équivalence préservée)."
echo "          CHUNK=5/SYNC_EVERY=1 = baseline L1a. swap_final > 0 = fuite résiduelle (R1) ; compare les configs."
echo "Note : run complet 1020 steps (~55 min baseline). Pour itérer vite, tronque la fixture `expected`."
