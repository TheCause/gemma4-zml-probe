#!/usr/bin/env bash
set -uo pipefail

# Sweep B du banc batché (gate B4). Exécute LE PROTOCOLE PRÉ-ENREGISTRÉ de
# docs/BATCH_BENCH_PROTOCOL.md — toute déviation doit être publiée dans BATCHING_RESULTS.md.
#
# À lancer SUR la machine GPU, depuis le workspace ZML (les binaires sont dans bazel-bin/).
#
# Usage :
#   WEIGHTS=... TOKENIZER=... FIXTURES=... PROMPTS_B4=... ./61_batch_sweep.sh
#
# Custody (doctrine G2.3 §7.1) : le moteur est shape-polymorphe (gate T0) → UN SEUL binaire
# sert tous les B. Son sha256 est consigné une fois et re-vérifié avant chaque run : s'il
# change en cours de sweep, le sweep est invalide.

WEIGHTS="${WEIGHTS:?WEIGHTS manquant}"
TOKENIZER="${TOKENIZER:?TOKENIZER manquant}"
FIXTURES="${FIXTURES:?FIXTURES manquant (répertoire des oracle_lane*.safetensors)}"
PROMPTS_B4="${PROMPTS_B4:?PROMPTS_B4 manquant}"
BB=./bazel-bin/examples/rqz/gemma4_bbatch
GA=./bazel-bin/examples/rqz/gemma4_gen_auto
B_LIST="${B_LIST:-1 2 4 8 16 32}"
RUNS=3                 # 3 runs par bras, statistique = médiane (protocole §4)
VRAM_CEILING_MIB=22528 # 22 GiB : arrêt par PROJECTION, jamais par crash OOM (protocole §2)

SHA_BB=$(sha256sum "$BB" | cut -c1-16)
SHA_GA=$(sha256sum "$GA" | cut -c1-16)
echo "=== custody : gemma4_bbatch sha256=$SHA_BB ; gemma4_gen_auto sha256=$SHA_GA"

# Attend que le GPU soit VRAIMENT libre. Sans cette attente, le run suivant se fait refuser par
# la garde de contention du runner (error.GpuBusy) parce que le process précédent n'a pas fini de
# libérer sa VRAM — ce qui se lit comme un « FAIL de fidélité » alors que le run n'a jamais tourné.
# (Piège vécu au 1er sweep : B=4 rapporté FAIL, alors qu'il PASSE 4/4 en isolation.)
wait_gpu_free() {
  local waited=0
  while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -ne 0 ]; do
    sleep 2; waited=$((waited + 2))
    if [ "$waited" -gt 120 ]; then
      echo "!! CONTENTION GPU persistante (>120 s) — protocole §1, point invalide" >&2
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
      return 1
    fi
  done
  return 0
}

check_sha() {
  local now
  now=$(sha256sum "$1" | cut -c1-16)
  [ "$now" = "$2" ] || { echo "!! sha256 de $1 a changé ($now != $2) — SWEEP INVALIDE" >&2; exit 2; }
}

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }

# tok/s de GÉNÉRATION agrégée (bbatch) ou de génération (gen_auto)
gen_rate_bb() { grep -oE 'génération agrégée [0-9]+ tokens en [0-9.]+s \(([0-9.]+) tok/s\)' | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+' | head -1; }
gen_rate_ga() { grep -oE 'génération [0-9]+ tokens en [0-9.]+s \(([0-9.]+) tok/s\)' | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+' | head -1; }

# Pic VRAM réel : --no-prealloc OBLIGATOIRE (sous prealloc, nvidia-smi ne montre que la réserve
# BFC 0.90×libre — « piège 14 »), échantillonnage pendant le run, scopé au PID du runner.
measure_vram_peak() {  # $1 = commande complète (string)
  local peak=0 used pid
  eval "$1" >/tmp/sweep_vram_run.log 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
    [ -n "${used:-}" ] && [ "$used" -gt "$peak" ] 2>/dev/null && peak=$used
    sleep 0.25
  done
  wait "$pid"; local rc=$?
  echo "$peak $rc"
}

echo
echo "=== B4 — sweep. Protocole : docs/BATCH_BENCH_PROTOCOL.md"
printf '%-4s | %-10s | %-10s | %-9s | %-9s | %s\n' "B" "agrégé" "par lane" "pic MiB" "compile" "fidélité"
echo "-----|------------|------------|-----------|-----------|---------"

prev_peak=0
for B in $B_LIST; do
  wait_gpu_free || exit 1
  check_sha "$BB" "$SHA_BB"

  # Arrêt par PROJECTION (protocole §2) : pic(B) ≈ pic(B/2) + (B/2) × Δ_lane
  if [ "$prev_peak" -gt 0 ]; then
    proj=$(( prev_peak + (B / 2) * 40 ))   # Δ ≈ 40 MiB/lane (cache f32 + marge)
    if [ "$proj" -gt "$VRAM_CEILING_MIB" ]; then
      echo "=== ARRÊT : projection pic(B=$B) ≈ ${proj} MiB > plafond ${VRAM_CEILING_MIB} MiB"
      echo "=== plafond B tenu = $((B / 2))"
      break
    fi
  fi

  # Charge (protocole §3) : B<=4 => prompts distincts + oracles par lane ; B>=8 => --replicate + spot-check lane 0
  if [ "$B" -le 4 ]; then
    orc=""; for i in $(seq 0 $((B - 1))); do orc="${orc}${orc:+,}$FIXTURES/oracle_lane${i}.safetensors"; done
    head -n "$B" "$PROMPTS_B4" > /tmp/sweep_prompts.txt
    FID_ARGS="--prompts /tmp/sweep_prompts.txt --oracles $orc"
  else
    rep=$((B / 4))
    orc=""; for i in 0 1 2 3; do orc="${orc}${orc:+,}$FIXTURES/oracle_lane${i}.safetensors"; done
    FID_ARGS="--prompts $PROMPTS_B4 --replicate $rep --oracles $orc"
  fi

  # 1) run fidélité — critère CONFORME AU PROTOCOLE §3 :
  #    B<=4 : oracle par lane (toutes les lanes doivent matcher) ;
  #    B>=8 : SPOT-CHECK lane 0 (les lanes répliquées peuvent bifurquer sur un tie — cf B=8, §4).
  fid_out=$($BB "$WEIGHTS" "$TOKENIZER" $FID_ARGS 2>&1)
  if [ "$B" -le 4 ]; then
    if echo "$fid_out" | grep -q "B2 PASS"; then fid="PASS"; else fid="FAIL"; fi
  else
    if echo "$fid_out" | grep -q "B2 lane 0 : PASS"; then fid="PASS(spot)"; else fid="FAIL"; fi
  fi
  # lanes en échec (bifurcations sur tie) rapportées explicitement — jamais masquées
  nfail=$(echo "$fid_out" | grep -c "B2 lane .* : FAIL" || true)
  [ "$nfail" -gt 0 ] && fid="$fid/${nfail}bif"
  if [ "$fid" = "FAIL" ]; then echo "  !! cause du FAIL B=$B :"; echo "$fid_out" | grep -E "error" | head -3; fi
  compile=$(echo "$fid_out" | grep -oE 'compile: [0-9.]+s' | grep -oE '[0-9.]+' | head -1)

  # 2) runs perf (3×, médiane)
  rates=()
  for _ in $(seq $RUNS); do
    wait_gpu_free || exit 1
    r=$($BB "$WEIGHTS" "$TOKENIZER" $FID_ARGS 2>&1 | gen_rate_bb)
    rates+=("${r:-0}")
  done
  wait_gpu_free || exit 1
  agg=$(median "${rates[@]}")
  per_lane=$(awk -v a="$agg" -v b="$B" 'BEGIN{printf "%.1f", a/b}')

  # 3) pic VRAM (--no-prealloc, run long 999 : 19 + 999 = 1018 <= L_MAX)
  res=$(measure_vram_peak "$BB '$WEIGHTS' '$TOKENIZER' $FID_ARGS --no-prealloc --max-tokens 999")
  peak=$(echo "$res" | cut -d' ' -f1)
  prev_peak=$peak

  printf '%-4s | %-10s | %-10s | %-9s | %-9s | %s\n' "$B" "$agg" "$per_lane" "$peak" "${compile:-?}" "$fid"
done

# Bras apparié gen_auto (B=1) — runs FRAIS dans la MÊME fenêtre de session (protocole §4)
echo
echo "=== non-régression : bras appariés gen_auto vs bbatch à B=1 (3 runs, médiane, seuil 0,95×)"
check_sha "$GA" "$SHA_GA"
ga_rates=(); bb_rates=()
head -n 1 "$PROMPTS_B4" > /tmp/sweep_p1.txt
for _ in $(seq $RUNS); do
  wait_gpu_free || exit 1
  r=$($GA "$WEIGHTS" "$TOKENIZER" --oracle "$FIXTURES/oracle_lane0.safetensors" --prompt "$(head -1 "$PROMPTS_B4")" 2>&1 | gen_rate_ga)
  ga_rates+=("${r:-0}")
  wait_gpu_free || exit 1
  r=$($BB "$WEIGHTS" "$TOKENIZER" --prompts /tmp/sweep_p1.txt --oracles "$FIXTURES/oracle_lane0.safetensors" 2>&1 | gen_rate_bb)
  bb_rates+=("${r:-0}")
done
med_ga=$(median "${ga_rates[@]}"); med_bb=$(median "${bb_rates[@]}")
echo "  gen_auto B=1 : runs=${ga_rates[*]}  médiane=$med_ga tok/s"
echo "  bbatch   B=1 : runs=${bb_rates[*]}  médiane=$med_bb tok/s"
awk -v bb="$med_bb" -v ga="$med_ga" 'BEGIN{
  seuil = 0.95 * ga;
  printf "  critère : médiane(bbatch) >= 0,95 x médiane(gen_auto) = %.1f tok/s\n", seuil;
  printf "  VERDICT : %s (ratio %.3f)\n", (bb >= seuil ? "PASS" : "FAIL"), (ga > 0 ? bb/ga : 0);
}'
