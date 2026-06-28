#!/usr/bin/env bash
# regen_fixtures.sh — Régénère TOUTES les fixtures (gitignorées) depuis les oracles Python, dans l'ordre
# des dépendances des gates (R5, cf analyse point 10).
#
# Les fixtures (.safetensors/.npy/.pt) sont gitignorées (régénérables) : un clone frais ne peut rien
# exécuter sans elles. Ce script orchestre la chaîne P-1 → P5.7.8 + TurboQuant + génération longue,
# dans l'ordre imposé par les gates (chacun consomme la sortie du précédent).
#
# Cible : la 3090 (venv gemma4-probe, weights/, HF cache). Lance depuis le repo sur la 3090.
#
# Usage :
#   bash scripts/regen_fixtures.sh             # tout, dans l'ordre
#   bash scripts/regen_fixtures.sh ple          # une phase : ple|yoco|p52|p54-p56|p57|decode|tq|genlong
#   bash scripts/regen_fixtures.sh p52 p57      # plusieurs phases
#
# Prérequis : venv actif (transformers 5.9.0, torch 2.12.0), HF_TOKEN ou hf login, weights/model.safetensors,
#            HF_HOME=/data/hf_cache. Les scripts GPU (46 gen_long, 33 gen_vq_measure) basculent en cuda auto.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

ok=0; fail=0; failed_list=()
run() {  # run <label> <script>
  local label="$1" script="$2"
  if [ ! -f "scripts/$script" ]; then echo "  [SKIP] $label : $script absent"; return; fi
  printf "  [%-34s] %s ... " "$label" "$script"
  if $PY "scripts/$script" >/tmp/regen_$$.log 2>&1; then
    echo "PASS"; ok=$((ok+1))
  else
    echo "FAIL (cf /tmp/regen_$$.log)"; fail=$((fail+1)); failed_list+=("$script")
  fi
}

# Phases (ordre des dépendances). Les labels font foi dans PLANNING/ENGINE_LOG.
phase_ple() {
  run "P4-prep contract PLE"        02_contract_ple.py
  run "P3 PLE reference"            03_ple_reference.py
  run "P4-prep PLE fixture export"  06_export_ple_fixture.py
  run "P4.4.0 safetensors fixture"   08_export_safetensors_fixture.py
  run "P4-prep PLE raw pytorch"     05_ple_raw_pytorch.py
  run "P4.3 selfcheck"              07_fixture_selfcheck.py
}
phase_yoco() {
  run "P5.0 YOCO config map"        09_yoco_config_map.py
  run "P5.0 YOCO weight map"        10_yoco_weight_map.py
  run "P5.1 YOCO policy table"      13_yoco_policy_table.py
}
phase_p52() {
  run "P5.2.D0 KV oracle L13"       14_kv_oracle_layer13.py
  run "P5.2.D0 q-only reader"      14_q_only_reader_oracle.py
  for s in 15_p5_2_d1_export_fixture.py 16_p5_2_d2_export_fixture.py 17_p5_2_d3_export_fixture.py \
           18_p5_2_d4_export_fixture.py 19_p5_2_d5_export_fixture.py 20_p5_2_d2b_export_fixture.py \
           21_attention_oracle_layer15_reader_kv13.py 22_p5_2_e1_export_fixture.py 23_p5_2_emask_oracle.py \
           24_p5_2_esoftmax_export_fixture.py 25_p5_2_econtext_export_fixture.py 26_p5_2_f_oproj_oracle.py \
           27_p5_2_g_attn_residual_oracle.py 28_p5_2_h_mlp_oracle.py; do
    run "P5.2" "$(basename "$s")"
  done
}
phase_p54_p56() {
  run "P5.4 embed oracle"           30_p5_4_embed_oracle.py
  run "P5.5 head oracle"           31_p5_5_head_oracle.py
  run "P5.3 layer oracle"          32_p5_3_layer_oracle.py
  run "P5.6 full qrope oracle"     29_p5_6_full_qrope_oracle.py
  run "P5.6.K full krope oracle"   33_p5_6k_full_krope_oracle.py
}
phase_p57() {
  run "P5.7.0 loader manifest"     34_p5_7_0_loader_manifest.py
  run "P5.7.1 load ref"            35_p5_7_1_load_ref.py
  run "P5.7.3 runtime plan"        36_p5_7_3_runtime_plan.py
  run "P5.7.4 full layer oracle"   37_p5_7_4_full_layer_oracle.py
  run "P5.7.5 prefill oracle"      38_p5_7_5_prefill_oracle.py
  run "P5.7.5 prefill oracle HYBRIDE" 39_p5_7_5_prefill_oracle_hybrid.py
}
phase_decode() {
  run "P5.7.7 decode pilot oracle" 40_p5_7_7_decode_pilot_oracle.py
  run "P5.7.7 decode prim oracle"  41_p5_7_7_decode_prim_oracle.py
  run "P5.7.7 decode2 oracle"      42_p5_7_7_decode2_oracle.py
  run "P5.7.7 decode3 oracle"     43_p5_7_7_decode3_oracle.py
  run "P5.7.8 gen oracle"         44_p5_7_8_gen_oracle.py
}
phase_tq() {
  run "TQ Task0 export constants" 30_export_turboquant_constants.py
  run "TQ Q3 vquant oracle"       31_vquant_oracle.py
  run "TQ Q4 decode_vq oracle"    32_decode_vq_oracle.py
  run "TQ Q5 cost measure (GPU)"  33_gen_vq_measure.py
  run "TQ Q5 gen_vq oracle"       45_gen_vq_oracle.py
}
phase_genlong() {
  run "GEN-LONG L0 oracle (GPU)"  46_gen_long_oracle.py
  run "GEN-LONG L1b ring oracle"  47_gen_long_ring_oracle.py
}

ALL="ple yoco p52 p54-p56 p57 decode tq genlong"
phases=("$@"); [ ${#phases[@]} -eq 0 ] && phases=($ALL)

for ph in "${phases[@]}"; do
  echo "=== Phase : $ph ==="
  case "$ph" in
    ple)      phase_ple;;
    yoco)     phase_yoco;;
    p52)      phase_p52;;
    p54-p56)  phase_p54_p56;;
    p57)      phase_p57;;
    decode)   phase_decode;;
    tq)       phase_tq;;
    genlong)  phase_genlong;;
    *) echo "  phase inconnue: $ph (valides: $ALL)";;
  esac
done

echo "==============================="
echo "PASS=$ok  FAIL=$fail"
[ $fail -gt 0 ] && { echo "Échecs : ${failed_list[*]}"; exit 1; }
echo "Toutes les fixtures régénérées."
