#!/usr/bin/env bash
set -euo pipefail

# Génère UNE fixture oracle HF par prompt d'un jeu de banc (gate B2 du chantier batching).
#
# Pourquoi N fixtures mono et pas un run HF batché : la référence de fidélité doit rester
# HF mono B=1 par séquence (un run HF batché appliquerait un padding qui changerait la
# numérique de la RÉFÉRENCE elle-même — spec §4).
#
# --n-tokens 48 est FIGÉ (le défaut de 49 est 200) : le critère pré-enregistré B2 est 48/48
# et le bras B=1 du protocole B4 suppose cette longueur. 48 steps bornent aussi l'exposition
# aux bifurcations d'argmax légitimes (ties, GEMM différentes — spec §7).
#
# À exécuter sur la machine GPU (HF offline, cache HF local). Le venv du projet porte torch +
# transformers : le passer par PY (le python système ne les a pas).
#
# Usage : PY=<venv>/bin/python3 ./60_batch_oracles.sh <prompts.txt> <out_dir>
#   ex.  : PY=/path/to/venv/bin/python3 ./60_batch_oracles.sh fixtures/bench_prompts_b4.txt ./batch_fixtures

PROMPTS="${1:?usage: 60_batch_oracles.sh <prompts.txt> <out_dir>}"
OUT_DIR="${2:?usage: 60_batch_oracles.sh <prompts.txt> <out_dir>}"
PY="${PY:-python3}"
N_TOKENS=48

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT_DIR"

i=0
while IFS= read -r prompt; do
  [ -z "$prompt" ] && continue
  out="$OUT_DIR/oracle_lane${i}.safetensors"
  echo "=== lane $i : $prompt"
  "$PY" "$SCRIPT_DIR/49_gen_custom_oracle.py" \
    --prompt "$prompt" \
    --n-tokens "$N_TOKENS" \
    --out "$out"
  i=$((i + 1))
done < "$PROMPTS"

echo
echo "=== $i fixtures écrites dans $OUT_DIR"
echo "=== vérification des seq_len (doivent être IDENTIQUES — contrainte V1 de positions uniformes) :"
for m in "$OUT_DIR"/oracle_lane*.safetensors.manifest.json; do
  "$PY" -c "import json,sys; d=json.load(open('$m')); print(f\"  {d['seq_len']:3d} tok  n_decode={d['n_decode']}  {d['prompt']!r}\")"
done
