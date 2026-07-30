#!/usr/bin/env bash
set -euo pipefail

# Build canonique du runner sur la machine de compute — SOURCE UNIQUE de la commande.
#
# ⚠ POURQUOI CE SCRIPT EXISTE (docs/MODE_BUILD_AUDIT.md) : `-c opt` seul ne règle QUE le backend
# C++. Le mode du frontend Zig est un flag Bazel INDÉPENDANT (`--@rules_zig//zig/settings:mode`,
# défaut `debug`). Une mesure faite sans ce second flag chronomètre du code Zig NON optimisé —
# c'est ce qui a produit un artefact ×8,7 sur les warpers du sampling (dette D9, rectifiée par
# D10). Le binaire publie son mode réel au démarrage (`BUILD: mode=…`) : un log de gate qui
# n'affiche pas `mode=ReleaseFast` est INEXÉCUTABLE, pas PASS.
#
# Configure via variables d'environnement (mêmes conventions que deploy_to_3090.sh) :
#   ZML_REMOTE  user@host de la machine de compute
#   ZML_JUMP    jump host SSH optionnel, vide = direct
#   ZML_WS      chemin du workspace ZML sur la machine distante
#   TARGETS     cibles bazel (défaut : le runner 12B + la variante 4k)
#
# Exemple :
#   ZML_REMOTE=me@gpu-box ZML_WS=/data/zml ./build_3090.sh

ZML_REMOTE="${ZML_REMOTE:-user@gpu-host}"
ZML_JUMP="${ZML_JUMP:-}"
ZML_WS="${ZML_WS:-/path/to/zml}"
TARGETS="${TARGETS:-//examples/rqz:gemma4_g12auto //examples/rqz:gemma4_g12a4k}"

SSH_OPT=()
if [ -n "$ZML_JUMP" ]; then SSH_OPT=(-J "$ZML_JUMP"); fi

# Les DEUX flags de mode + CUDA. Ne pas en retirer un « pour aller plus vite ».
BUILD_CMD="cd '$ZML_WS' && ./bazel.sh build \
  -c opt \
  --@rules_zig//zig/settings:mode=release_fast \
  --@zml//platforms:cuda=true \
  $TARGETS"

echo "== Commande canonique =="
echo "$BUILD_CMD"
echo
ssh "${SSH_OPT[@]}" "$ZML_REMOTE" "$BUILD_CMD"
echo
echo "OK. Rappel : vérifier 'BUILD: mode=ReleaseFast' dans le log de CHAQUE run de gate."
