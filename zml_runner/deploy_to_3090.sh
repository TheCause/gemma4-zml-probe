#!/usr/bin/env bash
set -euo pipefail

# Deploy gemma4-zml-probe Zig runner sources into the ZML workspace on 3090.
# Source canonique : ~/dev/gemma4-zml-probe/zml_runner/ (M1)
# Cible compile    : /data/rqz_workspace/zml/examples/rqz/ (3090)

ZML_REMOTE="user@gpu-host"
ZML_JUMP="macmini"
ZML_DST="/data/rqz_workspace/zml/examples/rqz"

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

ssh -J "$ZML_JUMP" "$ZML_REMOTE" "mkdir -p '$ZML_DST'"

rsync -av \
  --delete \
  --exclude 'deploy_to_3090.sh' \
  -e "ssh -J $ZML_JUMP" \
  "$SRC_DIR/" \
  "$ZML_REMOTE:$ZML_DST/"

echo "Deployed $SRC_DIR/ -> $ZML_REMOTE:$ZML_DST/"
