#!/usr/bin/env bash
set -euo pipefail

# Deploy the ZML runner sources into a ZML workspace on a remote compute host.
# Configure via environment variables (or edit the defaults below):
#   ZML_REMOTE  user@host of the compute machine          (e.g. user@gpu-host)
#   ZML_JUMP    optional SSH jump host, empty = direct     (e.g. bastion)
#   ZML_DST     path of the ZML `examples/rqz/` dir on the remote
#
# Example:
#   ZML_REMOTE=me@gpu-box ZML_DST=/data/zml/examples/rqz ./deploy_to_3090.sh

ZML_REMOTE="${ZML_REMOTE:-user@gpu-host}"
ZML_JUMP="${ZML_JUMP:-}"
ZML_DST="${ZML_DST:-/path/to/zml/examples/rqz}"

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

SSH_OPT=()
RSYNC_SSH="ssh"
if [ -n "$ZML_JUMP" ]; then
  SSH_OPT=(-J "$ZML_JUMP")
  RSYNC_SSH="ssh -J $ZML_JUMP"
fi

ssh "${SSH_OPT[@]}" "$ZML_REMOTE" "mkdir -p '$ZML_DST'"

rsync -av \
  --delete \
  --exclude 'deploy_to_3090.sh' \
  -e "$RSYNC_SSH" \
  "$SRC_DIR/" \
  "$ZML_REMOTE:$ZML_DST/"

echo "Deployed $SRC_DIR/ -> $ZML_REMOTE:$ZML_DST/"
