#!/usr/bin/env bash
# Fabrique les topologies du 3ᵉ véhicule de GC1 (découverte 1-hop) — spec §4.5bis.
# À lancer SUR LA MACHINE DU SELFTEST : le manifest référence ces chemins en absolu.
# Idempotent (rm -rf de la racine au début). Aucun selftest du repo ne crée de symlink : la
# topologie se fabrique ici, pas dans le Zig.
set -euo pipefail
ROOT="${1:-/tmp/gc1_disco}"
rm -rf "$ROOT"
mkdir -p "$ROOT"/{plain,weights_abs,snap_abs,weights_rel,snap_rel,orphan}

CFG='{"eos_token_id": [1, 106, 50], "suppress_tokens": [258883, 258882]}'

# (2) fichier régulier, config à côté
: > "$ROOT/plain/model.safetensors"
echo "$CFG" > "$ROOT/plain/generation_config.json"

# (3a) symlink ABSOLU vers un snapshot — topologie de weights_12b/ (mesurée)
: > "$ROOT/snap_abs/model.safetensors"
echo "$CFG" > "$ROOT/snap_abs/generation_config.json"
ln -s "$ROOT/snap_abs/model.safetensors" "$ROOT/weights_abs/model.safetensors"

# (3b) symlink RELATIF — topologie du cache HF (`tokenizer.json -> ../../blobs/…`)
: > "$ROOT/snap_rel/model.safetensors"
echo "$CFG" > "$ROOT/snap_rel/generation_config.json"
ln -s "../snap_rel/model.safetensors" "$ROOT/weights_rel/model.safetensors"

# (4) fichier régulier SANS config nulle part : doit échouer, pas se replier
: > "$ROOT/orphan/model.safetensors"

echo "topologies GC1 prêtes sous $ROOT"
