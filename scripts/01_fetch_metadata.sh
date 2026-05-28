#!/usr/bin/env bash
set -euo pipefail

REPO="google/gemma-4-E2B-it"
OUT_DIR="./gemma4-e2b-it-meta"
LOG_JSON="./logs/01_config_dump.json"

mkdir -p "$OUT_DIR" logs

echo "=== Downloading lightweight metadata from $REPO ==="
# huggingface_hub >= 1.0 a renomme la CLI 'huggingface-cli' -> 'hf'. Fallback robuste.
HF_CLI="$(command -v hf || command -v huggingface-cli || true)"
if [ -z "$HF_CLI" ]; then
  echo "BLOCK: ni 'hf' ni 'huggingface-cli' trouve dans le PATH" >&2
  exit 1
fi
"$HF_CLI" download "$REPO" \
  config.json tokenizer.json tokenizer_config.json processor_config.json \
  --local-dir "$OUT_DIR"

echo
echo "=== Dumping text_config ==="
cat "$OUT_DIR/config.json" | jq '{
  root: {
    model_type: .model_type,
    architectures: .architectures
  },
  text: {
    model_type: .text_config.model_type,
    num_hidden_layers: .text_config.num_hidden_layers,
    hidden_size: .text_config.hidden_size,
    hidden_size_per_layer_input: .text_config.hidden_size_per_layer_input,
    vocab_size_per_layer_input: .text_config.vocab_size_per_layer_input,
    num_kv_shared_layers: .text_config.num_kv_shared_layers,
    final_logit_softcapping: .text_config.final_logit_softcapping,
    layer_types: .text_config.layer_types
  }
}' | tee "$LOG_JSON"

echo
echo "PASS: metadata downloaded and config dumped"
