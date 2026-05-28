# gemma4-zml-probe

Validation mathématique du contrat PLE de **google/gemma-4-E2B-it** avec Transformers,
**avant** tout portage ZML.

Règle : ne pas écrire `gemma4.zig` tant que `logs/02_contract_ple.log` et
`logs/03_ple_reference.log` ne sont pas propres.

## Phases

- **P-1 — Carte du modèle** : `00_env_check.py`, `01_fetch_metadata.sh`, `02_contract_ple.py`
- **P2 — Référence mathématique PLE** : `03_ple_reference.py`

## Lancer

```bash
source .venv/bin/activate
./scripts/04_run_all.sh
```

## Vérifier les PASS

```bash
grep -R "PASS:" logs/
grep -R "BLOCK:" logs/ || true
```

## Prérequis

- `huggingface-cli login` (licence Gemma acceptée)
- `jq`
- venv Python avec `requirements.txt`

Contrat attendu : `embed_tokens_per_layer` de shape `[262144, 8960]`.
