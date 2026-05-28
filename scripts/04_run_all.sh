#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "=================================================="
echo "00 / ENV CHECK"
echo "=================================================="
python scripts/00_env_check.py | tee logs/00_env_check.log

echo
echo "=================================================="
echo "01 / FETCH METADATA"
echo "=================================================="
./scripts/01_fetch_metadata.sh

echo
echo "=================================================="
echo "02 / PLE CONTRACT"
echo "=================================================="
python scripts/02_contract_ple.py | tee logs/02_contract_ple.log

echo
echo "=================================================="
echo "03 / PLE REFERENCE"
echo "=================================================="
python scripts/03_ple_reference.py | tee logs/03_ple_reference.log

echo
echo "=================================================="
echo "DONE"
echo "=================================================="
echo "Logs:"
echo "  logs/00_env_check.log"
echo "  logs/01_config_dump.json"
echo "  logs/02_contract_ple.log"
echo "  logs/03_ple_reference.log"
