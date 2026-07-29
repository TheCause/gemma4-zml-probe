#!/usr/bin/env bash
# GC11 — la passe de nuance sur la claim « == HF » est faite, et le reste.
#
# POURQUOI CE GATE. « ids == HF » est VRAIE au sens « même argmax sur les logits bruts » et
# FAUSSE au sens « reproduit ce que generate() produirait » : le portage n'appliquait pas
# `generation_config.json` (docs/FINDING_GENERATION_CONFIG.md). Sans passe de nuance, le chantier
# corrige le code et laisse la documentation affirmer l'inverse — livrable purement déclaratif.
#
# CE QUE LE GATE VÉRIFIE (catégorie (i) = les documents VIVANTS de référence, listés ci-dessous) :
# tout document de cette liste qui énonce « == HF » doit AUSSI porter le marqueur de portée.
# Les plans et journaux historiques sont des ARCHIVES DATÉES : on ne les réécrit pas — les
# qualifier après coup falsifierait ce qui était su au moment où ils ont été écrits.
#
# CONTRE-PREUVE (exigée par la spec §5) : `--self-test` fabrique un document de catégorie (i)
# portant une formulation nue et vérifie que le gate ÉCHOUE dessus. Un gate qu'on n'a jamais vu
# échouer n'est pas un gate (leçon feedback_invariant_tue_le_controle).
set -u

MARQUEUR="argmax sur les logits bruts"
CIBLES=(
  README.md
  PLANNING.md
  docs/CARTOGRAPHIE_portage.md
  docs/DOCUMENTATION.md
  docs/U_12B_RESULTS.md
  docs/MASKS_INGRAPH_RESULTS.md
  docs/CACHE_DONATION_RESULTS.md
  docs/REPL_RESULTS.md
  docs/GENERATION_CONFIG_RESULTS.md
)

cd "$(dirname "$0")/.." || exit 2

check() {
  # `$1` non vide : n'examiner QUE ces fichiers (utilisé par --self-test, pour que la
  # contre-preuve porte sur le canary SEUL — sinon elle « réussirait » grâce à n'importe quel
  # autre document nu, et ne prouverait rien sur le canary).
  local only="${1:-}"
  local fail=0 n_avec=0
  local liste=("${CIBLES[@]}")
  [ -n "$only" ] && liste=($only)
  for f in "${liste[@]}"; do
    [ -f "$f" ] || continue
    if grep -q "== HF" "$f"; then
      n_avec=$((n_avec + 1))
      if ! grep -qF "$MARQUEUR" "$f"; then
        echo "  NU  $f : $(grep -c '== HF' "$f") occurrence(s) de « == HF », marqueur de portée ABSENT"
        fail=1
      else
        echo "  OK  $f : $(grep -c '== HF' "$f") occurrence(s), marqueur présent"
      fi
    fi
  done
  echo "  ($n_avec document(s) de catégorie (i) énonçant la claim)"
  return $fail
}

if [ "${1:-}" = "--self-test" ]; then
  tmp="$(mktemp -d)"
  canary="$tmp/CANARY_claim.md"
  printf 'Le portage 12B est == HF sur 1150 positions.\n' > "$canary"
  echo "CONTRE-PREUVE — un document nu, EXAMINÉ SEUL, doit faire ÉCHOUER le gate :"
  if check "$canary"; then
    echo "GC11 SELF-TEST FAIL — le gate a ACCEPTÉ une formulation nue : il ne peut pas échouer."
    rm -rf "$tmp"; exit 1
  fi
  rm -rf "$tmp"
  echo "GC11 SELF-TEST PASS — le gate détecte bien une formulation nue."
  exit 0
fi

echo "GC11 — recensement de la claim « == HF » dans les documents de catégorie (i) :"
if check; then
  echo "GC11 PASS — 0 site de catégorie (i) sans qualificatif de portée."
  exit 0
fi
echo "GC11 FAIL — au moins un document énonce « == HF » sans marqueur de portée."
exit 1
