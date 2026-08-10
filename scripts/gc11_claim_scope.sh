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

# Le marqueur de portée est accepté dans l'UNE OU L'AUTRE langue (OU logique) — extension du
# 10 août 2026, quand le README est passé à l'anglais intégral (dette « README bilingue »,
# décision Régis). La PORTÉE du gate est strictement CONSTANTE : un document de catégorie (i)
# qui énonce « == HF » doit toujours porter le qualificatif ; seule la langue dans laquelle il
# peut l'écrire s'élargit. Ni marqueur FR ni marqueur EN ⇒ toujours NU, toujours FAIL.
MARQUEUR_FR="argmax sur les logits bruts"
MARQUEUR_EN="same argmax on the raw logits"
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
  # Ajoutés le 10 août 2026 (recensement fait pendant la clôture K5). Ce ne sont PAS des
  # documents « en plus » : ce sont des documents de RÉSULTATS, exactement la nature de ceux
  # déjà listés, qui avaient été OUBLIÉS — D10_RESULTS.md a même été écrit le 30 juillet, soit
  # le LENDEMAIN de l'instauration de ce gate. Réparer une omission, pas élargir la portée.
  docs/D10_RESULTS.md
  docs/SAMPLING_RESULTS.md
)
# ⚠ NON ajoutés, et c'est délibéré : docs/BATCHING_RESULTS.md (12 juil), docs/W4_RESULTS.md
# (24 juil) et docs/TURBOQUANT_ZML_RESULTS.md (4 juin) énoncent aussi la claim nue, mais tous
# trois sont ANTÉRIEURS au gate — même statut que les plans et journaux : ils disaient ce qui
# était su quand ils ont été écrits. Les qualifier après coup demanderait une décision, pas un
# correctif. Recensement complet et daté : PLANNING.md, section K5.
#
# ⚠ LIMITE STRUCTURELLE CONNUE de ce gate : cette liste est EN DUR. Un document de résultats
# créé demain n'y sera pas, et le gate ne le dira pas — il RÉTRÉCIT à chaque document ajouté au
# repo. C'est ce qui a laissé D10_RESULTS.md dehors pendant 11 jours. La découverte automatique
# (tout docs/*_RESULTS.md, moins une liste d'exclusion explicite) est le correctif de fond ;
# elle change la portée du gate et attend une décision Régis.

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
      if grep -qF "$MARQUEUR_FR" "$f"; then
        echo "  OK  $f : $(grep -c '== HF' "$f") occurrence(s), marqueur présent (FR)"
      elif grep -qF "$MARQUEUR_EN" "$f"; then
        echo "  OK  $f : $(grep -c '== HF' "$f") occurrence(s), marqueur présent (EN)"
      else
        echo "  NU  $f : $(grep -c '== HF' "$f") occurrence(s) de « == HF », marqueur de portée ABSENT (ni FR ni EN)"
        fail=1
      fi
    fi
  done
  echo "  ($n_avec document(s) de catégorie (i) énonçant la claim)"
  return $fail
}

if [ "${1:-}" = "--self-test" ]; then
  tmp="$(mktemp -d)"
  rc=0

  # (a) canary NU en français — le cas historique.
  canary_fr="$tmp/CANARY_claim_fr.md"
  printf 'Le portage 12B est == HF sur 1150 positions.\n' > "$canary_fr"
  echo "CONTRE-PREUVE (a) — un document nu FR, EXAMINÉ SEUL, doit faire ÉCHOUER le gate :"
  if check "$canary_fr"; then
    echo "  ✗ le gate a ACCEPTÉ une formulation nue française."
    rc=1
  fi

  # (b) canary NU en anglais — ajouté le 10 août 2026 avec l'alternative bilingue : sans lui, le
  # gate ne serait jamais vu mordre dans la langue où le README est désormais écrit.
  canary_en="$tmp/CANARY_claim_en.md"
  printf 'The 12B port is == HF over 1150 positions.\n' > "$canary_en"
  echo "CONTRE-PREUVE (b) — un document nu EN, EXAMINÉ SEUL, doit faire ÉCHOUER le gate :"
  if check "$canary_en"; then
    echo "  ✗ le gate a ACCEPTÉ une formulation nue anglaise."
    rc=1
  fi

  # (c) La question dans l'AUTRE SENS (leçon feedback_controle_qui_ne_peut_pas_reussir) : la
  # branche EN du OU doit pouvoir RÉUSSIR. Un marqueur anglais mal orthographié ici rendrait la
  # branche morte — le gate resterait « vert » en n'acceptant jamais que le français.
  canary_ok="$tmp/CANARY_claim_en_ok.md"
  printf 'The 12B port is == HF, i.e. same argmax on the raw logits.\n' > "$canary_ok"
  echo "CONTRE-ÉPREUVE (c) — un document EN PORTANT le marqueur anglais doit PASSER :"
  if ! check "$canary_ok"; then
    echo "  ✗ le gate a REFUSÉ un document anglais correctement qualifié : la branche EN est MORTE."
    rc=1
  fi

  rm -rf "$tmp"
  if [ "$rc" -ne 0 ]; then
    echo "GC11 SELF-TEST FAIL — voir les ✗ ci-dessus."
    exit 1
  fi
  echo "GC11 SELF-TEST PASS — le gate mord dans les DEUX langues, et accepte les DEUX langues."
  exit 0
fi

echo "GC11 — recensement de la claim « == HF » dans les documents de catégorie (i) :"
if check; then
  echo "GC11 PASS — 0 site de catégorie (i) sans qualificatif de portée."
  exit 0
fi
echo "GC11 FAIL — au moins un document énonce « == HF » sans marqueur de portée."
exit 1
