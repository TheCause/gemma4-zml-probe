#!/usr/bin/env bash
# GC11 — la passe de nuance sur la claim « == HF » est faite, et le reste.
#
# POURQUOI CE GATE. « ids == HF » est VRAIE au sens « même argmax sur les logits bruts » et
# FAUSSE au sens « reproduit ce que generate() produirait » : le portage n'appliquait pas
# `generation_config.json` (docs/FINDING_GENERATION_CONFIG.md). Sans passe de nuance, le chantier
# corrige le code et laisse la documentation affirmer l'inverse — livrable purement déclaratif.
#
# CE QUE LE GATE VÉRIFIE (catégorie (i) = les documents VIVANTS de référence) : tout document de
# cette catégorie qui énonce « == HF » doit AUSSI porter le marqueur de portée. Les plans,
# designs et journaux sont des ARCHIVES DATÉES : on ne les réécrit pas — les qualifier après coup
# falsifierait ce qui était su au moment où ils ont été écrits.
#
# ⚡ CHANGEMENT DU 10 AOÛT 2026 — LA LISTE EST AUTO-DÉCOUVERTE (décision Régis).
# Elle était EN DUR, et elle RÉTRÉCISSAIT à chaque document créé : `docs/D10_RESULTS.md`, écrit
# le LENDEMAIN de l'instauration de ce gate, en est resté dehors 11 jours sans que rien ne le
# signale. Un gate dont le périmètre se périme tout seul finit par ne plus rien garder.
# La catégorie (i) est donc DÉRIVÉE du dépôt :
#   * les références vivantes de la racine et de docs/ (liste courte, ci-dessous) ;
#   * TOUT `docs/*_RESULTS.md` — un document de résultats créé demain est couvert le jour même.
# Les exclusions sont EXPLICITES, DATÉES, et AFFICHÉES à chaque exécution : rien ne sort du
# périmètre en silence.
#
# CONTRE-PREUVE (exigée par la spec §5) : `--self-test` fabrique des documents nus et vérifie que
# le gate ÉCHOUE dessus — y compris un document que PERSONNE N'A LISTÉ, pour prouver que la
# découverte découvre. Un gate qu'on n'a jamais vu échouer n'est pas un gate
# (leçon feedback_invariant_tue_le_controle).
set -u

# Le marqueur de portée est accepté dans l'UNE OU L'AUTRE langue (OU logique) — extension du
# 10 août 2026, quand le README est passé à l'anglais intégral (dette « README bilingue »,
# décision Régis). La PORTÉE du gate est strictement CONSTANTE : un document de catégorie (i)
# qui énonce « == HF » doit toujours porter le qualificatif ; seule la langue dans laquelle il
# peut l'écrire s'élargit. Ni marqueur FR ni marqueur EN ⇒ toujours NU, toujours FAIL.
MARQUEUR_FR="argmax sur les logits bruts"
MARQUEUR_EN="same argmax on the raw logits"

# (1) Références vivantes hors motif `*_RESULTS.md` — elles n'ont pas de nom régulier, elles sont
# donc nommées. Ajouter un document ICI est un acte délibéré ; en oublier un ne concerne que
# cette poignée de fichiers, pas la famille entière des résultats.
REFS_VIVANTES=(
  README.md
  PLANNING.md
  docs/CARTOGRAPHIE_portage.md
  docs/DOCUMENTATION.md
)

# (2) Exclusions EXPLICITES du motif `docs/*_RESULTS.md`. Chacune porte sa date et sa raison.
# Tous ces documents énoncent la claim nue, et tous sont ANTÉRIEURS au gate (29 juil 2026) :
# ils ont le même statut que les plans et journaux — ils disaient ce qui était su alors.
# Les qualifier après coup demanderait une décision, pas un correctif.
EXCLUSIONS=(
  "docs/TURBOQUANT_ZML_RESULTS.md:créé le 4 juin 2026, antérieur au gate"
  "docs/BATCHING_RESULTS.md:créé le 12 juil 2026, antérieur au gate"
  "docs/W4_RESULTS.md:créé le 24 juil 2026, antérieur au gate"
)

cd "$(dirname "$0")/.." || exit 2

est_exclu() {
  local f="$1" e
  for e in "${EXCLUSIONS[@]}"; do
    [ "${e%%:*}" = "$f" ] && return 0
  done
  return 1
}

# Catégorie (i), DÉRIVÉE du dépôt. Ordre déterministe (le glob est trié), sortie une ligne par
# fichier — c'est cette fonction, et elle seule, qui définit le périmètre du gate.
decouvrir() {
  local f
  for f in "${REFS_VIVANTES[@]}"; do
    [ -f "$f" ] && echo "$f"
  done
  for f in docs/*_RESULTS.md; do
    [ -f "$f" ] || continue
    est_exclu "$f" || echo "$f"
  done
}

# Les exclusions sont AFFICHÉES, et leur pertinence est re-contrôlée à chaque run : une exclusion
# qui ne désigne plus rien, ou qui protège un document désormais QUALIFIÉ, est un vestige — le
# gate le dit au lieu de la traîner (anti-pattern « vestige à 3 couches »).
montrer_exclusions() {
  local e f raison
  for e in "${EXCLUSIONS[@]}"; do
    f="${e%%:*}"; raison="${e#*:}"
    if [ ! -f "$f" ]; then
      echo "  ⚠ EXCLUSION VESTIGE : $f n'existe plus — la retirer de EXCLUSIONS"
    elif grep -qF "$MARQUEUR_FR" "$f" || grep -qF "$MARQUEUR_EN" "$f"; then
      echo "  ⚠ EXCLUSION INUTILE : $f porte désormais le marqueur — il peut rentrer dans le périmètre"
    elif ! grep -q "== HF" "$f"; then
      echo "  ⚠ EXCLUSION INUTILE : $f n'énonce plus « == HF » — la retirer de EXCLUSIONS"
    else
      echo "  exclu  $f ($raison)"
    fi
  done
}

check() {
  # `$1` non vide : n'examiner QUE ces fichiers (utilisé par --self-test, pour que la
  # contre-preuve porte sur le canary SEUL — sinon elle « réussirait » grâce à n'importe quel
  # autre document nu, et ne prouverait rien sur le canary).
  local only="${1:-}"
  local fail=0 n_avec=0 f
  local liste=()
  if [ -n "$only" ]; then
    liste=($only)
  else
    while IFS= read -r f; do liste+=("$f"); done < <(decouvrir)
  fi
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
  echo "  ($n_avec document(s) de catégorie (i) énonçant la claim, sur ${#liste[@]} dans le périmètre)"
  return $fail
}

# `--perimetre` : imprimer le périmètre découvert, un fichier par ligne, et rien d'autre.
# Existe pour que tout contrôle externe (non-régression, revue) lise le périmètre RÉEL du gate
# au lieu de ré-implémenter sa règle — deux implémentations de la même règle divergent.
if [ "${1:-}" = "--perimetre" ]; then
  decouvrir
  exit 0
fi

if [ "${1:-}" = "--self-test" ]; then
  tmp="$(mktemp -d)"
  # Le canary de découverte vit DANS docs/ — c'est la seule façon de prouver que la découverte
  # le trouve. Le trap garantit qu'il ne survit pas au script, y compris sur interruption.
  canary_dec="docs/ZZZ_CANARY_DECOUVERTE_RESULTS.md"
  trap 'rm -rf "$tmp" "$canary_dec"' EXIT INT TERM
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

  # (d) LA CONTRE-PREUVE DE L'AUTO-DÉCOUVERTE (10 août 2026). Un document de résultats que
  # PERSONNE N'A LISTÉ est déposé dans docs/, et le gate est lancé SANS lui indiquer quoi que ce
  # soit. Il doit le TROUVER et le condamner. C'est ce cas, et lui seul, qui distingue le
  # nouveau gate de l'ancien : sous l'ancienne liste en dur, ce canary serait passé inaperçu —
  # exactement comme D10_RESULTS.md pendant 11 jours.
  printf 'Un document de résultats non listé, qui affirme == HF sans qualificatif.\n' > "$canary_dec"
  echo "CONTRE-PREUVE (d) — un *_RESULTS.md NON LISTÉ, nu, doit être DÉCOUVERT puis condamné :"
  sortie_d="$(check)"
  rc_d=$?
  echo "$sortie_d" | grep -F "$canary_dec" | sed 's/^/  /'
  if [ "$rc_d" -eq 0 ]; then
    echo "  ✗ le gate a PASSÉ alors qu'un document nu non listé traînait dans docs/."
    rc=1
  elif ! echo "$sortie_d" | grep -qF "$canary_dec"; then
    echo "  ✗ le gate a échoué, mais SANS voir le canary : il a condamné pour une autre raison,"
    echo "    la découverte n'est donc PAS prouvée."
    rc=1
  fi
  rm -f "$canary_dec"

  # (e) Et dans l'AUTRE SENS : une fois le canary retiré, le dépôt réel doit repasser au vert.
  # Sans ce cas, (d) « réussirait » aussi bien si le gate condamnait tout, tout le temps.
  echo "CONTRE-ÉPREUVE (e) — canary retiré, le dépôt réel doit repasser au VERT :"
  if ! check > /dev/null; then
    echo "  ✗ le dépôt réel est ROUGE hors canary : (d) ne prouvait pas la découverte."
    rc=1
  else
    echo "  OK  le dépôt réel repasse au vert."
  fi

  if [ "$rc" -ne 0 ]; then
    echo "GC11 SELF-TEST FAIL — voir les ✗ ci-dessus."
    exit 1
  fi
  echo "GC11 SELF-TEST PASS — le gate mord dans les DEUX langues, accepte les DEUX langues,"
  echo "                      et DÉCOUVRE un document que personne ne lui a désigné."
  exit 0
fi

echo "GC11 — périmètre AUTO-DÉCOUVERT (références vivantes + tout docs/*_RESULTS.md) :"
montrer_exclusions
echo "GC11 — recensement de la claim « == HF » dans les documents de catégorie (i) :"
if check; then
  echo "GC11 PASS — 0 site de catégorie (i) sans qualificatif de portée."
  exit 0
fi
echo "GC11 FAIL — au moins un document énonce « == HF » sans marqueur de portée."
exit 1
