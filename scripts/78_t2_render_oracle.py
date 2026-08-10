#!/usr/bin/env python3
"""K5/PF6 — rendu HF multi-tour RÉEL (apply_chat_template), la VÉRITÉ du tour 2.

TROIS cas :
  (a) [user1] + generation prompt          — le témoin single-turn connu du runner
  (c) [user1, assistant1] SANS generation prompt — le tour assistant FERMÉ : c'est LUI qui
      correspond à `ids_fed ++ [fed_next] ++ clôture` côté runner
  (b) [user1, assistant1, user2] + generation prompt

Le suffixe du tour 2 = ids_b − préfixe_commun(ids_b, ids_c) : comparer à (a) inclurait la
réponse assistant re-tokenisée (finding bloquant de 1re revue — échec par construction).
La réécriture d'historique se juge sur préfixe_commun(ids_b, ids_c) < len(ids_c).

RECOUPEMENT DE LA FRONTIÈRE (spec §4.4.1, plan Step 2.3) — scripté, pas lu : tout id
excédentaire en queue de `ids_c` au-delà du contenu assistant appartient à l'INJECTION
D-K5-5 (clôture du tour 1), JAMAIS au littéral du tour 2. Ni PF6 ni PF1 ne peuvent voir
une erreur ici (PF6 ne compare que le suffixe ; PF1 fait consommer à l'oracle le `ctx_ids`
du runner — les deux côtés partageraient l'erreur). D'où `closure_tail_ids`, publié ici.

AUCUN verdict ici : la comparaison au rendu Zig est le gate PF6 (script 79)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

# sha256 du chat_template.jinja du snapshot 12B — constante assertée par
# scripts/69_u8_gen_oracle.py:54 (TEMPLATE_SHA_12B), RECOPIÉE, jamais devinée.
TEMPLATE_SHA = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)  # export dq (tokenizer + chat_template.jinja y vivent)
ap.add_argument("--user1", required=True)
ap.add_argument("--assistant1", required=True)  # texte du tour modèle, DÉTOKENISÉ du run A réel
ap.add_argument("--user2", required=True)
ap.add_argument("--fed-next", type=int, default=None,
                help="dernier id généré par le run A (manifest KVDUMP) — recoupement de frontière")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if "RECOPIER" in TEMPLATE_SHA:
    sys.exit("TEMPLATE_SHA est encore le placeholder — recopier la constante de 69:54")
# Même source de hash que le 69 : les OCTETS du fichier jinja (pas tok.chat_template, qu'une
# normalisation de fin de ligne suffirait à faire diverger).
jinja_path = Path(a.weights).expanduser() / "chat_template.jinja"
sha = hashlib.sha256(jinja_path.read_bytes()).hexdigest()
if sha != TEMPLATE_SHA:
    sys.exit(f"chat_template sha {sha} != attendu {TEMPLATE_SHA} — template DIFFÉRENT, STOP")

tok = AutoTokenizer.from_pretrained(str(Path(a.weights).expanduser()))
conv1 = [{"role": "user", "content": a.user1}]
convc = [{"role": "user", "content": a.user1},
         {"role": "assistant", "content": a.assistant1}]
conv2 = convc + [{"role": "user", "content": a.user2}]



def chat_ids(conv, gen_prompt):
    """apply_chat_template rend un BatchEncoding (transformers 5.14), PAS une liste : slicer
    l'objet donnerait des `tokenizers.Encoding`. Même extraction que le 69 (`enc["input_ids"]`)."""
    enc = tok.apply_chat_template(conv, add_generation_prompt=gen_prompt, return_dict=True)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):  # batché : une seule conversation ici
        ids = ids[0]
    return list(ids)


ids_a = chat_ids(conv1, True)
ids_c = chat_ids(convc, False)
ids_b = chat_ids(conv2, True)
text_b = tok.apply_chat_template(conv2, add_generation_prompt=True, tokenize=False)
text_c = tok.apply_chat_template(convc, add_generation_prompt=False, tokenize=False)


def common_len(x, y):
    n = 0
    for u, v in zip(x, y):
        if u != v:
            break
        n += 1
    return n


common_bc = common_len(ids_b, ids_c)
rewritten = common_bc < len(ids_c)  # (b) réécrit le tour assistant fermé de (c)
suffix_ids = ids_b[common_bc:]

# --- Recoupement de frontière (plan Step 2.3) -------------------------------------------
# Le contenu assistant tokenisé HORS template : sa position dans ids_c donne la frontière
# exacte entre le CONTENU (que le cache porte déjà, fed_next inclus) et la CLÔTURE (que le
# runner doit injecter, D-K5-5). Si la sous-séquence ne se retrouve pas telle quelle, on le
# DIT (content_subsequence_found=false) : la re-tokenisation hors contexte aurait divergé,
# et la frontière serait à établir autrement — c'est un signal, pas un détail.
ids_content = tok(a.assistant1, add_special_tokens=False)["input_ids"]


def find_last_sub(hay, needle):
    if not needle or len(needle) > len(hay):
        return -1
    for i in range(len(hay) - len(needle), -1, -1):
        if hay[i:i + len(needle)] == needle:
            return i
    return -1


pos = find_last_sub(ids_c, ids_content)
found = pos >= 0
closure_tail_ids = ids_c[pos + len(ids_content):] if found else []
report = {
    "template_sha256": sha,
    "ids_a": ids_a, "ids_c": ids_c, "ids_b": ids_b,
    "common_bc": common_bc, "history_rewritten": rewritten,
    "suffix_ids": suffix_ids,
    "suffix_texts": [tok.decode([i]) for i in suffix_ids],
    "content_subsequence_found": found,
    "content_ids": ids_content,
    "closure_tail_ids": closure_tail_ids if found else None,
    "closure_tail_texts": [tok.decode([i]) for i in closure_tail_ids] if found else None,
    "ids_c_tail8": ids_c[-8:],
    "ids_c_tail8_texts": [tok.decode([i]) for i in ids_c[-8:]],
    "fed_next": a.fed_next,
    # Le fed_next du manifest DOIT être le dernier id du contenu assistant : c'est le dernier
    # token généré par le run A, et le cache le porte. S'il ne l'est pas, la re-tokenisation a
    # divergé et le contexte HF n'est pas celui du runner.
    "fed_next_is_content_last": (a.fed_next == ids_content[-1]) if (a.fed_next is not None and ids_content) else None,
    "text_b": text_b, "text_c": text_c,
}
json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)
print(f"common_bc={common_bc}/{len(ids_c)} rewritten={rewritten} suffix={len(suffix_ids)} ids")
print(f"content_found={found} closure_tail_ids={closure_tail_ids} "
      f"fed_next_is_content_last={report['fed_next_is_content_last']}")
