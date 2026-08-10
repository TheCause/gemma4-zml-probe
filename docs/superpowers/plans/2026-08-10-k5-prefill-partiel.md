# K5 — Prefill partiel : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal :** reprendre un cache KV dumpé et **absorber un prompt neuf** (`--load-cache` +
`--prompt`) — la brique conversation/préfixe partagé — prouvée teacher-forcée contre HF.

**Architecture :** chantier **entièrement host-side** : le graphe ne distingue pas prefill et
génération (position ≡ `ctrl.step`, masques et RoPE dérivés in-graph ou de tables couvrant
`L_MAX`). Le code construit `ids_full = ids_fed ++ [fed_next] ++ ids_t2` et laisse la boucle
prefill-par-decode existante faire le reste. La preuve est un aller-retour teacher-forcé :
top-5 ZML aux positions du tour 2 (`top5 @ ctx=`) ↔ argmax HF fp32 au même préfixe.

**Tech stack :** Zig (zml_runner), Python (oracle HF M4, dépouilleurs), VM 3090, Bazel.

**Spec de référence :** `docs/superpowers/specs/2026-08-10-k5-prefill-partiel-design.md`
(**rév. 3** — 2 passes de revue adversariale, 22 findings corrigés + 1 contrôle vacueux
retiré avant d'être codé) — **LA SPEC FAIT FOI** en cas d'écart. Ne PAS réémettre ses
sections : la référencer (leçon `feedback_prolonger_une_spec_revue`).

---

## Contexte pour une session neuve (lis ceci d'abord)

Règles non négociables du repo, toutes déjà payées :

1. **Le graphe ne bouge pas** : md5 HLO `before_optimizations` attendu
   **`297679847aa04b719942d75d093adf2b`** (stable sur six chantiers). Tout autre md5 ⇒ STOP.
2. **Chaque preuve dans `docs/evidence/k5/`** : logs bruts `> out.log 2> err.log` (flux
   séparés, jamais entrelacés), verdicts, prédictions écrites AVANT mesure.
3. **Build UNIQUEMENT via `./zml_runner/build_3090.sh`** (2 flags de mode) ; un log de gate
   sans `BUILD: mode=ReleaseFast` est **INEXÉCUTABLE**, pas PASS.
4. **`ALLOC-LOOP: alloc=0`** sur tout run de gate (interdit D10).
5. **⚡ Règle du 10 août : le md5 du BINAIRE est consigné avec TOUT témoin de sortie**
   (`md5sum bazel-bin/examples/rqz/gemma4_g12auto` dans le log d'evidence) — un témoin d'ids
   sans binaire identifié n'est pas une référence (`FINDING_temoin_ids_non_reproductible.md`).
6. **VRAM avant tout run long** : `nvidia-smi --query-compute-apps=pid,name,used_memory
   --format=csv` — un ollama résident a déjà coûté un `GpuBusy`.
7. **Branche + PR, jamais sur `main`** : branche `k5-prefill-partiel`.
8. **Runs longs** : `nohup <cmd> > x.out.log 2> x.err.log; echo DONE rc=$?` — jamais en
   avant-plan SSH.
9. **Grep anonymisation APRÈS la dernière écriture** (leçon incident 10 août) :
   `git grep -nE 'Users/regis|macmini|192\.168' -- ':!docs/superpowers'` doit être vide
   au moment du push, ET re-vérifier les TAGS après toute réécriture d'historique.
10. **FAIL ⇒ STOP, jamais de requalification à chaud.** Une requalification exige une
    décision Régis écrite (le cas C-K5-F §2bis de la spec est PRÉ-déclaré, lui).

### Invocations réelles

```bash
# Déploiement + build (depuis M1, racine du repo) :
export ZML_REMOTE=ia@192.168.1.163 ZML_DST=/data/rqz_workspace/zml/examples/rqz
./zml_runner/deploy_to_3090.sh
ZML_REMOTE=ia@192.168.1.163 ZML_WS=/data/rqz_workspace/zml ./zml_runner/build_3090.sh

# Runner (sur la VM ia@192.168.1.163) :
cd /data/rqz_workspace/zml
B1=./bazel-bin/examples/rqz/gemma4_g12auto     # variante 1280
B4=./bazel-bin/examples/rqz/gemma4_g12a4k      # variante 4k
W=/data/gemma4-zml-probe/weights_12b
K=/data/gemma4-zml-probe/k5                    # mkdir -p au premier usage

# Oracle fp32 (sur M4, venv ~/ml-venvs/g12b, export dq ~/ml-data/weights_12b_dq/) :
python3 scripts/69_u8_gen_oracle.py --weights ~/ml-data/weights_12b_dq --compute-fp32 ...
# ⚠ la VM et M4 n'ont PAS le repo git : scp les scripts, consigner script_md5 (règle GC7).
```

### Faits établis par la session de cadrage (10 août, tous vérifiés sur `f184c46`)

1. **Aucun graphe de prefill** : boucle unique `gemma4_g12auto.zig:3161`, discriminant
   `in_gen_phase = step + 1 >= ids.len` (`:3189`), en prefill `fed = ids[step+1]`
   (`:3387-3391`). La reprise = `step = step_next; fed = fed_next` (`:3153-3156`), le saut
   du prefill est structurel (pas une branche).
2. **Position ≡ step** : scalaire runtime `ctrl.step`, table `positions` identité
   (`:571-580`), masques in-graph (`engine.zig:418-439`), RoPE sliding in-graph
   (`engine.zig:554,573`), RoPE full en table host remplie jusqu'à `L_MAX` (`:510-530`),
   scatter cache linéaire sans modulo (`engine.zig:599-612`). **Rien à étendre côté graphe.**
3. **La garde à lever** : `--load-cache + --prompt ⇒ error.LoadCacheWithPrompt`
   (`:1968-1971`). Les gardes à CONSERVER : `--ignore-prompt + --load-cache` (`:1983-1986`),
   place `:2291-2293` et `:3051-3054`, oracle `positions[0]` (`:2249-2252`).
4. **`ids` sous reprise** : rempli par `ids_fed` à `:2226` (phase 1 manifest, avant compile) ;
   `Resume = { step_next, fed_next, t_load0 }` (`:2865`) ; `ManifestCheck.fed_next` validé
   contre le vocab (`:2969-2972`).
5. **Template chat single-turn SEULEMENT** (`renderChatTemplate` `:111-113`, périmètre
   `:106-107`) ; tokens de tour `<|turn>`=105 / `<turn|>`=106, BOS=2 préfixé en id, l'encoder
   n'ajoute AUCUN token spécial et exige `reset()` avant chaque encode (`:120-131`).
6. **`--dump-top5` est gated `in_gen_phase`** (`:3363`) — les positions de prefill ne sortent
   jamais. `--out-ids` n'écrit que les générés (`writeIdsSafetensors` `:1814-1828`, site
   `:3615-3618`).
7. **`69_u8_gen_oracle.py`** : mode `--teacher-force` = UN prefill HF de
   `render(--prompt) ++ gen[:-1]`, tête manuelle par chunks de 64, self-check préfixe 8
   (`:483-496`), témoin fenêtre sliding (`:452-465`), marges consignées AVANT verdict.
   `--prompt` y est REQUIS — d'où le mode nouveau `--context-ids`.
8. **`--oracle` du runner n'est PAS teacher-forcé** (`fed = tok` `:3444`, verdict post-boucle
   `:3621-3647`) — et `docs/DOCUMENTATION.md:252` l'annote faussement « teacher-forcé » :
   correction dans ce chantier (Task 11).
9. **Prompt vide en reprise simple** : `prompt_text` vaut `""` quand `--prompt` est absent —
   le chemin K5 ne s'active que si `args.prompt != null`.

---

## Task 0 : décisions Régis + préflight

**Files :** aucun.

- [ ] **Step 0.1 : poser les 5 décisions (spec §4.7) et attendre le GO explicite**

| # | Question | Défaut proposé (spec) |
|---|---|---|
| D-K5-1 | `fed_next` dans `ids_full` ? | Toujours inclus |
| D-K5-2 | Rendu tour 2 | VÉRITÉ = rendu HF mesuré (Task 2) ; si le jinja réécrit l'historique : sémantique « fidèle au généré », écart publié, GO Régis |
| D-K5-3 | Garde fenêtre | Transposée : `n_new >= 1024` ⇒ refus (dette : la lever exigerait son gate) |
| D-K5-4 | Variantes gatées | 1280 seule pour PF1-PF4/PF6 ; PF5 sur 1280 ET 4k ; gate teacher-forcé 4k = dette |
| D-K5-5 | Tour 1 non clos (`fed_next` ≠ EOS, cas nominal d'un run A à `max_tokens`) | Clôture injectée : `eot_id` appendé entre `fed_next` et le tour 2, logué (spec §4.1) |

- [ ] **Step 0.2 : préflight**

```bash
cd ~/dev/gemma4-zml-probe && git status --porcelain && git log --oneline -1
git checkout -b k5-prefill-partiel
ssh ia@192.168.1.163 'nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv'
```

Expected : arbre propre sur `f184c46` (ou descendant), VRAM libre (sinon `ollama stop`).

---

## Task 1 : témoins AVANT toute modification (PF0-avant + binaire)

**Files :** `docs/evidence/k5/hlo_witness_avant.md5`, `docs/evidence/k5/binaire_avant.md5`.

- [ ] **Step 1.1 : déployer l'état NON modifié, capturer le témoin HLO et le md5 binaire**

```bash
# depuis M1 :
export ZML_REMOTE=ia@192.168.1.163 ZML_DST=/data/rqz_workspace/zml/examples/rqz
./zml_runner/deploy_to_3090.sh
ZML_REMOTE=ia@192.168.1.163 ZML_WS=/data/rqz_workspace/zml ./zml_runner/build_3090.sh
# sur la VM :
mkdir -p /data/gemma4-zml-probe/k5
XLA_FLAGS="--xla_dump_to=/data/gemma4-zml-probe/k5/hlo_avant" \
  $B1 $W/model.safetensors $W/tokenizer.json --prompt "witness" --max-tokens 4 \
  > /data/gemma4-zml-probe/k5/witness.out.log 2> /data/gemma4-zml-probe/k5/witness.err.log
md5sum /data/gemma4-zml-probe/k5/hlo_avant/*before_optimizations.txt \
  > /data/gemma4-zml-probe/k5/hlo_witness_avant.md5
md5sum ./bazel-bin/examples/rqz/gemma4_g12auto \
  > /data/gemma4-zml-probe/k5/binaire_avant.md5
cat /data/gemma4-zml-probe/k5/*.md5
grep "BUILD: mode=" /data/gemma4-zml-probe/k5/witness.err.log
```

Expected : md5 HLO = **`297679847aa04b719942d75d093adf2b`**, `BUILD: mode=ReleaseFast`.
Tout autre md5 ⇒ **STOP** (le graphe a bougé avant nous).

- [ ] **Step 1.2 : rapatrier + commit**

```bash
mkdir -p ~/dev/gemma4-zml-probe/docs/evidence/k5   # le répertoire n'existe PAS encore
scp ia@192.168.1.163:/data/gemma4-zml-probe/k5/hlo_witness_avant.md5 \
    ia@192.168.1.163:/data/gemma4-zml-probe/k5/binaire_avant.md5 \
    ~/dev/gemma4-zml-probe/docs/evidence/k5/
git add docs/evidence/k5/ && git commit -m "k5(PF0) : témoins HLO + binaire AVANT la première ligne de code"
```

---

## Task 2 : mesurer le rendu HF multi-tour (D-K5-2) — AVANT tout littéral Zig

**Files :** Create `scripts/78_t2_render_oracle.py`, `docs/evidence/k5/rendu_tour2_hf.json`.

- [ ] **Step 2.1 : écrire le script de mesure (M4)**

```python
#!/usr/bin/env python3
"""K5/PF6 — rendu HF multi-tour RÉEL (apply_chat_template), la VÉRITÉ du tour 2.
TROIS cas : (a) tour 1 + generation prompt (le témoin single-turn connu du runner),
(c) [user1, assistant1] SANS generation prompt (le tour assistant FERMÉ — c'est lui qui
correspond à ids_fed ++ fed_next ++ clôture), (b) conversation 2 tours + generation prompt.
Le suffixe du tour 2 = ids_b − préfixe_commun(ids_b, ids_c) : comparer à (a) inclurait la
réponse assistant re-tokenisée (finding bloquant de 1re revue — échec par construction).
La réécriture d'historique se juge sur préfixe_commun(ids_b, ids_c) < len(ids_c).
AUCUN verdict ici : la comparaison au rendu Zig est le gate PF6 (script 79)."""
import argparse, hashlib, json, sys
from pathlib import Path
from transformers import AutoTokenizer

TEMPLATE_SHA = "RECOPIER_DEPUIS_69_LIGNE_54"  # ⚠ PLACEHOLDER OBLIGATOIRE À REMPLACER :
# la valeur exacte est la constante assertée par scripts/69_u8_gen_oracle.py:54 — la LIRE,
# jamais la deviner. Le script refuse de tourner tant que le placeholder est en place.

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)  # export dq (tokenizer + chat_template.jinja y vivent)
ap.add_argument("--user1", required=True)
ap.add_argument("--assistant1", required=True)  # texte du tour modèle, DETOKENISÉ du run A réel
ap.add_argument("--user2", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

if "RECOPIER" in TEMPLATE_SHA:
    sys.exit("TEMPLATE_SHA est encore le placeholder — recopier la constante de 69:54")
# Même source de hash que le 69 : les OCTETS du fichier jinja (pas tok.chat_template,
# qu'une normalisation de fin de ligne suffirait à faire diverger).
jinja_path = Path(a.weights) / "chat_template.jinja"
sha = hashlib.sha256(jinja_path.read_bytes()).hexdigest()
if sha != TEMPLATE_SHA:
    sys.exit(f"chat_template sha {sha} != attendu {TEMPLATE_SHA} — template DIFFÉRENT, STOP")

tok = AutoTokenizer.from_pretrained(a.weights)
conv1 = [{"role": "user", "content": a.user1}]
convc = [{"role": "user", "content": a.user1},
         {"role": "assistant", "content": a.assistant1}]
conv2 = convc + [{"role": "user", "content": a.user2}]
ids_a = tok.apply_chat_template(conv1, add_generation_prompt=True)
ids_c = tok.apply_chat_template(convc, add_generation_prompt=False)
ids_b = tok.apply_chat_template(conv2, add_generation_prompt=True)
text_b = tok.apply_chat_template(conv2, add_generation_prompt=True, tokenize=False)
text_c = tok.apply_chat_template(convc, add_generation_prompt=False, tokenize=False)

def common_len(x, y):
    n = 0
    for u, v in zip(x, y):
        if u != v: break
        n += 1
    return n

common_bc = common_len(ids_b, ids_c)
rewritten = common_bc < len(ids_c)  # (b) réécrit le tour assistant fermé de (c)
json.dump({"template_sha256": sha, "ids_a": ids_a, "ids_c": ids_c, "ids_b": ids_b,
           "common_bc": common_bc, "history_rewritten": rewritten,
           "suffix_ids": ids_b[common_bc:], "text_b": text_b, "text_c": text_c},
          open(a.out, "w"), indent=1)
print(f"common_bc={common_bc}/{len(ids_c)} rewritten={rewritten} suffix={len(ids_b)-common_bc} ids")
```

- [ ] **Step 2.2 : exécuter sur M4 avec un tour 1 RÉEL**

Le `--assistant1` doit être un texte réellement générable (détokenisé) — utiliser la sortie
texte d'un run court quelconque du runner (ou d'un run HF), pas une invention. Exemple :

```bash
scp scripts/78_t2_render_oracle.py macmini:/tmp/ && ssh macmini \
  '~/ml-venvs/g12b/bin/python3 /tmp/78_t2_render_oracle.py --weights ~/ml-data/weights_12b_dq \
   --user1 "My name is Aldebaran and I live in a lighthouse. Tell me a story about my home." \
   --assistant1 "<texte du run A détokenisé>" --user2 "What is my name?" \
   --out /tmp/rendu_tour2_hf.json'
scp macmini:/tmp/rendu_tour2_hf.json docs/evidence/k5/
python3 -c "import json; d=json.load(open('docs/evidence/k5/rendu_tour2_hf.json')); print(d['history_rewritten'], d['text_b'][-400:])"
```

- [ ] **Step 2.3 : trancher selon le résultat**

- `history_rewritten == false` : le suffixe est une concaténation propre → extraire le
  **texte littéral** du suffixe (`text_b` moins `text_c`) : c'est le gabarit de
  `renderChatTemplateTurn2` (Task 3). Forme attendue :
  `\n<|turn>user\n{user2}<turn|>\n<|turn>model\n…` — noter où s'insère `{s}` et si le
  suffixe commence par un `\n` (id 107, celui qui suit le `<turn|>` de clôture).
- `history_rewritten == true` (cas pré-déclaré C-K5-F) : **STOP, publier le diff, demander
  le GO Régis** sur la sémantique « fidèle au généré » (spec §4.4.3) avant de continuer.
- ⚠ Recouper la frontière — et c'est un RECOUPEMENT SCRIPTÉ, pas une lecture : `ids_c`
  doit se terminer par le tour assistant FERMÉ, et **tout id excédentaire en queue de
  `ids_c`** (au-delà du contenu + EOS 106, ex. un `\n` 107) **appartient à l'injection
  D-K5-5 (Step 3.3), JAMAIS au littéral du tour 2** — PF6 ne compare que le suffixe et
  ne peut pas voir ces ids ; PF1 non plus (l'oracle consomme le `ctx_ids` du runner, les
  deux côtés partageraient l'erreur). Concrètement : vérifier par script que
  `ids_c[len(commun(ids_b, ids_c)) - k :]` (la queue de clôture) == `[fed_next?] ++
  injection prévue`, et consigner le résultat dans `rendu_tour2_hf.json` (champ
  `closure_tail_ids`). Un FAIL PF6 sur un id de tête du suffixe ne se « corrige » PAS en
  retranchant cet id du littéral : il se diagnostique contre `closure_tail_ids`.

- [ ] **Step 2.4 : commit**

```bash
git add scripts/78_t2_render_oracle.py docs/evidence/k5/rendu_tour2_hf.json
git commit -m "k5(D-K5-2) : rendu HF multi-tour MESURÉ — la vérité du littéral tour 2"
```

---

## Task 3 : le code Zig — chemin `--load-cache` + `--prompt`

**Files :** Modify `zml_runner/gemma4_g12auto.zig` (6 sites).

- [ ] **Step 3.1 : rendu et tokenisation du tour 2** (après `promptToIds`, `:131`)

```zig
// K5 — rendu du tour 2 (spec §4.4) : le SUFFIXE CANONIQUE post-clôture, littéral FIGÉ PAR
// LA MESURE HF de la Task 2 (docs/evidence/k5/rendu_tour2_hf.json, clé suffix_ids), prouvé
// en ids par le gate PF6. SANS BOS : l'encoder n'ajoute aucun token spécial, le BOS est le
// préfixe du SEUL tour 1 (`:109-110`). La CLÔTURE du tour 1 (eot injecté si fed_next n'est
// pas un EOS, D-K5-5) n'est PAS ici : elle vit dans la construction d'ids_full (Step 3.3).
// ⚠ Le gabarit ci-dessous est la forme ATTENDUE — le remplacer par le littéral mesuré si la
// Task 2 en décide autrement (PF6 le contre-prouve de toute façon).
fn renderChatTemplateTurn2(allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "\n<|turn>user\n{s}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", .{prompt});
}

// Tour 2 → ids, SANS BOS. Même hygiène que promptToIds (reset : l'encoder iree est un
// automate à état).
fn promptToIdsTurn2(allocator: std.mem.Allocator, encoder: anytype, prompt_text: []const u8) !std.ArrayList(u32) {
    encoder.reset();
    const rendered = try renderChatTemplateTurn2(allocator, prompt_text);
    defer allocator.free(rendered);
    var prompt_tok = try encoder.encodeAlloc(allocator, rendered);
    defer prompt_tok.deinit(allocator);
    var ids: std.ArrayList(u32) = try .initCapacity(allocator, prompt_tok.items.len);
    errdefer ids.deinit(allocator);
    try ids.appendSlice(allocator, prompt_tok.items);
    return ids;
}
```

- [ ] **Step 3.2 : lever la garde `:1968-1971`, la remplacer par le commentaire de capacité**

```zig
    // K5 (spec 2026-08-10 §4.1) : --load-cache + --prompt est désormais le PREFILL PARTIEL —
    // le contexte vient du dump, le prompt est absorbé comme TOUR 2 aux positions step_next…
    // (l'ancienne garde LoadCacheWithPrompt, spec kvdump §3, est levée par ce chantier).
```

⚠ Retirer aussi la mention « exclut --load-cache » de l'usage string si elle y est, et
ajouter la capacité à l'usage (`--load-cache F [--prompt "tour 2"]`).

- [ ] **Step 3.3 : construction de `ids_full` en phase 1 du restore** (site `:2222-2227`)

```zig
    var mcheck: ?ManifestCheck = null;
    defer if (mcheck) |*mc| mc.deinit(allocator, io);
    var n_new: usize = 0; // K5 : tokens du tour 2 (0 = reprise simple, chemin kvdump inchangé)
    if (args.load_cache) |cache_path| {
        mcheck = try loadCacheManifest(allocator, io, cache_path, args.ckpt, policy.path);
        try ids.appendSlice(allocator, mcheck.?.ids_fed);
        if (args.prompt != null) {
            // K5 : prompt VIDE gardé AVANT le rendu — le rendu émet toujours ses marqueurs de
            // tour, n_new ne peut jamais valoir 0 après lui (2e revue : une garde post-rendu
            // serait à antécédent vide).
            if (prompt_text.len == 0) {
                log.err("K5 : --prompt vide sous reprise — un tour 2 sans contenu n'est pas un prefill partiel", .{});
                return error.PromptTooLong;
            }
            // K5 (spec §4.1) : ids_full = ids_fed ++ [fed_next] ++ [clôture?] ++ ids_t2.
            // fed_next est TOUJOURS inclus (D-K5-1) : dernier token généré, jamais feedé — il
            // fait partie du texte produit (y compris un EOS de fin de tour).
            try ids.append(allocator, @intCast(mcheck.?.fed_next));
            // D-K5-5 : le rendu HF ferme TOUJOURS le tour assistant. Si fed_next n'est pas un
            // EOS (arrêt max_tokens du run dumpé), l'eot mesuré est injecté — sinon le tour 2
            // s'ouvrirait dans un tour assistant jamais clos, non conforme au rendu que PF6
            // prouve.
            if (!policy.isEos(@intCast(mcheck.?.fed_next))) {
                try ids.append(allocator, eot_id);
                log.info("K5: tour 1 clos par eot {d} injecté (fed_next={d} n'est pas un EOS)", .{ eot_id, mcheck.?.fed_next });
            }
            var t2 = try promptToIdsTurn2(allocator, &encoder, prompt_text);
            defer t2.deinit(allocator);
            n_new = t2.items.len;
            // (garde n_new == 0 : DÉFENSE seulement, non exerçable — le rendu émet toujours
            // ses marqueurs ; le cas gaté PF4(d) est la garde prompt_text.len ci-dessus)
            if (n_new == 0) {
                log.err("K5 : tour 2 vide après rendu — prompt inutilisable", .{});
                return error.PromptTooLong;
            }
            // D-K5-3 : garde fenêtre TRANSPOSÉE au prompt neuf (la garde historique :2300 est
            // désactivée sous reprise ; sa raison d'être n'a jamais été écrite — c2211c0 — on
            // transpose la prudence, la LEVER exigerait son propre gate).
            if (n_new >= @as(usize, @intCast(SLIDING_WINDOW))) {
                log.err("K5 : tour 2 de {d} ids >= SLIDING_WINDOW({d})", .{ n_new, SLIDING_WINDOW });
                return error.PromptTooLong;
            }
            try ids.appendSlice(allocator, t2.items);
            log.info("K5: prefill partiel — contexte {d} ids + fed_next + tour2 {d} ids = {d} total", .{ mcheck.?.ids_fed.len, n_new, ids.items.len });
        }
    }
```

⚠ `encoder` doit être encore vivant à ce point (il l'est : même scope de `run()` que
`promptToIds` `:2077`) ; NE PAS toucher au chemin `args.prompt == null`.
⚠ La garde de place `:2291-2293` (`ids.len + limit > L_MAX`) travaille maintenant sur
`ids_full` — c'est le comportement voulu (spec §1.5), aucun code à changer.

- [ ] **Step 3.4 : le site de reprise `:3153-3156` ne change PAS — mettre à jour son
  commentaire seulement**

⚠ Une assertion `ids_full[step_next] == fed_next` avait été envisagée ici : RETIRÉE en
revue de spec — les deux membres dérivent du même champ du manifest (`mcheck.fed_next`),
elle ne peut pas échouer (`feedback_controle_qui_ne_peut_pas_reussir`). La limite réelle
(`fed_next` forgé = contexte forgé, détectable uniquement par l'aller-retour oracle) est
documentée spec §4.6 et ira dans `K5_RESULTS.md` §limites. Compléter le commentaire
`:3149-3152` : sous K5, `ids.len > step_next` (prompt neuf) rend `in_gen_phase` FAUX au
premier step — la boucle absorbe le tour 2 en prefill-par-decode, par construction.

- [ ] **Step 3.5 : log `top5 @ ctx=` en prefill de reprise** (site `:3363`)

```zig
        if (in_gen_phase and dump_top5) log.info("  top5 @ gen={d} : idx={any} val={any} rank_used={d} chosen={d}", .{ gen_top5.items.len - 1, top5.idx, top5.val, sel.rank, sel.tok });
        // K5/PF1 : les positions du PREFILL DE REPRISE sont les positions teacher-forcées par
        // construction (token feedé imposé, aucun effet boule de neige) — leur top-5 est LA
        // sortie que l'oracle compare. Émis SEULEMENT sous reprise : le prefill d'un tour 1
        // n'intéresse aucun gate (et 1000 lignes pollueraient les logs).
        if (!in_gen_phase and dump_top5 and resume_state != null) log.info("  top5 @ ctx={d} : idx={any} val={any}", .{ step, top5.idx, top5.val });
```

- [ ] **Step 3.6 : `ctx_ids` dans `--out-ids` sous reprise** (après `writeIdsSafetensors`,
  `:1828`, + site d'appel `:3615-3618`)

```zig
// K5 — variante 2 clés de writeIdsSafetensors : "ids" (générés, format historique intact
// pour tous les consommateurs existants) + "ctx_ids" (la séquence COMPLÈTE feedée avant
// génération : ids_full). L'oracle 69 --context-ids en a besoin : sous reprise, le contexte
// n'est pas exprimable par un --prompt templaté.
fn writeIdsCtxSafetensors(allocator: std.mem.Allocator, io: std.Io, path: []const u8, ids_gen: []const i64, ctx: []const u32) !void {
    const n = ids_gen.len;
    const c = ctx.len;
    const data = try allocator.alloc(i32, n + c);
    defer allocator.free(data);
    for (ids_gen, 0..) |t, k| data[k] = @intCast(t); // ids < vocab 262144 : cast sans perte
    for (ctx, 0..) |t, k| data[n + k] = @intCast(t);
    const header = try std.fmt.allocPrint(allocator, "{{\"ids\":{{\"dtype\":\"I32\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}},\"ctx_ids\":{{\"dtype\":\"I32\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}}}", .{ n, n * 4, c, n * 4, (n + c) * 4 });
    defer allocator.free(header);
    var len_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_le, header.len, .little);
    const f = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer f.close(io);
    try f.writePositionalAll(io, &len_le, 0);
    try f.writePositionalAll(io, header, 8);
    try f.writePositionalAll(io, std.mem.sliceAsBytes(data), 8 + header.len);
}
```

Site d'appel (remplace `:3615-3618`) :

```zig
    if (out_ids_path) |out_path| {
        // K5 : la clé ctx_ids n'apparaît que sous reprise AVEC prompt neuf (ids.len >
        // step_next) — la reprise SIMPLE garde le format historique à une clé, sinon son
        // log changerait et C-K5-E (« chemin actuel à l'identique ») serait violée.
        const k5_resume_prompt = if (resume_state) |rs| ids.len > rs.step_next else false;
        if (k5_resume_prompt) {
            try writeIdsCtxSafetensors(allocator, io, out_path, generated.items, ids);
            log.info("--out-ids : {d} ids générés + ctx_ids {d} écrits -> {s}", .{ generated.items.len, ids.len, out_path });
        } else {
            try writeIdsSafetensors(allocator, io, out_path, generated.items);
            log.info("--out-ids : {d} ids écrits -> {s}", .{ generated.items.len, out_path });
        }
    }
```

- [ ] **Step 3.7 : `--ids-only` couvre le tour 2** — sous `--load-cache` + `--prompt` +
  `--ids-only`, impossible (pas de dump requis pour un rendu !). À la place, flag host-only
  nouveau `--ids-only-turn2` (patron `--ids-only` `:2080-2107`, AVANT la garde VRAM) :

```zig
    if (args.ids_only_turn2) {
        var t2 = try promptToIdsTurn2(allocator, &encoder, prompt_text);
        defer t2.deinit(allocator);
        log.info("ids_turn2 = {any}", .{t2.items});
        return;
    }
```

Déclaration `ids_only_turn2: bool = false` dans `Args`, parsing
`--ids-only-turn2` à côté de `--ids-only`, usage string. C'est la sortie que PF6 compare.
⚠ Placer ce bloc APRÈS la tokenisation de `prompt_text` et AVANT `checkVram`.

- [ ] **Step 3.8 : build + commit**

```bash
ZML_REMOTE=ia@192.168.1.163 ZML_WS=/data/rqz_workspace/zml ./zml_runner/build_3090.sh
git add zml_runner/gemma4_g12auto.zig
git commit -m "k5 : prefill partiel — ids_full = ids_fed ++ [fed_next] ++ tour2, top5@ctx, ctx_ids, --ids-only-turn2"
```

Expected : build OK. Aucun changement `engine.zig` (PF0 le prouvera).

---

## Task 4 : outillage Python — oracle `--context-ids` + dépouilleur 80 + mutant 74

**Files :** Modify `scripts/69_u8_gen_oracle.py`, `scripts/74_kvdump_inspect.py` ;
Create `scripts/80_pf1_bridge.py`, `scripts/79_t2_render_check.py`.

- [ ] **Step 4.1 : mode `--context-ids` du 69**

Ajouts (le gros du mode `--teacher-force` est réutilisé — même prefill unique, même tête
manuelle par chunks, même self-check, même témoin fenêtre) :

```python
ap.add_argument("--context-ids", default=None,
    help="K5 : safetensors du runner (clés ids + ctx_ids) — contexte = ctx_ids, générés = ids ; exclusif de --prompt/--teacher-force")
ap.add_argument("--ctx-from", type=int, default=None,
    help="K5 : step_next — première position (absolue) à rapporter ; obligatoire avec --context-ids")
```

Dans le mode (nouvelle fonction `mode_context`, dérivée de `mode_teacher_force`) :

```python
def mode_context(model, tok, a):
    from safetensors import safe_open
    with safe_open(a.context_ids, framework="pt") as f:
        ctx = f.get_tensor("ctx_ids").tolist()   # ids_full : C tokens feedés avant génération
        gen = f.get_tensor("ids").tolist()       # m tokens générés par le runner
    C, m = len(ctx), len(gen)
    assert a.ctx_from is not None and 0 < a.ctx_from < C, "--ctx-from (step_next) requis, < len(ctx_ids)"
    full = ctx + gen[:-1]                        # position p prédit full[p+1] ; p=T-1 prédit gen[-1]
    T = len(full)
    # …prefill unique par chunks (code du mode teacher-force, borne de départ = a.ctx_from)…
    # Pour CHAQUE position p in [a.ctx_from, T-1], DEUX canaux (finding bloquant de revue —
    # le top-5 du runner est BRUT in-graph, `generated` est POST-politique ; cas mesuré :
    # docs/evidence/kvdump/dc4.err.log:16, argmax brut 258882 supprimé à gen=0) :
    #   argmax_raw / margin_raw / top5_raw          : logits fp32 bruts
    #   argmax_policy / margin_policy               : après SuppressTokensLogitsProcessor
    #     (build_gen_policy existant, 69:73 — le VRAI processor transformers, jamais réécrit)
    # régimes (dépouillés par le 80, PAS ici — l'oracle publie, il ne juge pas) :
    #   p in [ctx_from, C-2]  : positions du tour 2 (prefill de reprise) → canal RAW
    #   p == C-1              : produit la 1re génération (gen[0])       → canal POLICY
    #   p in [C, T-1]         : positions de génération                  → canal POLICY
    report = {"mode": "context", "ctx_from": a.ctx_from, "C": C, "m": m,
              "positions": positions_payload,
              # [{p, argmax_raw, margin_raw, top5_raw_ids, top5_raw_vals,
              #   argmax_policy, margin_policy}]
              "window": window_witness, "prefill_len": T, "versions": versions_payload,
              "script_md5": script_md5()}
```

⚠ Points de vigilance hérités : marges consignées AVANT tout verdict (piège 17) ; chunks
`HEAD_CHUNK=64` (jamais les 600 Mo de logits) ; le self-check tête reste actif ; le témoin
fenêtre publie `bites_in_prefill` (T > 1024 requis pour PF3). Le `--prompt` N'est PAS requis
dans ce mode (le contexte EST `ctx_ids`) — adapter la garde `required=True` de `--prompt`
(`69:574-575` ; la garde `--teacher-force` voisine est à `69:579-581`).

- [ ] **Step 4.2 : dépouilleur `scripts/80_pf1_bridge.py`** (pattern 75 : verdict machine,
  seuils en tête, refus bruyant sur ligne absente ; ⚠ 77 est PRIS par
  `77_ngram_repetition.py`, 78/79 par les scripts de ce chantier)

```python
#!/usr/bin/env python3
"""K5/PF1 — verdict machine : oracle context (69) vs log du runner (top5 @ ctx= / generated).
Le binaire publie des mesures BRUTES ; le juge est ICI, seuils pré-enregistrés (spec §5).
Canaux (spec §4.5) : ctx = BRUT (top-5 in-graph, politique pas encore appliquée) ;
gen = POLICY (`generated` sort de policy.select). Mélanger = FAIL spuré dès qu'une
suppression mord (cas réel : dc4.err.log:16).
Usage : 80_pf1_bridge.py <pf1.json> <runB.err.log> [--expect-fail]"""
import json, re, sys

TIE_MARGIN = 1.873e-3        # spec generation-config §2bis (2× bruit U7) — hérité, pas réglé ici
FAT_FACTOR = 10.0            # PF2 : un mismatch "gras" a une marge > 10× TIE_MARGIN

RE_BUILD = re.compile(r"BUILD: mode=(\w+)")
RE_KVLOAD = re.compile(r"KVLOAD: .* step_next=(\d+) fed_next=(\d+)")
RE_CTX = re.compile(r"top5 @ ctx=(\d+) : idx=\{ ([\d, ]+) \}")
RE_GEN = re.compile(r"generated = \{ ([\d, ]+) \}")

def need(m, what):
    if not m: sys.exit(f"DEPOUILLEMENT IMPOSSIBLE : {what} absent du log — jamais un zéro")
    return m

log = open(sys.argv[2]).read()
mode = need(RE_BUILD.search(log), "BUILD: mode=").group(1)
if mode != "ReleaseFast": sys.exit(f"INEXECUTABLE : BUILD mode={mode}")
step_next = int(need(RE_KVLOAD.search(log), "KVLOAD:").group(1))
ctx_lines = [(int(p), [int(x) for x in idx.split(",")]) for p, idx in RE_CTX.findall(log)]
generated = [int(x) for x in need(RE_GEN.search(log), "generated =").group(1).split(",")]

j = json.load(open(sys.argv[1]))
C = j["C"]
pos = {int(e["p"]): e for e in j["positions"]}
fails = []
# Régime (i) ctx : BRUT↔BRUT, appariement ORDINAL (le mutant PF2 décale les positions
# absolues ; l'ordinal compare quand même et le mismatch de CONTENU fait foi).
oracle_ctx = [pos[p] for p in sorted(pos) if p <= C - 2]
if len(ctx_lines) != len(oracle_ctx):
    fails.append(f"n_ctx runner={len(ctx_lines)} != oracle={len(oracle_ctx)} margin=999")
for k, ((p_run, idx_run), e) in enumerate(zip(ctx_lines, oracle_ctx)):
    if idx_run[0] != e["argmax_raw"]:
        fails.append(f"ctx[{k}] p_run={p_run} p_hf={e['p']} zml={idx_run[0]} hf={e['argmax_raw']} margin={e['margin_raw']:.6f}")
# Régime (i) suite : la 1re génération — POLICY↔POLICY
e0 = pos.get(C - 1)
if e0 is None: sys.exit("DEPOUILLEMENT IMPOSSIBLE : position C-1 absente du rapport oracle")
if generated[0] != e0["argmax_policy"]:
    fails.append(f"gen[0] zml={generated[0]} hf={e0['argmax_policy']} margin={e0['margin_policy']:.6f}")
# Régime (ii) : générations suivantes — POLICY↔POLICY, première divergence à marge <= TIE
# tolérée (bistabilité, régime DC3)
tie_note = None
for k in range(1, len(generated)):
    e = pos.get(C - 1 + k)
    if e is None: break
    if generated[k] != e["argmax_policy"]:
        if e["margin_policy"] <= TIE_MARGIN:
            tie_note = f"tie @ gen={k} margin={e['margin_policy']:.6f} <= {TIE_MARGIN} (publié, régime DC3)"
        else:
            fails.append(f"gen[{k}] zml={generated[k]} hf={e['argmax_policy']} margin={e['margin_policy']:.6f} > TIE")
        break
verdict_fail = bool(fails)
print(f"step_next={step_next} C={C} n_ctx={len(ctx_lines)} gen={len(generated)}")
if tie_note: print(tie_note)
for f_ in fails: print("MISMATCH:", f_)
if "--expect-fail" in sys.argv:
    fat = any(float(f_.split("margin=")[1].split()[0]) > FAT_FACTOR * TIE_MARGIN for f_ in fails)
    if verdict_fail and fat: print("PF2 : le mutant MORD (mismatches gras) — attendu"); sys.exit(0)
    sys.exit("PF2 : le mutant NE MORD PAS — le gate ne prouve rien, STOP diagnostic")
sys.exit(1 if verdict_fail else 0)
```

⚠ Le format `{any}` de Zig imprime `{ 1, 2, 3 }` — vérifier la regex `RE_CTX`/`RE_GEN`
contre un log réel AVANT de déclarer le moindre verdict, et **contre-prouver le
dépouilleur** (leçon « le juge doit être vu condamner ») : injecter à la main dans une copie
du log un idx faux → le 80 doit sortir MISMATCH ; retirer une ligne ctx → n_ctx doit FAIL.

- [ ] **Step 4.3 : sous-commande `shift-fwd` du 74** (mutant PF2)

```python
def cmd_shift_fwd(args):
    """K5/PF2 — forge un dump qui MENT d'une position : step_next+1, ids_fed étendu d'un id
    fantôme (fed_next recopié), fed_next inchangé. Manifest COHÉRENT (ids_fed.len ==
    step_next tient), caches INTACTS : le manifest déclare un token que le cache ne porte
    PAS — la position step_next du cache est restée aux zéros du dump (mécanisme DC4
    localisé à UNE position). ⚠ shift-BACK (tronquer) a été analysé et REJETÉ : l'état
    tronqué est auto-cohérent, le run forgé réécrirait le slot à l'identique — mordant nul
    plausible ET sain (spec C-K5-B). Bas niveau struct+json, comme make-zeroed."""
    hdr, meta, tensors = read_raw(args.src)          # helpers existants du 74
    import numpy as np
    step_next = int(meta["step_next"]) + 1
    ids = tensors["ids_fed"]                          # np.int32 [step_next-1]
    phantom = np.int32(int(meta["fed_next"]))         # id valide (< vocab par construction)
    tensors["ids_fed"] = np.append(ids, phantom)
    meta["step_next"] = str(step_next)
    meta["ids_fed_xxh64"] = xxh64_hex(tensors["ids_fed"].tobytes())
    write_raw(args.dst, meta, tensors)
    print(f"shift-fwd : {args.src} -> {args.dst} step_next={step_next} phantom={int(phantom)}")
```

CLI **positionnelle**, comme les sous-commandes existantes du 74 (`set-meta <file> <out>
<key> <value>` à `74:312-317`, `make-fixture <n> <p0> <out> [--source]` à `74:331-336`) :
`shift-fwd <src> <dst>`. (Adapter aux helpers réels — `inspect`/`make-zeroed`/`set-meta`
montrent le patron bas niveau ; recalculer UNIQUEMENT `ids_fed_xxh64`, les 4 caches
gardent leurs checksums.)

- [ ] **Step 4.4 : comparateur PF6 `scripts/79_t2_render_check.py`**

```python
#!/usr/bin/env python3
"""K5/PF6 — le rendu Zig du tour 2 == le suffixe HF mesuré (Task 2), en ids, littéralement.
Usage : 79_t2_render_check.py <rendu_tour2_hf.json> <ids_only_turn2.log>"""
import json, re, sys
j = json.load(open(sys.argv[1]))
log = open(sys.argv[2]).read()
m = re.search(r"ids_turn2 = \{ ([\d, ]+) \}", log)
if not m: sys.exit("DEPOUILLEMENT IMPOSSIBLE : ids_turn2 absent")
zig = [int(x) for x in m.group(1).split(",")]
hf = j["suffix_ids"]
if zig == hf: print(f"PF6 PASS — {len(zig)} ids identiques"); sys.exit(0)
k = next((i for i, (a, b) in enumerate(zip(zig, hf)) if a != b), min(len(zig), len(hf)))
sys.exit(f"PF6 FAIL — 1er écart à l'index {k} : zig={zig[k:k+4]} hf={hf[k:k+4]} (len {len(zig)} vs {len(hf)})")
```

⚠ Le `user2` passé au runner (`--ids-only-turn2 --prompt "<user2>"`) doit être LE MÊME
texte que le `--user2` de la Task 2 — sinon la comparaison est vide de sens.

- [ ] **Step 4.5 : commit**

```bash
git add scripts/69_u8_gen_oracle.py scripts/74_kvdump_inspect.py scripts/80_pf1_bridge.py scripts/79_t2_render_check.py
git commit -m "k5 : oracle --context-ids (raw+policy), dépouilleur 80 (verdict machine), mutant shift-fwd, check PF6"
```

⚠ **Contre-prouver le 80 AVANT tout verdict** (« le juge doit être vu condamner ») : sur
une copie du log réel de PF1, (a) altérer un `idx=` d'une ligne ctx → MISMATCH attendu ;
(b) supprimer une ligne ctx → `n_ctx` FAIL attendu ; (c) altérer un id de `generated` →
MISMATCH gen attendu. Archiver les trois sorties dans `docs/evidence/k5/80_selfproof.log`.

---

## Task 5 : gate PF6 — rendu tour 2 prouvé en ids

**Files :** `docs/evidence/k5/pf6_*.log`.

- [ ] **Step 5.1 : rendu Zig, comparaison, evidence**

```bash
# VM (host-only, pas de GPU) :
$B1 $W/model.safetensors $W/tokenizer.json --ids-only-turn2 --prompt "What is my name?" \
  > $K/pf6.out.log 2> $K/pf6.err.log
md5sum ./bazel-bin/examples/rqz/gemma4_g12auto >> $K/pf6.err.log
# M1 :
scp ia@192.168.1.163:/data/gemma4-zml-probe/k5/pf6.err.log docs/evidence/k5/
python3 scripts/79_t2_render_check.py docs/evidence/k5/rendu_tour2_hf.json docs/evidence/k5/pf6.err.log
```

Expected : `PF6 PASS — N ids identiques`. FAIL ⇒ corriger le littéral
`renderChatTemplateTurn2` d'après `text_b` mesuré, rebuild, re-run — le littéral n'a pas
le droit d'être « presque bon ».

- [ ] **Step 5.2 : commit + tag**

```bash
git add docs/evidence/k5/ && git commit -m "gate(pf6) : rendu tour 2 == suffixe HF mesuré, en ids"
git tag gate/pf6-pass && git push origin k5-prefill-partiel gate/pf6-pass
```

---

## Task 6 : gates PF0 (graphe intact) + PF5 (non-régression)

**Files :** `docs/evidence/k5/pf0_*.md5`, `docs/evidence/k5/pf5_*.log`.

- [ ] **Step 6.1 : PF0 — HLO après**

```bash
# VM :
XLA_FLAGS="--xla_dump_to=/data/gemma4-zml-probe/k5/hlo_apres" \
  $B1 $W/model.safetensors $W/tokenizer.json --prompt "witness" --max-tokens 4 \
  > $K/pf0.out.log 2> $K/pf0.err.log
md5sum /data/gemma4-zml-probe/k5/hlo_apres/*before_optimizations.txt
git diff main -- zml_runner/engine.zig   # sur M1 : DOIT être vide
```

Expected : md5 == `297679847aa04b719942d75d093adf2b`, diff engine.zig vide.

- [ ] **Step 6.2 : PF5 — selftest kvdump-eq inchangé, 1280 ET 4k**

```bash
# VM (le selftest exige --prompt ; reprendre l'invocation du gate DC2,
# docs/superpowers/plans/2026-08-09-kv-cache-dump-restore.md) :
$B1 $W/model.safetensors $W/tokenizer.json --selftest-kvdump-eq $K/pf5_eq_1280.kvdump \
  --prompt "Explain why the sky is blue." > $K/pf5_1280.out.log 2> $K/pf5_1280.err.log
$B4 $W/model.safetensors $W/tokenizer.json --selftest-kvdump-eq $K/pf5_eq_4k.kvdump \
  --prompt "Explain why the sky is blue." > $K/pf5_4k.out.log 2> $K/pf5_4k.err.log
grep -E "BUILD: mode=|32/32|ALLOC-LOOP" $K/pf5_1280.err.log $K/pf5_4k.err.log
```

Expected : `32/32` bit-identiques sur les DEUX variantes, `ALLOC-LOOP: alloc=0`.

- [ ] **Step 6.3 : PF5 — reprise simple inchangée** (un `--load-cache` sans `--prompt`)

```bash
$B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/pf5_eq_1280.kvdump --max-tokens 8 \
  > $K/pf5_resume.out.log 2> $K/pf5_resume.err.log
grep -E "KVLOAD:|KVLOAD-PERF|K5:" $K/pf5_resume.err.log
```

Expected : `KVLOAD:` présents, **aucune ligne `K5:`** (le chemin K5 ne s'active pas sans
prompt), génération normale.

- [ ] **Step 6.4 : commit + tags**

```bash
git add docs/evidence/k5/ && git commit -m "gates(pf0,pf5) : graphe intact, reprise simple et selftest kvdump-eq inchangés"
git tag gate/pf0-pass gate/pf5-pass && git push origin k5-prefill-partiel gate/pf0-pass gate/pf5-pass
```

---

## Task 7 : gate PF1 — équivalence teacher-forcée à la frontière (scénario court)

**Files :** `docs/evidence/k5/pf1_*.{log,json}`.

Scénario (spec §4.5) : P1 discriminant (leçon RP4c — le tour 2 doit DÉPENDRE du contexte),
k = 16, n_new ≈ 20, m = 32.

- [ ] **Step 7.1 : run A — contexte + dump**

```bash
# VM :
$B1 $W/model.safetensors $W/tokenizer.json \
  --prompt "My name is Aldebaran and I live in a lighthouse. Tell me a story about my home." \
  --max-tokens 16 --dump-cache $K/pf1.kvdump --dump-top5 \
  > $K/pf1_runA.out.log 2> $K/pf1_runA.err.log
md5sum ./bazel-bin/examples/rqz/gemma4_g12auto >> $K/pf1_runA.err.log
grep -E "KVDUMP:|BUILD: mode=" $K/pf1_runA.err.log
```

Noter `step_next` de la ligne `KVDUMP:`. Détokeniser la sortie du run A (elle est dans
`pf1_runA.out.log`) : c'est le `--assistant1` déjà utilisé en Task 2 — **si la Task 2 a été
faite avec un autre texte, la REFAIRE avec celui-ci** (l'ordre naturel : Task 7.1 peut être
exécutée AVANT la Task 2 pour produire ce texte ; seule la mesure Task 2 doit précéder le
littéral Zig de Task 3).

- [ ] **Step 7.2 : run B — restore + tour 2 + témoins**

```bash
$B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/pf1.kvdump \
  --prompt "What is my name?" --max-tokens 32 --dump-top5 --out-ids $K/pf1_outB.safetensors \
  > $K/pf1_runB.out.log 2> $K/pf1_runB.err.log
md5sum ./bazel-bin/examples/rqz/gemma4_g12auto >> $K/pf1_runB.err.log
grep -cE "top5 @ ctx=" $K/pf1_runB.err.log   # attendu : n_new + 1 (tour 2 + clôture eot
                                              # injectée — le run A s'arrête sur max_tokens,
                                              # fed_next n'est pas un EOS, D-K5-5 injecte)
grep -E "K5:|KVLOAD:|ALLOC-LOOP" $K/pf1_runB.err.log
```

- [ ] **Step 7.3 : oracle fp32 (M4)**

```bash
scp ia@192.168.1.163:/data/gemma4-zml-probe/k5/pf1_outB.safetensors /tmp/
scp /tmp/pf1_outB.safetensors macmini:/tmp/ && scp scripts/69_u8_gen_oracle.py macmini:/tmp/
ssh macmini '~/ml-venvs/g12b/bin/python3 /tmp/69_u8_gen_oracle.py \
  --weights ~/ml-data/weights_12b_dq --compute-fp32 \
  --context-ids /tmp/pf1_outB.safetensors --ctx-from <step_next du run B> \
  --out /tmp/pf1.json'
scp macmini:/tmp/pf1.json docs/evidence/k5/
```

- [ ] **Step 7.4 : verdict machine + discriminance**

```bash
scp ia@192.168.1.163:/data/gemma4-zml-probe/k5/pf1_runB.err.log docs/evidence/k5/
python3 scripts/80_pf1_bridge.py docs/evidence/k5/pf1.json docs/evidence/k5/pf1_runB.err.log
python3 - <<'EOF'
import json, statistics
j = json.load(open("docs/evidence/k5/pf1.json"))
raw = [e["margin_raw"] for e in j["positions"] if e["p"] <= j["C"] - 2]
pol = [e["margin_policy"] for e in j["positions"] if e["p"] >= j["C"] - 1]
print(f"marges ctx (raw)   : min={min(raw):.6f} médiane={statistics.median(raw):.6f}")
print(f"marges gen (policy): min={min(pol):.6f} médiane={statistics.median(pol):.6f}")
EOF
```

Expected : exit 0, zéro MISMATCH — régime (i) : `n_new + clôture + 1` comparaisons
(les positions ctx, clôture injectée incluse, + la 1ʳᵉ génération ; le comptage du 80 est
auto-cohérent, les deux côtés comptent les mêmes positions) ; médiane des marges ≥ 10×
1,873e-3 (discriminance §2bis C-K5-B — sinon le scénario est mou : changer P1/P2 AVANT
PF2, et le dire). Vérifier dans la sortie du run B que la réponse contient bien le
contexte (« Aldebaran ») — la capacité, pas seulement l'équivalence.

- [ ] **Step 7.5 : commit + tag** (⚠ le tag n'est posé que si PF2 mordra — la non-vacuité
  transversale de la spec §5 ; poser le tag en Task 8 est acceptable et plus honnête)

```bash
git add docs/evidence/k5/ && git commit -m "gate(pf1) : équivalence teacher-forcée à la frontière — verdict machine, marges publiées"
```

---

## Task 8 : gate PF2 — le mordant (mutant shift-fwd)

**Files :** `docs/evidence/k5/pf2_*.log`.

- [ ] **Step 8.1 : forger + rejouer**

```bash
# M1 → VM (le 74 tourne où vit le dump) :
scp scripts/74_kvdump_inspect.py ia@192.168.1.163:/data/gemma4-zml-probe/
# VM :
python3 /data/gemma4-zml-probe/74_kvdump_inspect.py shift-fwd $K/pf1.kvdump $K/pf2_shift.kvdump
$B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/pf2_shift.kvdump \
  --prompt "What is my name?" --max-tokens 32 --dump-top5 --out-ids $K/pf2_outB.safetensors \
  > $K/pf2_runB.out.log 2> $K/pf2_runB.err.log
md5sum ./bazel-bin/examples/rqz/gemma4_g12auto >> $K/pf2_runB.err.log
```

- [ ] **Step 8.2 : verdict — le mutant DOIT mordre contre l'oracle NOMINAL**

```bash
scp ia@192.168.1.163:/data/gemma4-zml-probe/k5/pf2_runB.err.log docs/evidence/k5/
python3 scripts/80_pf1_bridge.py docs/evidence/k5/pf1.json docs/evidence/k5/pf2_runB.err.log --expect-fail
```

⚠ La comparaison est CROISÉE À DESSEIN : le run forgé est dépouillé contre le `pf1.json`
**nominal** — c'est exactement la situation qu'un bug de position produirait (logits d'un
état faussé confrontés à la référence attendue). Le forgé attend à un slot cache JAMAIS
ÉCRIT (zéros) : divergence grasse attendue dès les premières positions ctx (mécanisme DC4
localisé). Expected : `PF2 : le mutant MORD (mismatches gras)`, exit 0. S'il ne mord pas :
**STOP diagnostic** — et souvenir RP4(c) : un mordant nul condamne l'ANTÉCÉDENT (le
scénario), pas la corruption.

- [ ] **Step 8.3 : re-passer le nominal (même binaire), commit + tags**

```bash
python3 scripts/80_pf1_bridge.py docs/evidence/k5/pf1.json docs/evidence/k5/pf1_runB.err.log
git add docs/evidence/k5/ && git commit -m "gate(pf2) : le mutant shift-fwd mord — PF1 non vacueux"
git tag gate/pf1-pass gate/pf2-pass && git push origin k5-prefill-partiel gate/pf1-pass gate/pf2-pass
```

---

## Task 9 : gate PF3 — fenêtre sliding à travers la frontière + M-K5-1

**Files :** `docs/evidence/k5/pf3_*.{log,json}`, `docs/evidence/k5/mk51_prediction.md`.

- [ ] **Step 9.1 : écrire la prédiction M-K5-1 AVANT les runs**

`docs/evidence/k5/mk51_prediction.md` — **les DEUX métriques, publiées ensemble** (spec
§2bis : l'une sans l'autre serait un chiffre juste et trompeur) : ≈ **×26 en positions
évitées** ((1005+40)/40) ET ≈ **×14 en temps de steps** (coût run B ≈ 73/9,6 ≈ 7,6 s de
steps vs re-prefill complet ~1078 pos ≈ 112 s ; le chargement des 840 Mio s'y ajoute,
KVLOAD-PERF le mesure). Committer AVANT d'exécuter.

- [ ] **Step 9.2 : run A long — contexte @~1005 par oracle borné** (l'EOS couperait un run
  libre bien avant — méthode DC7)

```bash
# VM — p0 = nombre d'ids du prompt rendu (ligne `ids = { ... }` du mode --ids-only) :
p0=$($B1 $W/model.safetensors $W/tokenizer.json --ids-only \
  --prompt "My name is Aldebaran and I live in a lighthouse. Tell me a story about my home." \
  2>&1 | grep -o "ids = {[^}]*}" | tr ',' '\n' | wc -l)
echo "p0=$p0"    # sanity : ~25-35 ids ; 0 ou vide = STOP, le grep n'a pas matché
# CLI POSITIONNELLE du 74 (74:331-336) : make-fixture <n> <p0> <out> [--source] ;
# --source défaut /data/gemma4-zml-probe/u9_ids.safetensors (présent sur la VM, ~3900 ids)
scp scripts/74_kvdump_inspect.py ia@192.168.1.163:/data/gemma4-zml-probe/  # si pas déjà fait
python3 /data/gemma4-zml-probe/74_kvdump_inspect.py make-fixture \
  $((1005 - p0)) $p0 $K/pf3_fixture.safetensors
nohup $B1 $W/model.safetensors $W/tokenizer.json \
  --prompt "My name is Aldebaran and I live in a lighthouse. Tell me a story about my home." \
  --oracle $K/pf3_fixture.safetensors --dump-cache $K/pf3.kvdump \
  > $K/pf3_runA.out.log 2> $K/pf3_runA.err.log ; echo DONE rc=$?
```

⚠ Le verdict A1 du run A sortira vraisemblablement en `A1Mismatch` (fixture factice, méthode
DC7) : le dump et `KVDUMP:` sont émis AVANT lui — accepté et noté. Vérifier
`step_next ≈ 1005` sur la ligne `KVDUMP:`. (~105 s de steps, nohup.)

- [ ] **Step 9.3 : run B long + oracle + verdict**

```bash
# VM :
$B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/pf3.kvdump \
  --prompt "Describe the door of my home, and remind me of my name." --max-tokens 32 \
  --dump-top5 --out-ids $K/pf3_outB.safetensors \
  > $K/pf3_runB.out.log 2> $K/pf3_runB.err.log
md5sum ./bazel-bin/examples/rqz/gemma4_g12auto >> $K/pf3_runB.err.log
# M1 — rapatrier PUIS pousser vers M4 (chaîne complète, patron Task 7.3) :
scp ia@192.168.1.163:/data/gemma4-zml-probe/k5/pf3_outB.safetensors /tmp/
scp ia@192.168.1.163:/data/gemma4-zml-probe/k5/pf3_runB.err.log docs/evidence/k5/
scp /tmp/pf3_outB.safetensors macmini:/tmp/
# M4 — nohup OBLIGATOIRE : prefill fp32 CPU à T≈1078 ≈ 15× l'oracle du scénario court
# (grandeur prédite spec §2bis) :
ssh macmini 'nohup ~/ml-venvs/g12b/bin/python3 /tmp/69_u8_gen_oracle.py \
  --weights ~/ml-data/weights_12b_dq --compute-fp32 \
  --context-ids /tmp/pf3_outB.safetensors --ctx-from <step_next du run B> \
  --out /tmp/pf3.json > /tmp/pf3_oracle.log 2>&1; echo DONE rc=$?'
scp macmini:/tmp/pf3.json docs/evidence/k5/
# M1 :
python3 scripts/80_pf1_bridge.py docs/evidence/k5/pf3.json docs/evidence/k5/pf3_runB.err.log
python3 -c "import json; print(json.load(open('docs/evidence/k5/pf3.json'))['window'])"
```

Expected : exit 0 **ET** `bites_in_prefill: true` (T ≈ 1078 > 1024 — la fenêtre a mordu,
sinon INEXÉCUTABLE : redimensionner, pas PASS).

- [ ] **Step 9.4 : M-K5-1 — le chiffre de la capacité**

```bash
grep -E "KVLOAD-PERF|PERF-RESUME" $K/pf3_runB.err.log
# re-prefill complet équivalent (baseline) : re-jouer ids_full par oracle borné
# (fixture make-fixture --n 73 --p0 1046 sur le MÊME prompt concaténé n'est PAS
# constructible par --prompt : publier la baseline par l'arithmétique du plan DC7 —
# temps(run A, ~980 steps) × (1078/1005) — et le DIRE dans K5_RESULTS §mesures)
```

Publier : temps run B total vs baseline re-calcul, KVLOAD-PERF, tok/s. Sans verdict ni tag.

- [ ] **Step 9.5 : commit + tag**

```bash
git add docs/evidence/k5/ && git commit -m "gate(pf3) : fenêtre sliding mordante à travers la frontière + mesure M-K5-1"
git tag gate/pf3-pass && git push origin k5-prefill-partiel gate/pf3-pass
```

---

## Task 10 : gate PF4 — les refus, chacun VU échouer

**Files :** `docs/evidence/k5/pf4_refus.log`.

⚠ Dépendances RÉELLES (le diagramme le note) : tous les cas exigent `$K/pf1.kvdump`
(créé en Task 7.1) et le cas (e) exige le 74 déjà scp'é sur la VM (Task 8.1 ou 9.2).

- [ ] **Step 10.1 : exercer les 5 cas (spec §4.6)**

```bash
{ # (a) tour 2 >= fenêtre : prompt de ~1100 mots
  $B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/pf1.kvdump \
    --prompt "$(python3 -c "print('word ' * 1100)")" --max-tokens 4 ; echo "rc(a)=$?"
  # (b) place insuffisante : max-tokens énorme
  $B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/pf1.kvdump \
    --prompt "What is my name?" --max-tokens 1300 ; echo "rc(b)=$?"
  # (c) ignore-prompt + load-cache (garde CONSERVÉE)
  $B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/pf1.kvdump \
    --prompt "x" --repetition-penalty 1.15 --ignore-prompt --max-tokens 4 ; echo "rc(c)=$?"
  # (d) prompt VIDE sous reprise (garde prompt_text.len == 0 AVANT rendu, Step 3.3 —
  # le parseur accepte "" : args.prompt != null, prompt_text.len == 0, vérifié :258-264)
  $B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/pf1.kvdump \
    --prompt "" --max-tokens 4 ; echo "rc(d)=$?"
  # (e) oracle p0 faux sous reprise+prompt — CLI POSITIONNELLE (74:331-336)
  python3 /data/gemma4-zml-probe/74_kvdump_inspect.py make-fixture 8 7 $K/pf4_p0faux.safetensors
  $B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/pf1.kvdump \
    --prompt "What is my name?" --oracle $K/pf4_p0faux.safetensors ; echo "rc(e)=$?"
} > $K/pf4_refus.log 2>&1
grep -E "rc\(|error\." $K/pf4_refus.log
```

Expected : (a) `PromptTooLong`, (b) `SequenceTooLong`, (c) `IgnorePromptWithLoadCache`,
(d) `PromptTooLong` (prompt vide — garde AVANT rendu ; l'ancienne formulation « tour 2
vide après rendu » était à antécédent vide, le rendu émettant toujours ses marqueurs de
tour), (e) `OraclePromptMismatch` — tous rc != 0, AUCUN crash non qualifié.
⚠ Le cas « fed_next forgé » n'est PAS un refus : limite documentée (spec §4.6), c'est
l'aller-retour oracle qui le voit — PF2 en est la démonstration.

- [ ] **Step 10.2 : commit + tag**

```bash
git add docs/evidence/k5/ && git commit -m "gate(pf4) : 5 refus exercés, chacun VU échouer"
git tag gate/pf4-pass && git push origin k5-prefill-partiel gate/pf4-pass
```

---

## Task 11 : clôture — docs, README, PLANNING, PR

**Files :** Create `docs/K5_RESULTS.md` ; Modify `docs/DOCUMENTATION.md`, `README.md`,
`PLANNING.md`.

- [ ] **Step 11.1 : `docs/K5_RESULTS.md`** — table gate/critère/mesuré/commit/tag (modèle
  `docs/KVDUMP_RESULTS.md`), les 6 claims §2bis jugées une à une, M-K5-1 publiée avec sa
  prédiction, dettes : gate teacher-forcé 4k (D-K5-4), garde fenêtre transposée non levée
  (D-K5-3), K4 (résident multi-tour) débloqué, coût du chaînage v1 (relecture ~840 Mio/tour).
- [ ] **Step 11.2 : `docs/DOCUMENTATION.md`** — corriger `:252` (`--load-cache --oracle`
  n'est PAS teacher-forcé — fait §3.2 du cadrage, `fed = tok` `:3444`) ; documenter la
  capacité (usage, sémantique ids_full, chaînage dump→restore→tour) ; piège nouveau si
  découvert en route.
- [ ] **Step 11.3 : `README.md`** (anglais, GC11 : marqueur à portée constante) — milestone
  K5, usage 2 lignes, et RAPPELER l'écart E2B (chaque capacité 12B le creuse — K3 au
  PLANNING).
- [ ] **Step 11.4 : `PLANNING.md`** — K5 soldée, K4 débloquée (référencer la fiche 2 du
  cadrage), dettes ci-dessus datées.
- [ ] **Step 11.5 : grep anonymisation APRÈS la dernière écriture, PR, tags**

```bash
git grep -nE 'Users/regis|macmini|192\.168' -- ':!docs/superpowers'   # DOIT être vide
git push origin k5-prefill-partiel
gh pr create --title "K5 : prefill partiel — reprendre un cache et feeder un prompt neuf" \
  --body "7 gates verts (PF0-PF6), spec 2026-08-10-k5-prefill-partiel-design.md, M-K5-1 publiée.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
# merge --no-ff sur GO Régis UNIQUEMENT, puis vérifier SUR LA CIBLE (HEAD origin/main relu,
# gate PF6 rejouable) et re-vérifier les tags : git tag --merged main | grep pf
```

- [ ] **Step 11.6 : fiche mémoire + handoff** — mettre à jour
  `~/dev/Ma_MEMOIRE/memory/project_gemma4_zml_probe.md` (session, verdicts, leçons neuves).

---

## Ordre d'exécution

```
Task 0 (décisions) ─→ Task 1 (témoins AVANT)
                          │
        ┌─────────────────┤
        ▼                 ▼
   Task 7.1 (run A)   Task 2 (mesure rendu HF — a besoin du texte du run A)
        │                 │
        └────────┬────────┘
                 ▼
            Task 3 (Zig)  ──→ Task 4 (Python)     [parallélisables après Task 2]
                 │                  │
                 └───────┬──────────┘
                         ▼
   Task 5 (PF6) → Task 6 (PF0+PF5) → Task 7.2-7.5 (PF1) → Task 8 (PF2)
                                                              │
                                            Task 9 (PF3+M-K5-1) ← peut suivre PF1
                                                              │
                       Task 10 (PF4) ─────────────────────────┘
                       [exige pf1.kvdump (7.1) + le 74 scp'é sur la VM (8.1/9.2)]
                                                              ▼
                                                        Task 11 (clôture)
```

⚠ Le run A de PF1 (7.1) précède la Task 2 : le `--assistant1` de la mesure HF doit être le
texte RÉELLEMENT généré, pas une invention. Tout FAIL ⇒ STOP + diagnostic, jamais de
requalification à chaud.
