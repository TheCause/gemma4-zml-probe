# Repetition penalty (phase 1) — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Donner au 12B une repetition penalty HF-compatible côté host, prouvée `ids == HF`, sans toucher au graphe ni à `engine.zig`.

**Architecture:** Les logits complets sortent déjà du graphe de `gemma4_g12auto` (3ᵉ sortie, `:830`) et sont post-softcap (`engine.zig:807`) — donc exactement l'objet que HF passe à ses `LogitsProcessor`. Un module `sampling.zig` sans dépendance ZML applique la penalty puis l'argmax côté host. À `penalty == 1.0` les logits ne sont pas lus : le chemin greedy actuel est strictement inchangé.

**Tech Stack:** Zig (std.Io 0.16), ZML/PJRT CUDA, bazel via `./bazel.sh` dans l'arbre ZML, Python + transformers pour l'oracle.

**Spec:** `docs/superpowers/specs/2026-07-27-sampling-repetition-penalty-design.md` (rév. 3, `717380b`)

---

## Amendement 1 à la spec (à porter dans la spec avant de commencer)

La spec §5 RP1 dit « `zig test sampling.zig` ». **Cet outillage n'existe pas** :
`zml_runner/BUILD.bazel:1` ne charge que `zig_binary` (`@rules_zig//zig:defs.bzl`), il n'y a
aucun target de test, et `zig` est absent du PATH de la machine de dev.

**RP1 est donc réalisé par un selftest**, sur le pattern déjà éprouvé du repo :
`--selftest-inputs` (`gemma4_g12auto.zig:449`, « indépendant du prompt/tokenizer/poids
(fixture only) », early-return avant toute init GPU). Nouveau drapeau :
**`--selftest-penalty <fixture>`**. Mêmes propriétés que visées par la spec : pas de GPU, pas de
poids, exécution instantanée. Un `zig_test` bazel reste possible plus tard (charger `zig_test`
dans le BUILD) ; ce n'est pas un prérequis et ce n'est pas éprouvé ici.

---

## Conventions d'exécution (à lire avant la Task 0)

**Déploiement / build.** `zml_runner/deploy_to_3090.sh` est **anonymisé** : il lit
`ZML_REMOTE`, `ZML_JUMP`, `ZML_DST` depuis l'environnement (défauts factices `user@gpu-host`,
`/path/to/zml/examples/rqz`). **Ne jamais écrire les valeurs réelles dans un fichier du repo** —
les exporter dans le shell. Build sur la machine GPU : `./bazel.sh build //examples/rqz:<cible>`.

**Runs distants longs** (leçon payée deux fois au gate U7) : **toujours** `nohup` + log distant +
**stdin fermé** (`< /dev/null`). Un run de génération 12B ne se lance jamais en avant-plan sur
une session SSH.

**Guetteurs de logs** : filtrer sur des **états terminaux uniquement** (`PASS`, `FAIL`, `error`).
Ne jamais filtrer sur un identifiant de gate — au chantier U7 la chaîne « U7 » matchait la
bannière de démarrage et faisait croire à une terminaison.

**Dumps HLO** : `XLA_FLAGS=--xla_dump_to=<dir>`, et le gold est le **pré-opt** — jamais le
post-opt (piège 15, cf `docs/BATCHING_RESULTS.md:37`).

---

## Task 0 : Témoins AVANT toute édition (RP0-a, RP2-témoin)

> Cette tâche vient en premier et ne peut pas être rattrapée plus tard. Au gate M0 du chantier
> masques, un témoin pris après une édition partielle avait produit un état hybride
> (moteur édité + runner pré-bascule) qui a coûté une investigation entière.

**Files:** aucun fichier du repo modifié — on produit des artefacts hors arbre.

- [ ] **Step 1 : vérifier que le worktree est propre et noter le HEAD**

```bash
cd ~/dev/gemma4-zml-probe
git status --short          # attendu : vide
git rev-parse --short HEAD  # noter la valeur, elle identifie les témoins
```

- [ ] **Step 2 : déployer l'état non modifié sur la machine GPU**

```bash
export ZML_REMOTE=...   # jamais committé
export ZML_DST=...
./zml_runner/deploy_to_3090.sh
```

- [ ] **Step 3 : builder et dumper le HLO témoin**

```bash
ssh "$ZML_REMOTE" 'cd "$ZML_DST/../.." && ./bazel.sh build //examples/rqz:gemma4_g12auto'
ssh "$ZML_REMOTE" 'XLA_FLAGS=--xla_dump_to=/tmp/hlo_before \
  nohup ./bazel-bin/examples/rqz/gemma4_g12auto <args habituels> \
  > /tmp/witness_before.log 2>&1 < /dev/null &'
```

Attendu : `/tmp/hlo_before` non vide (le compter : `ls /tmp/hlo_before | wc -l`, ordre de
grandeur ~1000 fichiers d'après `ENGINE_LOG.md:92`). **Un dossier vide invaliderait RP0**
(`diff -rq` de deux dossiers vides est vide) — c'est la garde de la spec.

- [ ] **Step 4 : produire le témoin d'ids de RP2**

Générer 48 tokens en greedy sur le prompt de référence et **archiver la sortie**. C'est *ce*
fichier qui sert de témoin à RP2 — **jamais `u8_gen48`**, contre lequel le runner marque 42/48
en bf16 (spec F6).

- [ ] **Step 5 : commit du protocole de témoin**

```bash
git add docs/superpowers/plans/2026-07-27-sampling-repetition-penalty.md
git commit -m "plan(sampling): plan d'implémentation + amendement 1 (RP1 par selftest, pas zig test)"
```

---

## Task 1 : RP-1 — refonder l'oracle de décode en fp32

> Prérequis absolu (spec C6/F6). Tant que RP-1 n'est pas PASS, la penalty n'est pas armée :
> l'oracle bf16 fabrique des ties artificiels, et la penalty en fabriquerait davantage.

**Files:**
- Modify: `scripts/69_u8_gen_oracle.py:371-372` (restriction `--compute-fp32`), `:148` (assert bf16)

- [ ] **Step 1 : lire les deux points de blocage**

```bash
sed -n '369,373p;145,150p' scripts/69_u8_gen_oracle.py
```

Attendu : `ap.error("--compute-fp32 n'existe qu'en --teacher-force …")` et
`assert out.logits.dtype == torch.bfloat16`.

- [ ] **Step 2 : lever la restriction et rendre l'assert conditionnel**

Remplacer l'`ap.error` par une autorisation du mode décode, et l'assert par un contrôle du dtype
**attendu** :

```python
expected_dtype = torch.float32 if args.compute_fp32 else torch.bfloat16
assert out.logits.dtype == expected_dtype, f"logits {out.logits.dtype} != {expected_dtype}"
```

Les hooks fp32 existent déjà (`install_fp32_hooks`, `:69-93`) : les appeler dans le chemin
décode comme ils le sont en teacher-force.

- [ ] **Step 3 : régénérer la fixture en fp32**

```bash
ssh "$ORACLE_HOST" 'cd <repo> && nohup python3 scripts/69_u8_gen_oracle.py \
  --weights <export dq> --compute-fp32 --out <fixture fp32> \
  > /tmp/rp1_fixture.log 2>&1 < /dev/null &'
```

Attendu au log : `logits float32`, 48 tokens produits, aucun assert.
L'oracle fp32 est **~5× plus rapide** que le bf16 émulé (`U_12B_RESULTS.md:41`) — si c'est plus
lent, quelque chose ne va pas.

- [ ] **Step 4 : RE-VALIDER le greedy 48/48 STRICT contre la fixture fp32**

C'est **ce run** qui refonde l'instrument, et il tourne sur le code **non modifié**.

Attendu : **48/48**. Rappel : 42/48 est la signature du quantum bf16, pas un échec du runner.
**Si ce run ne donne pas 48/48, le chantier penalty s'arrête ici** et le problème est instruit
dans l'instrument.

- [ ] **Step 5 : commit + tag**

```bash
git add scripts/69_u8_gen_oracle.py
git commit -m "gate(rp-1): oracle décode 12B refondé en fp32 — greedy 48/48 STRICT re-validé (le bf16 fabriquait des ties, U_12B_RESULTS F6)"
git tag gate/rp-1-pass
```

---

## Task 2 : le producteur de fixtures RP1 (Python, vrai processor)

**Files:**
- Create: `scripts/71_penalty_vectors.py`
- Create: `fixtures/penalty_vectors.safetensors` (committée)

- [ ] **Step 1 : écrire le producteur**

Il **appelle le vrai processor** (spec C5) — retranscrire la formule est interdit :

```python
#!/usr/bin/env python3
"""RP1 — vecteurs de référence de la repetition penalty, produits par le VRAI processor HF.
Interdiction de retranscrire le torch.where (spec C5) : une faute commune aux deux
implémentations passerait le gate."""
import json, torch
from safetensors.torch import save_file
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
import transformers

VOCAB = 512  # petit, suffisant : la formule est indépendante de la taille
PENALTIES = [0.8, 1.0, 1.15, 1.5]

def main():
    g = torch.Generator().manual_seed(20260727)
    # logits des DEUX signes, amplitude comparable au post-softcap réel (±30)
    logits = (torch.rand(VOCAB, generator=g, dtype=torch.float32) * 60.0) - 30.0
    # historique AVEC doublons (exerce la déduplication) et couvrant les deux signes
    hist = torch.tensor([7, 7, 7, 42, 100, 100, 3, 511, 0], dtype=torch.int64)

    out = {"logits_in": logits, "hist": hist.to(torch.int32)}
    meta = {"transformers_version": transformers.__version__, "penalties": PENALTIES}
    for p in PENALTIES:
        proc = RepetitionPenaltyLogitsProcessor(penalty=float(p))
        got = proc(hist.unsqueeze(0), logits.clone().unsqueeze(0)).squeeze(0)
        out[f"logits_out_{p}"] = got.contiguous()
        # métadonnées de non-vacuité : le test Zig les vérifiera aussi
        touched = (got != logits).sum().item()
        meta[f"touched_{p}"] = touched
    save_file(out, "fixtures/penalty_vectors.safetensors")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : exécuter et vérifier les propriétés de non-vacuité**

```bash
python3 scripts/71_penalty_vectors.py
```

Attendu : `touched_1.0 == 0` (penalty neutre) et `touched_0.8`, `touched_1.15`, `touched_1.5`
**égaux au nombre de tokens DISTINCTS de `hist`, soit 6** (`{7,42,100,3,511,0}`) — et non 9.
**Si l'un vaut 9, le processor ne déduplique pas et toute la spec est à revoir.**
Noter la version de transformers affichée (spec §4-5).

- [ ] **Step 3 : commit**

```bash
git add scripts/71_penalty_vectors.py fixtures/penalty_vectors.safetensors
git commit -m "test(rp1): vecteurs de référence penalty produits par le VRAI RepetitionPenaltyLogitsProcessor (dédup vérifiée : 6 tokens distincts sur 9 entrées)"
```

---

## Task 3 : `sampling.zig` + selftest (RP1) — TDD

**Files:**
- Create: `zml_runner/sampling.zig`
- Modify: `zml_runner/gemma4_g12auto.zig` (drapeau `--selftest-penalty` + dispatch)
- Modify: `zml_runner/BUILD.bazel` (ajouter `sampling.zig` aux `srcs` de `gemma4_g12auto`)

- [ ] **Step 1 : écrire le selftest qui échoue (le test AVANT le module)**

Dans `gemma4_g12auto.zig`, sur le modèle de `selftestInputs` (`:449`) et de son dispatch
(« Task 3 : `--selftest-inputs` », early-return avant toute init GPU) :

```zig
// RP1 : vérifie sampling.applyRepetitionPenalty contre les vecteurs du VRAI processor HF.
// Indépendant du prompt/tokenizer/poids/GPU (fixture only), comme --selftest-inputs.
fn selftestPenalty(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !void {
    // lecture via readFixtureAlloc, comme --oracle / --selftest-inputs
    const logits_in = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "logits_in");
    const hist_i32  = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "hist");
    // ... pour chaque penalty : appliquer et comparer BIT À BIT
}
```

Critères que le selftest doit imposer (spec RP1) :
1. **0 ULP** — comparaison bit à bit (`@bitCast` des f32 puis égalité entière), aucun epsilon ;
2. **assertions sur la fixture elle-même**, échec si un compteur vaut 0 :
   au moins 1 token en doublon dans `hist`, au moins 1 logit négatif pénalisé, au moins 1
   positif pénalisé ;
3. **tie-break** : un vecteur à ties f32 exacts, `argmax` doit rendre le **premier** indice.

- [ ] **Step 2 : builder et vérifier que ça NE COMPILE PAS**

```bash
ssh "$ZML_REMOTE" 'cd "$ZML_DST/../.." && ./bazel.sh build //examples/rqz:gemma4_g12auto'
```

Attendu : **échec de compilation**, `sampling.zig` introuvable / `applyRepetitionPenalty`
non définie. C'est le rouge du TDD.

- [ ] **Step 3 : écrire le module minimal**

```zig
//! Transforme un vecteur de logits en un token. Aucune dépendance ZML (f32 nus) : ce fichier
//! est exerçable sans GPU, sans PJRT, sans poids (cf --selftest-penalty).
//!
//! Formule : HF RepetitionPenaltyLogitsProcessor — logit négatif ×penalty, positif ou nul
//! ÷penalty, AU PLUS UNE FOIS par token distinct (token_mask booléen côté HF).
const std = @import("std");

pub const Params = struct {
    repetition_penalty: f32 = 1.0,
    ignore_prompt: bool = false,
};

/// Bitset minimal — pas de dépendance à une API std qui bouge entre versions.
pub const Bitset = struct {
    words: []u64,
    pub fn init(allocator: std.mem.Allocator, n: usize) !Bitset {
        const w = try allocator.alloc(u64, (n + 63) / 64);
        @memset(w, 0);
        return .{ .words = w };
    }
    pub fn deinit(self: *Bitset, allocator: std.mem.Allocator) void {
        allocator.free(self.words);
    }
    pub fn clear(self: *Bitset) void {
        @memset(self.words, 0);
    }
    /// true si le bit était DÉJÀ posé (donc : token déjà pénalisé).
    pub fn testAndSet(self: *Bitset, i: usize) bool {
        const w = i >> 6;
        const b: u6 = @truncate(i);
        const mask = @as(u64, 1) << b;
        const was = (self.words[w] & mask) != 0;
        self.words[w] |= mask;
        return was;
    }
};

/// Applique la penalty IN-PLACE. `seen` doit être remis à zéro par l'appelant (clear()).
/// ⚠ La division reste une DIVISION : `× (1/penalty)` est mathématiquement équivalent mais
/// casse la bit-exactitude f32 exigée par RP1.
pub fn applyRepetitionPenalty(logits: []f32, hist: []const u32, penalty: f32, seen: *Bitset) void {
    if (penalty == 1.0) return;
    for (hist) |t| {
        const i: usize = @intCast(t);
        if (seen.testAndSet(i)) continue; // au plus une fois par token distinct
        const v = logits[i];
        logits[i] = if (v < 0) v * penalty else v / penalty;
    }
}

/// argmax. Tie-break : le PREMIER indice gagne (comparaison stricte `>`).
pub fn argmax(logits: []const f32) u32 {
    var best: usize = 0;
    for (logits[1..], 1..) |v, i| {
        if (v > logits[best]) best = i;
    }
    return @intCast(best);
}
```

- [ ] **Step 4 : builder et lancer le selftest**

```bash
ssh "$ZML_REMOTE" 'cd "$ZML_DST/../.." && ./bazel.sh build //examples/rqz:gemma4_g12auto \
  && ./bazel-bin/examples/rqz/gemma4_g12auto --selftest-penalty <fixture>'
```

Attendu : `RP1 PASS` — 4 penalties × 512 valeurs bit-identiques, compteurs de fixture non nuls,
tie-break conforme. **Aucun GPU n'est touché** (early-return avant l'init platform).

- [ ] **Step 5 : commit + tag**

```bash
git add zml_runner/sampling.zig zml_runner/gemma4_g12auto.zig zml_runner/BUILD.bazel
git commit -m "gate(rp1): sampling.zig — penalty bit-exacte 0 ULP vs le vrai processor HF, dédup par bitset, tie-break premier indice ; selftest sans GPU (amendement 1)"
git tag gate/rp1-pass
```

---

## Task 4 : câblage dans la boucle + RP0 + RP2

**Files:**
- Modify: `zml_runner/gemma4_g12auto.zig` — parse CLI, `generateOnce` (`:1296`), boucle (`:1340-1420`)

- [ ] **Step 1 : parse CLI avec la garde en ACCEPTATION**

```zig
} else if (std.mem.eql(u8, a, "--repetition-penalty")) {
    i += 1;
    if (i >= process_args.len) { log.err("--repetition-penalty attend une valeur", .{}); return error.MissingValue; }
    const p = std.fmt.parseFloat(f32, process_args[i]) catch {
        log.err("--repetition-penalty : valeur non numérique", .{}); return error.InvalidPenalty;
    };
    // ⚠ FORMULATION EN ACCEPTATION : `p <= 0` laisserait passer NaN (NaN <= 0 est faux),
    // qui empoisonnerait tous les logits et rendrait un token arbitraire SANS erreur.
    if (!(p > 0 and std.math.isFinite(p))) {
        log.err("--repetition-penalty doit être fini et > 0 (reçu {d})", .{p}); return error.InvalidPenalty;
    }
    args.repetition_penalty = p;
}
```

- [ ] **Step 2 : allouer bitset et buffer de logits UNE FOIS**

Dans `generateOnce`, avant la boucle de steps — **pas par step** : un `toSliceAlloc` de 1 Mio
par step ferait 4 Gio d'allocations sur 20 prompts × 200 tokens et casserait le critère RSS de
RP5 de bonne foi. Dimensionner sur le vocab **runtime** (`:1239`), jamais sur 262144 en dur.

- [ ] **Step 3 : brancher le chemin host dans la boucle**

Après `call_results.get`, remplacer la lecture inconditionnelle du top-5 par :

```zig
var tok: i64 = @intCast(top5.idx[0]);
if (in_gen_phase and params.repetition_penalty != 1.0) {
    // r_logits : D2H de {voc} f32 dans le buffer préalloué (transfert DÉJÀ synchrone).
    try readLogitsInto(&r_logits, logits_host, io);
    seen.clear();
    sampling.applyRepetitionPenalty(logits_host, hist.items, params.repetition_penalty, &seen);
    const host_tok = sampling.argmax(logits_host);
    if (host_tok != top5.idx[0]) {
        try divergences.append(allocator, .{ .step = step, .margin = top5.val[0] - top5.val[1] });
    }
    tok = @intCast(host_tok);
}
// hist : en PREFILL on mémorise `fed` (token du prompt), PAS `tok` (argmax ignoré, cf :1415).
try hist.append(allocator, if (in_gen_phase) @intCast(tok) else @intCast(fed));
```

- [ ] **Step 4 : en fin de génération, appliquer les règles du compteur**

```zig
if (params.repetition_penalty != 1.0) {
    if (divergences.items.len == 0) {
        log.err("penalty={d} active mais AUCUNE divergence host/device : la penalty n'a rien fait (paramètre non propagé ? hist vide ? in_gen_phase inversé ?)", .{params.repetition_penalty});
        return error.PenaltyInert;   // FAIL bruyant, spec §3.2.1
    }
    for (divergences.items) |d| if (d.margin == 0.0) {
        log.err("divergence à marge EXACTEMENT 0.0 au step {d} : désaccord de tie-break host/device, sans rapport avec la penalty", .{d.step});
        return error.TieBreakMismatch;  // le seul endroit où un vrai tie est observable
    };
    log.info("divergences : {d} step(s), marge min {d:.6}", .{ divergences.items.len, minMargin(divergences.items) });
}
```

- [ ] **Step 5 : RP0 — prouver que le graphe n'a pas bougé**

```bash
# après deploy + build de l'état MODIFIÉ
ssh "$ZML_REMOTE" 'XLA_FLAGS=--xla_dump_to=/tmp/hlo_after nohup ./bazel-bin/examples/rqz/gemma4_g12auto <mêmes args que Task 0> > /tmp/witness_after.log 2>&1 < /dev/null &'
ssh "$ZML_REMOTE" 'diff -rq /tmp/hlo_before /tmp/hlo_after | tee /tmp/rp0_diff.txt; wc -l < /tmp/rp0_diff.txt'
```

Attendu : **uniquement les tolérances de F5** — `debug_options` (le chemin de dump lui-même) et
éventuellement un `.ir-with-opt.ll` à noms SSA alpha-équivalents. **Publier le nombre observé** :
il tranche la contradiction interne du repo (`ENGINE_LOG.md:92` en compte 2,
`ZML_MODULAR_ENGINE_DESIGN.md:132` en compte 1).

- [ ] **Step 6 : RP0 — contre-test de non-vacuité**

Dans un **worktree jetable** (jamais committé, spec §9) : perturber `RMS_EPS` 1e-6 → 1e-2,
rebuilder, re-dumper. **RP0 doit FAIL.** S'il passe, RP0 ne prouve rien et c'est ça le finding.

- [ ] **Step 7 : RP2 — non-régression à penalty=1.0**

Rejouer 48 tokens greedy sans `--repetition-penalty`, comparer au **témoin de la Task 0 Step 4**
(pas à `u8_gen48`). Attendu : **bit-identique**.

- [ ] **Step 8 : commit + tags**

```bash
git add zml_runner/gemma4_g12auto.zig
git commit -m "gate(rp0,rp2): penalty câblée host-side — HLO inchangé (tolérances F5, N diffs publiés), contre-test RMS_EPS FAIL comme attendu, greedy bit-identique au témoin pré-édition ; garde NaN en acceptation, compteur de divergences (FAIL si inerte, FAIL si marge 0.0)"
git tag gate/rp0-pass && git tag gate/rp2-pass
```

---

## Task 5 : oracle 69 pénalisé + RP3 + RP4

**Files:**
- Modify: `scripts/69_u8_gen_oracle.py` (processor, `s0`, `prompt_ids`)

- [ ] **Step 1 : brancher le VRAI processor dans la boucle de décode**

```python
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
# ...
proc = RepetitionPenaltyLogitsProcessor(penalty=args.repetition_penalty) if args.repetition_penalty != 1.0 else None
```

Appliqué **avant** l'argmax, sur `out.logits[0, -1, :]`, avec l'historique = prompt ++ généré
(défaut HF). **Interdiction de retranscrire le `torch.where`** (spec C5).

- [ ] **Step 2 : appliquer la penalty à `s0` aussi**

`s0` est produit par le prefill **hors boucle** (`:143-153`, `seq = [int(idxs[0])]`, la boucle
démarre `:160`). Si la penalty n'y est pas appliquée, divergence garantie dès le token 0 —
c'est un off-by-one certain, pas une hypothèse.

- [ ] **Step 3 : exporter `prompt_ids` en TENSEUR**

`save_file` (`:191`) ne l'exporte pas, il ne vit que dans le manifest JSON (`:204`). Sous
penalty les ids du prompt entrent dans le calcul, or le contrôle côté runner ne porte que sur
**la longueur** (`gemma4_g12auto.zig:977-981`). Ajouter la clé et **comparer littéralement** les
ids du template Zig aux `prompt_ids` de la fixture avant de conclure quoi que ce soit de RP3.

- [ ] **Step 4 : pré-calculer le mordant SANS GPU**

```bash
# deux runs d'oracle seulement — aucune ressource GPU engagée
python3 scripts/69_u8_gen_oracle.py --compute-fp32 --repetition-penalty 1.0  --out /tmp/o_greedy
python3 scripts/69_u8_gen_oracle.py --compute-fp32 --repetition-penalty 1.15 --out /tmp/o_p115
python3 scripts/69_u8_gen_oracle.py --compute-fp32 --repetition-penalty 0.8  --out /tmp/o_p08
# hamming des ids
```

Attendu : `hamming ≥ 3` sur 48 pour **chacune** des deux penalties. **Publier le hamming réel**
(attendu autour de ~40 par effet de cascade : un changement de décision entraîne tout le
suffixe). **Vérifier explicitement le cas `0.8`** : elle *récompense* la répétition, c'est le
cas le moins évident. Si le mordant est sous 3, la configuration est **déclarée inutilisable** et
le prompt change — le gate ne peut pas passer à vide.

- [ ] **Step 5 : RP3 — le gate**

Lancer le runner avec `--oracle <fixture pénalisée>` pour penalty ∈ {0.8, 1.15}.
Attendu : **ids == HF**, `divergences.len > 0`, marge min publiée.
En cas de mismatch : appliquer §7-3 (les **trois** conditions, ε = **2e-3**). Rappel : la marge
minimale jamais observée est **0,0279 sur 1150 steps**, donc la fenêtre d'admissibilité est
**attendue vide** — un mismatch dans la fenêtre serait un événement à instruire, pas une routine.

- [ ] **Step 6 : RP4 — les trois corruptions**

| Corruption | Où | Attendu |
|---|---|---|
| (a) signes inversés | `sampling.zig`, échanger les branches `v < 0` | RP3 **FAIL** |
| (b) dédup supprimée | retirer le `if (seen.testAndSet(i)) continue;` | RP3 **FAIL** |
| (c) périmètre du prompt | forcer `ignore_prompt` à l'inverse | RP3 **FAIL** |

**Publier le mordant de chaque corruption**, plancher 1. Un mordant
`< mordant_naturel(RP3) / 2` est **à instruire, pas à valider** : vu la cascade, une vraie
corruption sémantique mord massivement. Précédent à garder en tête : `ENGINE_LOG.md:277`,
« les contre-tests argmax restaient à 0 divergence — l'argmax greedy est trop robuste ».

- [ ] **Step 7 : commit + tags** (`gate/rp3-pass`, `gate/rp4-pass`)

---

## Task 6 : directives repl + RP5 + RP6

**Files:**
- Modify: `zml_runner/gemma4_g12auto.zig` — boucle stdin du mode `--repl`

> ⚠ Les deux pièges `std.Io` 0.16 du chantier repl s'appliquent ici : `takeDelimiter` (et non
> `Exclusive`, qui laisse le `\n` en tête et provoque une sortie prématurée **exit 0** — un faux
> succès démasqué seulement par le **comptage** des générations) ; et **un seul writer** sur
> stdout (deux writers sur le même fd entrelacent leurs octets).

- [ ] **Step 1 : parser les directives avant le rendu du prompt**

Une ligne commençant par `:` n'est **jamais** un prompt. `:penalty <f>` (même garde en
acceptation qu'en CLI), `:ignore-prompt on|off`, `:params`, `:help`. Valeur invalide → message
d'erreur, **la session continue**.

- [ ] **Step 2 : RP5 — aucun état ne survit entre prompts**

`--repl` étant exclusif de `--oracle`/`--out-ids` (`:868-874`, garde **non relâchée**), la
comparaison porte sur le **texte détokenisé de stdout**.

Même prompt joué 2×, penalty active, `max_tokens = 32`. Attendu : **texte identique**, **et
`divergences.len > 0` sur les DEUX passes** — sans quoi deux runs greedy identiques passeraient
le gate à vide. Puis 20 prompts : **RSS ≤ +1 Mo** (repère R2), atteignable parce que le buffer
de logits est alloué une fois. 20 × 32 tokens ≈ 2 min de génération.

- [ ] **Step 3 : RP6 — les quatre sous-critères**

(a) `:penalty 1.15` ne produit **aucune** génération → vérifié par **comptage** des générations,
pas par le code de sortie. (b) la valeur s'applique au prompt **suivant**, sur un prompt dont la
sensibilité est **prouvée par RP3**, avec `divergences.len > 0`. (c) `:params` vérifié **contre
le comportement** : `:params` annonce 1.15 ⇒ **les 48 premiers ids** égalent la référence 1.15
(bornage explicite : le repl s'arrête sur EOT/`max_tokens`, l'oracle sur `fed.len`). (d) valeurs
invalides **énumérées** : `0`, `-1`, `nan`, `inf`, `abc`, vide — chacune rejetée sans tuer la
session. **`nan` est le cas qui compte** : c'est celui que la garde naïve laissait passer.

- [ ] **Step 4 : commit + tags** (`gate/rp5-pass`, `gate/rp6-pass`)

---

## Task 7 : RP7 (récitation) + M1 (coût)

- [ ] **Step 1 : implémenter la métrique de récitation**

Longueur maximale de n-gramme répété sur les 200 derniers tokens. Petit utilitaire host,
appliqué aux ids générés.

- [ ] **Step 2 : établir la ligne de base SAINE**

Mesurer la métrique sur le run U9 « 1150 tok stables, texte cohérent »
(`U_12B_RESULTS.md:64`) — c'est la référence d'un texte sain **du même modèle**, et non un ratio
choisi à la main.

- [ ] **Step 3 : publier le témoin AVANT le balayage**

`penalty = 1.0`, `max_tokens = 200`. **Si le témoin ne récite pas, RP7 est déclaré vacué** et le
prompt change — sinon « récitation levée » serait vrai à vide.

- [ ] **Step 4 : balayage à chaud**

Valeurs `{1.05, 1.1, 1.15, 1.3}` via `:penalty`, **une seule compile**. PASS = métrique ≤ ligne
de base saine pour au moins une valeur, avec `divergences.len > 0`.
⚠ RP7 **ne mesure pas la qualité** : une penalty haute peut casser la boucle en produisant du
charabia. **Joindre la sortie de chaque valeur au rapport** pour lecture humaine.

- [ ] **Step 5 : M1 — chronomètre host, bras appariés**

Instrumenter le bloc hôte (µs/step : D2H + penalty + argmax) et faire **deux bras entrelacés
dans une seule session `--repl`** — ce qui élimine la variance inter-run. Attendu **~0,5 %** de
~110 ms/step. Le tok/s bout-en-bout n'est qu'un recoupement grossier.
*M1 est une mesure, pas un gate : le plafond 10 % est un tripwire, non discriminant.*

- [ ] **Step 6 : commit + tag** (`gate/rp7-pass`)

---

## Task 8 : documentation et PR

- [ ] **Step 1 : `docs/SAMPLING_RESULTS.md`**

Doit contenir **les chiffres, pas des adjectifs** : nombre de diffs HLO observés en RP0 (et la
contradiction du repo tranchée), hamming réel de RP3 pour 1.15 **et** 0.8, mordant de chacune
des 3 corruptions de RP4, marge min avec et sans penalty, ligne de base et sorties du balayage
RP7, µs/step de M1, version de `transformers`, dtype de la fixture RP-1.

- [ ] **Step 2 : porter l'amendement 1 dans la spec** (RP1 par selftest, pas `zig test`)

- [ ] **Step 3 : mettre à jour `PLANNING.md`** et retirer l'entrée sampling du backlog

- [ ] **Step 4 : grep d'anonymisation avant push** (repo public)

```bash
grep -rn -E "/Users/|([0-9]{1,3}\.){3}[0-9]{1,3}|[a-z0-9._-]+@[a-z0-9.-]+|\bmacmini\b|\bregis\b" \
  docs/SAMPLING_RESULTS.md docs/superpowers/plans/2026-07-27-*.md zml_runner/sampling.zig scripts/71_*.py
```

Attendu : **aucune occurrence**. Vérifier aussi que le motif sait détecter (le passer sur une
ligne factice contenant une IP) — un contrôle qui ne peut pas réussir ne prouve rien.

- [ ] **Step 5 : PR**

Titre : `feat(sampling): repetition penalty host-side sur le 12B — ids == HF, graphe intact`.
Corps : RP-1 (l'oracle refondé en fp32) en premier, puis les gates dans l'ordre, avec les
chiffres. Mentionner la dette écrite (câblage E2B) et la phase 2 (SM0-SM3) restée au backlog.

---

## Ordre d'exécution et parallélisme

```
Task 0 (témoins)  ──→ Task 4 (câblage, RP0/RP2) ──→ Task 5 (RP3/RP4) ──→ Task 6 ──→ Task 7 ──→ Task 8
Task 1 (RP-1)     ──────────────────────────────────↗
Task 2 (fixtures) ──→ Task 3 (module, RP1)  ────────↗
```

**Task 2 et Task 3 ne demandent aucun GPU** et peuvent démarrer immédiatement, en parallèle de
Task 1. **Task 5 est bloquée par Task 1** (RP-1) : c'est la contrainte dure du plan.
