# Runtime autonome — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un binaire ZML texte→texte (`gemma4_gen_auto`) : prompt en CLI → réponse Gemma-4-E2B
sur stdout, sans Python ni fixture par prompt — validé == HF par les gates A0-A3 de la spec.

**Architecture:** Spec approuvée `docs/GEN_AUTONOME_DESIGN.md`. Le moteur `engine.zig` ne change
pas d'un octet : on compile l'entrée **`forwardStep`** (engine.zig:632, embeds token-dépendants)
et tout le neuf est host-side — tokenizer ZML natif, chat template Zig, gather embeds **bruts**
(scalings in-graph), fabrication host du `Packed` (cos/sin full, masques, positions) et du cache
à zéros, boucle prefill-par-decode → argmax → early-stop EOS.

**Tech Stack:** Zig 0.16-dev + ZML (workspace 3090 `/data/rqz_workspace/zml`), Bazel (`./bazel.sh`),
oracles Python HF (scripts 46/49) côté validation uniquement.

**Contexte d'exécution (tous les runs GPU) :**
- Éditer sur M1 (`~/dev/gemma4-zml-probe/zml_runner/`), déployer : `zml_runner/deploy_to_3090.sh`.
- Builder/runner sur la 3090 : `ssh user@gpu-host` puis
  `cd /data/rqz_workspace/zml && ./bazel.sh run //examples/rqz:<cible> -- <args>`.
- Checkpoint : `/data/gemma4-zml-probe/weights/model.safetensors`. Venv oracle :
  `/data/venvs/gemma4-probe` (`source /data/venvs/gemma4-probe/bin/activate`).
- Pièges connus : nom de cible COURT (quota comptime pjrt) ; patch local
  `@setEvalBranchQuota(100_000)` dans pjrt.zig à réappliquer si workspace resync ;
  vérifier `nvidia-smi --query-compute-apps` avant tout run (contention).
- Avant chaque commit : `git add` UNIQUEMENT les fichiers du step (le repo peut porter
  d'autres travaux).

---

## Fichiers

| Fichier | Rôle | Action |
|---|---|---|
| `zml_runner/gemma4_gen_auto.zig` | LE runner autonome : CLI, tokenizer, template, inputs host, gather, boucle, gates | Créer |
| `zml_runner/BUILD.bazel` | cible `gemma4_gen_auto` | Modifier |
| `docs/GEN_AUTONOME_DESIGN.md` | statut des gates au fil de l'eau | Modifier |
| `docs/DOCUMENTATION.md` | courte section usage CLI (Task 9) | Modifier |
| `PLANNING.md` | solder l'item backlog à la fin | Modifier |

Un seul fichier Zig neuf : le runner est un assemblage de patterns existants
(`gemma4_gen_long_gpu.zig` = squelette compile/boucle GPU ; `gemma4_gchunk_auto.zig:137-296`
= gather streaming + argmax + reinject). `engine.zig` N'EST PAS modifié.

---

### Task 0 : Spike tokenizer ZML (prérequis bloquant, spec §3.1)

**Files:** aucun (spike sans code committé)

- [ ] **Step 0.1 : Localiser tokenizer.json sur la 3090**

```bash
ssh user@gpu-host 'ls /data/hf_cache/hub/models--google--gemma-4-E2B-it/snapshots/*/tokenizer.json'
```
Expected: un chemin existant (noté `$TOKJSON` ci-dessous). S'il est absent :
`ls /data/hf_cache/hub/models--google--gemma-4-E2B-it/snapshots/*/` et chercher
`tokenizer.model`/`tokenizer.json` ; en dernier recours le télécharger via le venv
(`AutoTokenizer.from_pretrained` sans offline). Consigner le chemin.

- [ ] **Step 0.2 : Le module tokenizer ZML parse-t-il le tokenizer Gemma ?**

```bash
ssh user@gpu-host 'cd /data/rqz_workspace/zml && ./bazel.sh run //zml/tokenizer -- --tokenizer=$TOKJSON --prompt="What is the capital of France? Answer in one word."'
```
Expected: log `✅ Loaded tokenizer` + une liste d'ids. **FAIL = STOP** : remonter à Régis
(le design §3.1 fait de ce parse un prérequis ; alternative à discuter, pas à improviser).

- [ ] **Step 0.3 : Comparer à l'encodage HF du même texte brut (sans template)**

```bash
ssh user@gpu-host 'source /data/venvs/gemma4-probe/bin/activate && HF_HOME=/data/hf_cache python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(\"google/gemma-4-E2B-it\")
print(t(\"What is the capital of France? Answer in one word.\", add_special_tokens=False).input_ids)
print(\"bos:\", t.bos_token_id, \"eot:\", t.convert_tokens_to_ids(\"<end_of_turn>\"))"'
```
Expected: les ids HF (sans tokens spéciaux) == les ids du step 0.2 (modulo BOS auto —
noter la différence exacte, elle sert en Task 2). Noter `bos_token_id` et l'id de
`<end_of_turn>` (utilisés partout ensuite). Divergence de fond (pas juste BOS) = STOP,
remonter à Régis.

---

### Task 1 : Oracles et fixtures de référence (3090)

**Files:** aucun (artefacts /data, non versionnés — convention fixtures gitignorées)

- [ ] **Step 1.1 : Dumper le rendu TEXTE du chat template (source de vérité pour le Zig)**

```bash
ssh user@gpu-host 'source /data/venvs/gemma4-probe/bin/activate && HF_HOME=/data/hf_cache python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(\"google/gemma-4-E2B-it\")
s = t.apply_chat_template([{\"role\":\"user\",\"content\":\"PROMPT_ICI\"}], add_generation_prompt=True, tokenize=False)
print(repr(s))"'
```
Expected: la chaîne exacte (attendu de la famille Gemma :
`'<bos><start_of_turn>user\nPROMPT_ICI<end_of_turn>\n<start_of_turn>model\n'` — **prendre
le repr() comme vérité**, ne rien supposer). La consigner en commentaire du runner (Task 2).

- [ ] **Step 1.2 : Fixture A1/A3 (courte, 48 tokens) — vérifier ou régénérer**

```bash
ssh user@gpu-host 'ls -la /data/gemma4-zml-probe/gen_custom.safetensors*'
# si absente :
ssh user@gpu-host 'source /data/venvs/gemma4-probe/bin/activate && cd /data/gemma4-zml-probe && HF_HOME=/data/hf_cache python3 scripts/49_gen_custom_oracle.py --prompt "What is the capital of France? Answer in one word." --n-tokens 48 --out /data/gemma4-zml-probe/gen_custom.safetensors'
```
Expected: fixture + `gen_custom.safetensors.manifest.json` (contient `prompt_ids`, `seq_len`,
`expected_head`). Noter `seq_len` (S_REF) et `prompt_ids`. Si la fixture existe mais que le
manifest montre un AUTRE prompt ou `n_decode` ≠ 48 : régénérer (commande ci-dessus).

- [ ] **Step 1.3 : Localiser l'EOS dans expected (critère A3)**

```bash
ssh user@gpu-host 'source /data/venvs/gemma4-probe/bin/activate && python3 -c "
from safetensors import safe_open
f = safe_open(\"/data/gemma4-zml-probe/gen_custom.safetensors\", \"pt\")
exp = f.get_tensor(\"expected\").tolist()
EOT = <id_end_of_turn_du_step_0.3>
print(\"expected[:12] =\", exp[:12])
print(\"premier EOT à l index\", exp.index(EOT) if EOT in exp else \"ABSENT\")"'
```
Expected: un index petit (réponse « Paris » + fin de tour). **Si ABSENT** : générer une
fixture A3 dédiée avec un prompt encore plus fermé (ex. `--prompt "Say only the word: yes" --n-tokens 32`)
et noter son index EOT. A3 a besoin d'UNE fixture dont `expected` contient l'EOT.

- [ ] **Step 1.4 : Fixture A2 (longue, jusqu'à L_MAX)**

```bash
# 1er run pour lire seq_len (l'assert du script borne) :
ssh user@gpu-host '... python3 scripts/49_gen_custom_oracle.py --prompt "Tell me the story of the number zero, from its invention to modern mathematics." --n-tokens 1000 --out /data/gemma4-zml-probe/gen_auto_long.safetensors'
# lire "→ N tokens" dans la sortie ; si seq_len=S, relancer avec --n-tokens $((1024-S)) pour le max exact
```
Expected: fixture longue, `n_decode ≥ 1000` (critère A2 = N/N sur CE N). ~10 min GPU
(génération HF token par token). Vérifier `nvidia-smi` avant (pas de contention).

---

### Task 2 : Runner squelette — CLI + tokenizer + chat template → **gate A0**

**Files:**
- Create: `zml_runner/gemma4_gen_auto.zig`
- Modify: `zml_runner/BUILD.bazel`

- [ ] **Step 2.1 : Cible Bazel**

Dans `zml_runner/BUILD.bazel`, après la cible `gemma4_gchunk_auto`, dupliquer son bloc en
l'adaptant (mêmes deps — le tokenizer est exposé via le module `zml` : cf
`examples/llm/main.zig:168` qui utilise `zml.tokenizer.Tokenizer` avec les deps standard) :

```python
zig_binary(
    name = "gemma4_gen_auto",
    srcs = ["engine.zig", "mem_probe.zig"],
    main = "gemma4_gen_auto.zig",
    deps = ["//zml"],   # reprendre EXACTEMENT les deps de gemma4_gchunk_auto
)
```
NB : recopier le bloc réel de `gemma4_gchunk_auto` (attrs exacts du repo), ne pas inventer.
Si `zml.tokenizer` n'est pas ré-exporté par `//zml`, ajouter la dep `//zml/tokenizer` et
importer `@import("zml/tokenizer")` (cf `zml/tokenizer/main.zig:4`).

- [ ] **Step 2.2 : Squelette du runner — CLI, tokenizer, template, mode `--ids-only`**

`gemma4_gen_auto.zig` (première tranche — pas encore de moteur) :

```zig
// Runtime AUTONOME texte→texte (spec docs/GEN_AUTONOME_DESIGN.md).
// Gates : A0 tokenizer+template ; A1 prefill-par-decode 48/48 ; A2 long N/N ; A3 early-stop EOS.
// Le moteur engine.zig est INTACT — entrée compilée : forwardStep (embeds host token-dépendants).
const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const SLIDING_WINDOW: i64 = 512;
const HD_F: i64 = 512; // dim cos/sin full
const D: i64 = 1536;
const LF: i64 = 8960;
const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
const PackedLong = engine.Packed(true);

// Chat template Gemma — VÉRITÉ = repr() du step 1.1 (mesuré 10 juil) :
//   '<bos><|turn>user\nPROMPT<turn|>\n<|turn>model\n'
// ⚠ Les tokens de tour sont <|turn>/(id 105) et <turn|>/(id 106) — PAS <start_of_turn>/<end_of_turn>.
// BOS (id 2) : PRÉFIXÉ en id (l'encoder ZML n'ajoute AUCUN token spécial, Task 0) — le rendu
// texte ci-dessous commence donc APRÈS <bos>. Ids de réf complets (prompt capital of France) :
//   [2, 105, 2364, 107, …, 106, 107, 105, 4368, 107]
fn renderChatTemplate(allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "<|turn>user\n{s}<turn|>\n<|turn>model\n", .{prompt});
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000); // piège quota comptime (cf gemma4_gchunk_auto.zig:96)
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena.allocator());
    // usage : gemma4_gen_auto <model.safetensors> <tokenizer.json> --prompt "..."
    //         [--max-tokens N] [--oracle fixture.safetensors] [--ids-only] [--selftest-inputs f] [--selftest-gather f]
    // (parse à la main comme les runners existants ; --oracle active la comparaison expected)
    ...
    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, tokjson_path);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    const rendered = try renderChatTemplate(arena.allocator(), prompt);
    var prompt_tok = try encoder.encodeAlloc(allocator, rendered);
    defer prompt_tok.deinit(allocator);
    // BOS : ajuster selon le constat du step 0.3 (préfixer bos_id si l'encoder ne l'émet pas).
    if (ids_only) {
        log.info("ids = {any}", .{prompt_tok.items});
        return;
    }
    ...
}
```

- [ ] **Step 2.3 : Déployer + builder + run `--ids-only`**

```bash
~/dev/gemma4-zml-probe/zml_runner/deploy_to_3090.sh
ssh user@gpu-host 'cd /data/rqz_workspace/zml && ./bazel.sh run //examples/rqz:gemma4_gen_auto -- /data/gemma4-zml-probe/weights/model.safetensors $TOKJSON --prompt "What is the capital of France? Answer in one word." --ids-only'
```
Expected: liste d'ids. Comparer à `prompt_ids` du manifest de la fixture A1 (step 1.2) :

```bash
ssh user@gpu-host 'python3 -c "import json; print(json.load(open(\"/data/gemma4-zml-probe/gen_custom.safetensors.manifest.json\"))[\"prompt_ids\"])"'
```
**Itérer sur BOS/template jusqu'à égalité EXACTE** (c'est attendu : 1-2 ajustements).
Puis 2e prompt de contrôle (celui de la fixture A2, step 1.4) : égalité aussi.

- [ ] **Step 2.4 : Round-trip détok**

Ajouter au mode `--ids-only` : `decoder.decode(ids du prompt hors préfixe template)` puis
re-encode == ids (log PASS/FAIL). Run : PASS.

- [ ] **Step 2.5 : Gate A0 — commit + tag**

```bash
cd ~/dev/gemma4-zml-probe && git add zml_runner/gemma4_gen_auto.zig zml_runner/BUILD.bazel
git commit -m "feat(gen-auto): A0 — tokenizer ZML + chat template Zig, ids == HF sur 2 prompts (round-trip détok OK)"
git tag gate/gen-auto-a0-pass
```

---

### Task 3 : Inputs host — cos/sin full, masques, positions, cache zéros (validés vs fixture)

**Files:** Modify: `zml_runner/gemma4_gen_auto.zig`

- [ ] **Step 3.1 : Lire la formule EXACTE du rotary full dans la source HF (3090)**

```bash
ssh user@gpu-host 'grep -n -B2 -A40 "class Gemma4TextRotaryEmbedding" /data/venvs/gemma4-probe/lib/python3*/site-packages/transformers/models/gemma4/modeling_gemma4.py'
```
Consigner : `rope_theta` full (attendu 1e6), `partial_rotary_factor` (attendu 0.25),
`rope_scaling` "proportional" (le facteur exact), la construction d'`inv_freq`, l'ordre
cos/sin (duplication des moitiés). **Ne PAS coder de mémoire — copier la formule lue.**

- [ ] **Step 3.2 : Implémenter `ropeFull`, `maskRows`, positions, cache zéros en Zig**

```zig
// cos/sin full pour la position p — formule COPIÉE du step 3.1 (partial 0.25, proportional, theta 1e6).
fn ropeFull(p: i64, cos_out: *[HD_F]f32, sin_out: *[HD_F]f32) void { ... }

// Masques additifs f32 : 0 = visible, -floatMax = masqué (== torch.finfo(float32).min).
const MASK_MIN: f32 = -std.math.floatMax(f32);
fn maskRows(p: i64, sliding_out: []f32, full_out: []f32) void {
    const lo = @max(0, p - (SLIDING_WINDOW - 1));
    for (0..@intCast(L_MAX)) |j| {
        const ji: i64 = @intCast(j);
        sliding_out[j] = if (ji > p or ji < lo) MASK_MIN else 0;
        full_out[j] = if (ji > p) MASK_MIN else 0;
    }
}
```
Fabrication des arrays host complets (steps 0..L_MAX-1) : `cos_full/sin_full {L_MAX,1,1,512} f32`,
`masks_sliding/masks_full {L_MAX,1,1,1,L_MAX} f32`, `positions {L_MAX} i32 = 0..L_MAX-1`,
embeds/embptls **factices zéros** `{L_MAX,1,1,D}/{L_MAX,1,1,LF}` bf16 (déclarés par `Packed`
mais non consommés par `forwardStep`), cache zéros f32 (shapes de `engine.Cache` :
sl `{12,1,1,L_MAX,256}` ×2, fl `{3,1,1,L_MAX,512}` ×2).

- [ ] **Step 3.3 : Selftest vs fixture — mode `--selftest-inputs <fixture>`**

Charge la fixture A1 (49) et compare, pour chacun de ses `n_decode` steps k (position
`p = seq_len + k` lue de `positions[k]`) : cos/sin host vs fixture (tolérance ci-dessous),
masques host vs fixture (**égalité bit-exacte**, valeurs ∈ {0, MASK_MIN}), positions.

> **Tolérance cos/sin — réalité mesurée (10 juil), pas le `≤ 1e-6` initialement visé** : le
> `pow()` f32 de Zig (libm) et celui de PyTorch arrondissent `theta**exp` à 1 ULP l'un de
> l'autre sur certains `inv_freq[i]` (aucun des deux n'est « le bon », vérifié en précision
> arbitraire). L'erreur d'angle résultante croît **linéairement en p** (Δangle ≈ 2 ULP × p ×
> ~6e-8 ≈ 1.2e-7×p), propagée par sin/cos à pente ≤ 1 : plancher 2^-18 = 3.8e-6 aux positions
> courtes (p ≤ 68, fixture 48) ; 6.0e-5 mesuré à p = 612 (fixture longue ; borne 2-ULP 7.3e-5,
> cohérente). Le selftest applique donc `tol(p) = 1e-5 + 1.5e-7×p` PAR STEP (dérivation
> complète en commentaire de `cosSinTol` dans le runner) et les DEUX fixtures doivent PASS.

```bash
ssh user@gpu-host '... //examples/rqz:gemma4_gen_auto -- <weights> $TOKJSON --selftest-inputs /data/gemma4-zml-probe/gen_custom.safetensors'
# + fixture longue (positions jusqu'à ~1023 — couvre le régime p ≥ 512 où le masque sliding mord) :
ssh user@gpu-host '... //examples/rqz:gemma4_gen_auto -- <weights> $TOKJSON --selftest-inputs /data/gemma4-zml-probe/gen_auto_long.safetensors'
```
Expected (les deux runs) : `SELFTEST INPUTS PASS (N steps, cos/sin max_abs=… max_ratio=… de tol(p), masks bit-exact, positions ==)`
avec `max_ratio ≤ 1` (le critère). Si cos/sin FAIL avec un écart de plusieurs ordres de grandeur
au-dessus de tol(p) : la formule 3.1 est mal transcrite — diff sur les 8 premières valeurs.

- [ ] **Step 3.4 : Commit**

```bash
git add zml_runner/gemma4_gen_auto.zig
git commit -m "feat(gen-auto): inputs host (ropeFull partial-0.25 proportional, masques, positions, cache zéros) — selftest vs fixture PASS"
```

---

### Task 4 : Gather embeds bruts en streaming (pattern gchunk_auto)

**Files:** Modify: `zml_runner/gemma4_gen_auto.zig`

- [ ] **Step 4.1 : Reprendre le gather streaming**

Recopier/adapter `gemma4_gchunk_auto.zig:137-230` : offsets absolus
(`emb_t.offset + tok*row_bytes` sur `model.language_model.embed_tokens.weight` et
`...embed_tokens_per_layer.weight`), lecture positionnelle d'UNE ligne bf16 par step
(scratch réutilisé), création des buffers device `{1,1,D}` / `{1,1,LF}` bf16 par step.
**BRUT — aucun scaling host** (les ×√1536/×16 sont in-graph, spec §3.3).

- [ ] **Step 4.2 : Selftest vs fixture — mode `--selftest-gather <fixture>`**

Pour chaque step k de la fixture A1 : `gather(fed[k])` == `embeds[k]`/`embptls[k]` de la
fixture (bit-exact — les deux sont les lignes brutes bf16, cf 49_gen_custom_oracle.py:176-178).

Expected: `SELFTEST GATHER PASS (48 steps, bit-exact)`.

- [ ] **Step 4.3 : Commit**

```bash
git commit -m "feat(gen-auto): gather embeds bruts streaming (offsets positionnels) — selftest vs fixture bit-exact"
```

---

### Task 5 : Boucle autonome prefill-par-decode → **gate A1**

**Files:** Modify: `zml_runner/gemma4_gen_auto.zig`

- [ ] **Step 5.1 : Compile `forwardStep` + assemblage manuel des Bufferized**

Tensors symboliques du `Packed` créés À LA MAIN (pas de fixture) :

```zig
const packed_sym = PackedLong{
    .embeds = zml.Tensor.init(.{ L_MAX, 1, 1, D }, .bf16).withTags(.{ .step, .b, .s, .d }),
    .embptls = zml.Tensor.init(.{ L_MAX, 1, 1, LF }, .bf16).withTags(.{ .step, .b, .s, .lf }),
    .cos_full = zml.Tensor.init(.{ L_MAX, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
    .sin_full = zml.Tensor.init(.{ L_MAX, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
    .masks_sliding = zml.Tensor.init(.{ L_MAX, 1, 1, 1, L_MAX }, .f32).withTags(.{ .step, .b, .h, .q, .k }),
    .masks_full = zml.Tensor.init(.{ L_MAX, 1, 1, 1, L_MAX }, .f32).withTags(.{ .step, .b, .h, .q, .k }),
    .positions = zml.Tensor.init(.{L_MAX}, .i32).withTags(.{.step}),
};
```
(mêmes shapes/tags que `Packed.init` — vérifier contre engine.zig:250-275 ; si
`zml.Tensor.init` n'accepte pas cette forme, reprendre la construction symbolique de
`gemma4_gchunk_auto.zig:176` (`embeds_sym`) qui fait exactement ça.)

Embeds/embptls symboliques PER-STEP `{1,1,D}`/`{1,1,LF}` bf16 (comme gchunk_auto), cache
symbolique `engine.Cache` shapes ci-dessus, `Ctrl.initSymbolic()`. Compile :

```zig
var exe = try platform.compileFn(allocator, io, Model.forwardStep, .{ model, embeds_sym, embptls_sym, packed_sym, cache_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
```
(motif `compileFn` de gchunk_auto:204 ; si `compileFn` exige une fn nommée, wrapper
comptime comme gchunk_auto. Backend CUDA + fallback : recopier gemma4_gen_long_gpu.zig:80-92.)

Bufferized assemblés à la main depuis les arrays host (pattern multi-store E2,
`Buffer.fromSlice`-équivalent utilisé par gchunk_auto pour les lignes gather).

**Risque nommé (spec §5)** : 1er compile de `forwardStep` mono sur GPU. Attendu OK
(gemma4_gen_long_gpu compile le mono `.forward` op-identique). Si OOM compile : précédent
chunké (`forwardStageStep`) — STOP et documenter avant de pivoter.

- [ ] **Step 5.2 : Boucle prefill-par-decode + argmax + arrêt**

```zig
// ids = prompt templaté (Task 2). Phase 1 (prefill) : steps 0..ids.len-1, fed = ids[step],
// argmax IGNORÉ sauf au dernier (il produit le 1er token généré = s0). Phase 2 (gen) :
// fed = argmax précédent ; stop si tok == EOT_ID ou n_gen == limit.
//
// ⚠ ALIGNEMENT vs fixture 49 (vérifié scripts/49:151-160) : seq=[s0,t1,…], fed=seq[:n],
// expected=seq[1:n+1]. Notre generated[k] == seq[k] == fed[k] (s0 INCLUS — le texte de la
// réponse commence par s0, cf. tok.decode(fed) dans l'oracle). Donc :
//   • A1 compare  generated[0..n] == fed[0..n]   (PAS expected — off-by-one garanti sinon)
//   • en mode --oracle : limit = fed.len (le compte de steps vient de la FIXTURE ;
//     neutralise à la fois l'early-stop EOS et le défaut --max-tokens=200)
//   • hors oracle : limit = max_tokens (défaut 200), early-stop EOS actif
var fed: i64 = ids[0];
var step: usize = 0;
var generated: std.ArrayList(i64) = .empty;
while (true) : (step += 1) {
    // gather(fed) → buffers embeds/embptls (Task 4) ; ctrl.step = step ;
    // exe.call ; cache swap (motif gemma4_gen_long_gpu.zig:139-168) ; tok = argmaxOf(logits)
    if (step + 1 < ids.len) { fed = ids[step + 1]; continue; }        // prefill
    try generated.append(allocator, tok);                              // gen (s0 inclus)
    if (oracle_mode) { if (generated.items.len >= fed_fixture.len) break; }
    else if (tok == eot_id or generated.items.len >= max_tokens) break;
    if (step + 1 >= L_MAX) break; // garde L_MAX
    fed = tok;
}
```
Garde au lancement : `ids.len + limit <= L_MAX` ET `ids.len < SLIDING_WINDOW`
(le banc n'a jamais validé un prompt qui déborde la fenêtre — même assert que l'oracle 49).

- [ ] **Step 5.3 : Gate A1 — mode `--oracle <fixture>`**

`--oracle gen_custom.safetensors` : le prompt vient du manifest (ou re-passé en `--prompt`,
avec vérif ids == prompt_ids), la comparaison porte sur **`generated[0..n] == fed[0..n]`**
(le tensor `fed` de la fixture — alignement du step 5.2 ; comparer à `expected` serait un
off-by-one) ; la boucle court sur `fed.len` steps de génération (early-stop et
`--max-tokens` neutralisés — l'oracle 49 ne s'arrête pas à EOT non plus).

```bash
ssh user@gpu-host '... //examples/rqz:gemma4_gen_auto -- <weights> $TOKJSON --prompt "What is the capital of France? Answer in one word." --oracle /data/gemma4-zml-probe/gen_custom.safetensors'
```
Expected: `A1 PASS — 48/48 argmax-match (autonome complet, zéro input fixture)` (~1 s de
boucle). **Si FAIL : diagnostic au niveau LOGITS** (dumper les logits du 1er step divergent,
comparer à l'oracle — leçon méthodo, argmax trop grossier pour diagnostiquer).

- [ ] **Step 5.4 : Commit + tag**

```bash
git add zml_runner/gemma4_gen_auto.zig
git commit -m "feat(gen-auto): A1 — boucle autonome prefill-par-decode, 48/48 == HF sans fixture d'inputs"
git tag gate/gen-auto-a1-pass
```

---

### Task 6 : **Gate A2** — bout-en-bout long

- [ ] **Step 6.1 : Run long**

```bash
ssh user@gpu-host '... //examples/rqz:gemma4_gen_auto -- <weights> $TOKJSON --prompt "<prompt exact du step 1.4>" --oracle /data/gemma4-zml-probe/gen_auto_long.safetensors'
```
Expected: `A2 PASS — N/N argmax-match` (N = 999 : fixture générée au plafond structurel
exact, seq_len 25 + 999 = L_MAX — mieux que le « ≥ 1000 » nominal ; ~10 s de génération +
prefill). Reporter le tok/s (référence : 109 tok/s en replay).

- [ ] **Step 6.2 : Commit (doc statut) + tag `gate/gen-auto-a2-pass`**

---

### Task 7 : **Gate A3** — early-stop EOS + démo texte pure

- [ ] **Step 7.1 : Run autonome SANS oracle (early-stop actif)**

```bash
ssh user@gpu-host '... //examples/rqz:gemma4_gen_auto -- <weights> $TOKJSON --prompt "What is the capital of France? Answer in one word." --max-tokens 48'
```
Expected: le runner s'arrête de lui-même à l'EOT et affiche le texte détokenisé (attendu :
« Paris » et fin de tour). **Critère A3 (alignement du step 5.2 : generated ≡ fed, s0
inclus ; expected = décalé d'un cran)** : si le premier EOT est à l'index `i` dans
`expected` (relevé au step 1.3), il doit apparaître à l'index `i+1` de `generated`, donc
**stop après exactement `i+2` tokens générés** et `generated[last] == EOT`.

- [ ] **Step 7.2 : Commit + tag `gate/gen-auto-a3-pass`**

---

### Task 8 : Non-régression + non-vacuité

- [ ] **Step 8.1 : Non-régression** — re-runs :

```bash
# E1 : fixture /data/gemma4-zml-probe/fixtures/p5_7_8_gen.safetensors, tokens attendus
# [1018,6398,25967,53121] (signature exacte de la commande : docs/ZML_MODULAR_ENGINE_PLAN.md)
ssh user@gpu-host '... //examples/rqz:gemma4_engine_e1 -- /data/gemma4-zml-probe/weights/model.safetensors /data/gemma4-zml-probe/fixtures/p5_7_8_gen.safetensors'  # E1 4/4
ssh user@gpu-host '... //examples/rqz:gemma4_gen_long_gpu -- /data/gemma4-zml-probe/weights/model.safetensors /data/gemma4-zml-probe/gen_custom.safetensors'  # replay 48/48
```
Expected: PASS aux deux (le moteur n'a pas bougé — attendu trivial, mais exigé).

- [ ] **Step 8.2 : Non-vacuité** — perturber le template (retirer le `\n` après `user`),
rebuild, run `--ids-only` vs manifest → **ids ≠** (A0 discrimine) ; puis run `--oracle`
court → divergence. REMETTRE le template, re-run A0+A1 PASS, documenter les deux runs
dans le doc (pattern G2.3 : les FAIL de non-vacuité se consignent).

- [ ] **Step 8.3 : Commit (doc)**

---

### Task 9 : Documentation + clôture

- [ ] **Step 9.1** : `docs/GEN_AUTONOME_DESIGN.md` — remplir un tableau « Résultats » (gate
par gate, mesures, date), statut spec → implémentée. Y noter l'alignement retenu
`generated ≡ fed` (s0 inclus) — la table §4 de la spec disait « oracle = expected »,
le plan a corrigé (off-by-one) ; consigner pour éviter une confusion future.
- [ ] **Step 9.2** : `PLANNING.md` — item backlog « Runtime 100 % autonome » → [x] avec
pointeurs (runner, tags, tok/s).
- [ ] **Step 9.3** : `DOCUMENTATION.md` — courte section « runtime autonome » (usage CLI).
- [ ] **Step 9.4** : Commit final + push (`git push origin main` — repo public sync).

---

## Récap gates — TOUS RENDUS (10-11 juil 2026, verdicts détaillés : DESIGN § Résultats)

- [x] A0 : ids ZML == HF (2 prompts) + round-trip — tag `gate/gen-auto-a0-pass`
- [x] A1 : 48/48 == HF autonome complet — tag `gate/gen-auto-a1-pass`
- [x] A2 : critère N/N pré-enregistré **FAIL publié** (bifurcation marge 0.006, le contrôle
  replay bifurque au même point) → requalifié **PASS différentiel** « autonome ≥ replay »
  (décision Régis) — tag réel `gate/gen-auto-a2-diff-pass` (pas `a2-pass`). NB : le
  « 590 ≥ 589 » compare deux indexations décalées d'un cran (autonome s0-inclus vs replay
  s0-exclus) — normalisé, c'est le MÊME point de bifurcation, le ≥ tient par égalité.
- [x] A3 : early-stop à l'index EOT d'expected (+2, s0 inclus) — tag `gate/gen-auto-a3-pass`
- [x] Non-régression E1 + replay ; non-vacuité template (FAIL de perturbation documenté)
