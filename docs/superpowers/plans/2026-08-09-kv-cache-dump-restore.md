# Dump/restore du KV-cache 12B — plan d'implémentation (rév. 3)

> Rév. 3 après la 2ᵉ passe de revue (8 findings sur les corrections, 4 majeurs) : la fenêtre
> `KVLOAD-PERF` incluait la COMPILE (load scindé en 2 phases : manifest avant compile,
> tenseurs+chrono après) · la fixture oracle fabriquée butait sur la garde
> `positions[0] == ids.len` (`:1767-1770`) et `--oracle` lit DEUX clés (`positions`, `fed`) →
> `make-fixture` gagne un paramètre `positions0`, valeur spécifiée PAR RUN · les gardes
> dupliquées de `run()` (`:1788` place, `:1793` fenêtre) n'étaient pas conditionnées au mode
> load → point de câblage de `ids = resume.ids_fed` spécifié AVANT elles · le writer avait un
> `if` runtime sur un paramètre `comptime` (fmt) · pseudo-diff de la garde `--prompt` réécrit
> pour la structure `orelse blk:` réelle · `mutate-shape` tranché en sous-commande ferme ·
> justification « ≥ 1150 tokens sans EOS » retirée (elle venait de runs ORACLE, EOS désactivé
> — elle ne témoignait de rien sur le mode libre).

> Rév. 2 après revue adversariale (1 relecteur, 19 findings, 3 BLOQUANTS) : le bloc de dump
> alloue via l'allocateur COMPTÉ alors que la ligne `ALLOC-LOOP:` est émise après lui → DC6
> échouait par construction (fix : deltas figés dans des locales AVANT le dump) · tout
> `--load-cache` mourait à la garde « `--prompt` requis » `:1626-1630`, jamais modifiée par la
> rév. 1 · DC7 visait 3 900 tokens en mode LIBRE : l'arrêt EOS l'aurait coupé bien avant
> (antécédent irréalisable) → bascule `--oracle` (EOS désactivé) sur fixture fabriquée ·
> `--prompt-ids` n'existe pas (refus retiré) · `PERF-RESUME` excluait la lecture des 2,6 GiB
> (biais vers le PASS) → instrument `KVLOAD-PERF:` depuis l'entrée de `loadCacheFile` ·
> `header.writer(allocator)` : API ArrayList non prouvée dans ce repo → patron `allocPrint` ·
> + refus DC5 étendus (format/shape/step_next forgés), DC6 ≥ 220 tokens (sinon `RSS-DELTA:
> INEXECUTABLE`), commandes bash écrites en entier, ancrages numériques corrigés.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal :** `--dump-cache` sérialise l'état complet d'une génération 12B (4 caches KV + tokens
feedés + manifest auto-décrivant) dans UN fichier safetensors ; `--load-cache` réimplante cet
état et reprend la génération sans re-calculer le préfixe — prouvé bit-exact intra-process
(DC2), borné par la bistabilité inter-process (DC3), mordant prouvé (DC4), refus bruyants tous
VUS (DC1/DC5), interdit D10 intact (DC6), gain mesuré (DC7).

**Architecture :** un module std-only `kvdump.zig` (writer/reader safetensors généralisé
depuis `writeIdsSafetensors`, manifest `__metadata__`, checksums) ; le dump = `toSlice` des 4
buffers finaux vers les slices `host.cache_*` existantes + écriture fichier + re-zéro (point
unique, après la boucle ET après le gel des deltas ALLOC-LOOP, avant les `deinit`) ; le
restore = remplir `host.cache_*` depuis le fichier au lieu de `@memset(0)` + `generateOnce`
démarre à `step_next`/`fed_next`. **Aucun octet dans `engine.zig`, aucun changement de graphe.**

**Tech stack :** Zig 0.16.0-dev.2722 (bazel rules_zig), ZML (lecture seule), RTX 3090 (VM),
Python (`safetensors` pour l'inspection/mutants).

**Source de vérité des critères :** la spec
`docs/superpowers/specs/2026-08-09-kv-cache-dump-restore-design.md` **rév. 2**. En cas d'écart
plan/spec, **LA SPEC FAIT FOI**. Gates nommés par leur slug de spec (§5) : DC0..DC7.

**⚖ AVANT DE COMMENCER — deux décisions ouvertes (spec, en-tête)** : (1) sampling armé + dump
= refus bruyant [proposé] ; (2) gates sur 1280+4k seulement, 8k = dette [proposé]. Si Régis
n'a pas tranché au lancement : **appliquer les propositions** et le consigner dans le doc de
résultats.

**Conventions d'exécution** (identiques au plan D10, vérifiées le 30 juil) :
- Build (sur la VM) : `cd /data/rqz_workspace/zml && ./bazel.sh build -c opt --@rules_zig//zig/settings:mode=release_fast --@zml//platforms:cuda=true //examples/rqz:gemma4_g12auto //examples/rqz:gemma4_g12a4k`
- Déploiement depuis M1 : `cd ~/dev/gemma4-zml-probe/zml_runner && ./deploy_to_3090.sh`
  (variables `ZML_REMOTE`/`ZML_DST` : valeurs réelles dans le plan D10 §Conventions et la
  fiche mémoire `infra_rtx3090.md` — non répétées ici, anonymisation des nouveaux docs).
- `ssh <VM>` ci-dessous = la cible de `ZML_REMOTE`. **Chaque commande ssh ci-dessous commence
  par ce préambule, écrit une fois ici et sous-entendu ensuite** :

```bash
ssh <VM> 'cd /data/rqz_workspace/zml
B1=./bazel-bin/examples/rqz/gemma4_g12auto
B4=./bazel-bin/examples/rqz/gemma4_g12a4k
W=/data/gemma4-zml-probe/weights_12b
K=/data/gemma4-zml-probe/kvdump
PY=/data/venvs/g12b/bin/python
S71=/data/gemma4-zml-probe/scripts/71_kvdump_inspect.py
PROMPT="Explique-moi la fenêtre glissante d'\''attention en trois phrases."
<COMMANDE>'
```

- Chaque run de gate : capture séparée `> xxx.out.log 2> xxx.err.log` ; le log DOIT contenir
  `BUILD: mode=ReleaseFast` (sinon INEXÉCUTABLE — rebuild, ne pas requalifier).
- FAIL d'un gate ⇒ STOP, écrire le résultat, ne pas requalifier à chaud.
- Preuves : `docs/evidence/kvdump/` (versionné — `logs/` est gitignoré, précédent D10).
- ⚠ Ancrages au HEAD `dbf6c21`. **Re-localiser par le motif cité avant chaque édition** — les
  numéros bougeront dès la Task 3.
- ⚠ Risque EOS en mode libre : couvert par les pré-conditions elles-mêmes — si un run libre
  s'arrête avant sa borne (`stop_reason != max_tokens`) : **INEXÉCUTABLE**, pas FAIL. (Aucun
  précédent de tenue n'est invocable : les longs runs historiques étaient en mode oracle, où
  l'EOS est désactivé.)

---

## Task 0 : préliminaires — branche, VM, outillage

**Files :** aucun changement de code.

- [ ] **Step 0.1 : branche** — `cd ~/dev/gemma4-zml-probe && git checkout -b kv-dump-restore`

- [ ] **Step 0.2 : GPU libre + répertoire de travail**

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader; mkdir -p $K
```
Attendu : `0 MiB` (ou < 500 MiB). Sinon STOP, identifier le processus.

- [ ] **Step 0.3 : espace disque (les dumps 4k pèsent ~2,62 GiB, il en faudra ~2 + les 1280)**

```bash
df -h /data | tail -1
```
Attendu : ≥ 15 G libres. Sinon STOP (fiche `infra_rtx3090.md`, épisodes disque).

- [ ] **Step 0.4 : décision xxh64 vs crc32 côté Python (décidée ICI, pas improvisée)**

```bash
$PY -c "import xxhash; print(xxhash.xxh64(b'x').intdigest())" || /data/venvs/g12b/bin/pip install xxhash
```
Si l'install échoue : **basculer TOUT le chantier sur crc32** (`std.hash.crc.Crc32` Zig,
`zlib.crc32` Python), clés manifest `*_crc32`, consigné au doc de résultats. Une seule
famille de checksum, partout.

- [ ] **Step 0.5 : API Zig des hachages (F9-style : confirmer, pas supposer)**

```bash
find /data/rqz_workspace -name xxhash.zig -path "*std*" 2>/dev/null | head -3
grep -n "pub fn hash" <chemin trouvé> | head -5
```
Noter la signature exacte de `XxHash64.hash` dans le commit de Task 2. Vérifier de même
`std.Io.File.readPositionalAll` (retour) et le parseur JSON — **copier le patron qui compile
déjà** : `gemma4_bbatch.zig:406-426`.

---

## Task 1 : DC0 témoin — HLO AVANT toute modification

**Files :** aucun changement de code.

- [ ] **Step 1.1 : dump HLO des deux variantes, md5 dans un fichier**

Reprendre la procédure exacte du témoin GC0/S2-G
(`grep -rn "before_optimizations\|XLA_FLAGS" docs/superpowers/plans/2026-07-29-generation-config.md`)
pour `gemma4_g12auto` ET `gemma4_g12a4k`, puis :

```bash
md5sum $K/hlo_witness_1280.txt $K/hlo_witness_4k.txt > $K/hlo_witness.md5
cat $K/hlo_witness.md5
```

- [ ] **Step 1.2 : rapatrier + commit**

```bash
scp <VM>:/data/gemma4-zml-probe/kvdump/hlo_witness.md5 ~/dev/gemma4-zml-probe/docs/evidence/kvdump/
git add docs/evidence/kvdump/ && git commit -m "kvdump(DC0) : témoins HLO 1280+4k AVANT la première ligne de code"
```

---

## Task 2 : `kvdump.zig` — module std-only + selftest host DC1

**Files :**
- Create: `zml_runner/kvdump.zig`
- Modify: `zml_runner/BUILD.bazel:527,537,547` (ajouter `"kvdump.zig"` aux `srcs` des 3
  cibles, à côté de `"gencfg.zig"`)
- Modify: `zml_runner/gemma4_g12auto.zig` — nouveau mode `--selftest-kvdump-io <dir>`
  (early-return AVANT chargement des poids, patron `--selftest-draw` `:1137` + dispatch
  `:1614` ; l'ajouter aussi à la garde d'exclusivité `--repl` `:1580-1588` qui liste
  nommément tous les `--selftest-*`)

- [ ] **Step 2.1 : le module.** Code complet ci-dessous (auto-portant — le writer utilise
  `allocPrint`/`append`/`appendSlice`, jamais `ArrayList.writer(allocator)`, API non prouvée
  dans ce repo — finding 5) :

```zig
/// Écrit un safetensors : __metadata__ d'abord, puis les tenseurs dans l'ordre donné,
/// data_offsets contigus. String-building par allocPrint/appendSlice (patron joinKeys,
/// gemma4_g12auto.zig:1386-1396) — jamais {any}, jamais d'API writer non prouvée.
pub fn write(allocator: std.mem.Allocator, io: std.Io, path: []const u8, tensors: []const TensorOut, meta: []const MetaKV) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.appendSlice(allocator, "{\"__metadata__\":{");
    for (meta, 0..) |kv, i| {
        if (i != 0) try header.append(allocator, ',');
        const frag = try std.fmt.allocPrint(allocator, "\"{s}\":\"{s}\"", .{ kv.k, kv.v });
        defer allocator.free(frag);
        try header.appendSlice(allocator, frag);
    }
    try header.append(allocator, '}');
    var off: usize = 0;
    for (tensors) |t| {
        const head = try std.fmt.allocPrint(allocator, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype });
        defer allocator.free(head);
        try header.appendSlice(allocator, head);
        for (t.shape, 0..) |d, i| {
            if (i != 0) try header.append(allocator, ','); // fmt est comptime : pas de `if` runtime dans allocPrint
            const dim = try std.fmt.allocPrint(allocator, "{d}", .{d});
            defer allocator.free(dim);
            try header.appendSlice(allocator, dim);
        }
        const tail = try std.fmt.allocPrint(allocator, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + t.bytes.len });
        defer allocator.free(tail);
        try header.appendSlice(allocator, tail);
        off += t.bytes.len;
    }
    try header.append(allocator, '}');
    var len_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_le, header.items.len, .little);
    const f = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer f.close(io);
    try f.writePositionalAll(io, &len_le, 0);
    try f.writePositionalAll(io, header.items, 8);
    var pos: usize = 8 + header.items.len;
    for (tensors) |t| {
        try f.writePositionalAll(io, t.bytes, pos);
        pos += t.bytes.len;
    }
}
```

Le reste du module, complet (API `parseFromSliceLeaky`/`readPositionalAll`/`length` à
confirmer Step 0.5 contre `gemma4_bbatch.zig:406-426` — noter les écarts dans le commit) :

```zig
// zml_runner/kvdump.zig — dump/restore de l'état de génération 12B.
// Spec : docs/superpowers/specs/2026-08-09-kv-cache-dump-restore-design.md §4.1.
// Std-only (pas de zml) : testable host, réutilisable par les 3 cibles.
const std = @import("std");

pub const FORMAT = "g12-kvdump-v1";

pub const TensorOut = struct {
    name: []const u8,
    dtype: []const u8, // "F32" | "I32"
    shape: []const i64,
    bytes: []const u8,
};

pub const MetaKV = struct { k: []const u8, v: []const u8 };

pub fn xxh64(bytes: []const u8) u64 {
    return std.hash.XxHash64.hash(0, bytes); // signature confirmée en Task 0.5
}

pub const Entry = struct { dtype: []const u8, shape: []i64, off0: usize, off1: usize };

pub const Header = struct {
    arena: std.heap.ArenaAllocator,
    meta: std.StringHashMapUnmanaged([]const u8),
    entries: std.StringHashMapUnmanaged(Entry),
    data_base: usize, // 8 + header_len : base absolue des data_offsets

    pub fn deinit(self: *Header) void {
        self.arena.deinit();
    }

    pub fn metaGet(self: *const Header, k: []const u8) ?[]const u8 {
        return self.meta.get(k);
    }
};

pub const ReadError = error{ KvDumpBadFormat, KvDumpTruncated };

/// Lit et parse le header. Ne lit AUCUN tenseur. `file` reste ouvert, possédé par l'appelant.
pub fn readHeader(gpa: std.mem.Allocator, io: std.Io, file: std.Io.File) !Header {
    var len_le: [8]u8 = undefined;
    const n0 = try file.readPositionalAll(io, &len_le, 0);
    if (n0 != 8) return ReadError.KvDumpTruncated;
    const hlen = std.mem.readInt(u64, &len_le, .little);
    if (hlen == 0 or hlen > 1 << 20) return ReadError.KvDumpBadFormat;
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const hbuf = try a.alloc(u8, hlen);
    const n1 = try file.readPositionalAll(io, hbuf, 8);
    if (n1 != hlen) return ReadError.KvDumpTruncated;
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, hbuf, .{});
    if (parsed != .object) return ReadError.KvDumpBadFormat;
    var meta: std.StringHashMapUnmanaged([]const u8) = .empty;
    var entries: std.StringHashMapUnmanaged(Entry) = .empty;
    var it = parsed.object.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        if (std.mem.eql(u8, key, "__metadata__")) {
            if (e.value_ptr.* != .object) return ReadError.KvDumpBadFormat;
            var mit = e.value_ptr.object.iterator();
            while (mit.next()) |m| {
                if (m.value_ptr.* != .string) return ReadError.KvDumpBadFormat;
                try meta.put(a, m.key_ptr.*, m.value_ptr.string);
            }
        } else {
            const obj = e.value_ptr.*;
            if (obj != .object) return ReadError.KvDumpBadFormat;
            const dt = obj.object.get("dtype") orelse return ReadError.KvDumpBadFormat;
            const sh = obj.object.get("shape") orelse return ReadError.KvDumpBadFormat;
            const offs = obj.object.get("data_offsets") orelse return ReadError.KvDumpBadFormat;
            if (dt != .string or sh != .array or offs != .array or offs.array.items.len != 2) return ReadError.KvDumpBadFormat;
            const shape = try a.alloc(i64, sh.array.items.len);
            for (sh.array.items, 0..) |v, i| {
                if (v != .integer) return ReadError.KvDumpBadFormat;
                shape[i] = v.integer;
            }
            if (offs.array.items[0] != .integer or offs.array.items[1] != .integer) return ReadError.KvDumpBadFormat;
            try entries.put(a, key, .{
                .dtype = dt.string,
                .shape = shape,
                .off0 = @intCast(offs.array.items[0].integer),
                .off1 = @intCast(offs.array.items[1].integer),
            });
        }
    }
    return .{ .arena = arena, .meta = meta, .entries = entries, .data_base = 8 + hlen };
}

/// Lit un tenseur ENTIER dans `dest` (taille exacte exigée), puis vérifie son checksum
/// contre `expected_xxh64` (ordre : lecture → hash → comparaison, spec §4.3).
pub fn readTensorInto(io: std.Io, file: std.Io.File, h: *const Header, name: []const u8, dest: []u8, expected_xxh64: u64) !void {
    const e = h.entries.get(name) orelse return error.KvDumpBadFormat;
    if (e.off1 - e.off0 != dest.len) return error.KvDumpShapeMismatch;
    const n = try file.readPositionalAll(io, dest, h.data_base + e.off0);
    if (n != dest.len) return ReadError.KvDumpTruncated;
    if (xxh64(dest) != expected_xxh64) return error.KvDumpChecksumMismatch;
}

/// Fingerprint d'un checkpoint safetensors : taille du fichier + xxh64 de (8 octets de
/// longueur + header JSON). Jamais les 24 Go de données. Spec §4.1.
pub fn ckptFingerprint(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !struct { bytes: u64, hdr_xxh64: u64 } {
    const f = try std.Io.Dir.openFile(.cwd(), io, path, .{});
    defer f.close(io);
    var len_le: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &len_le, 0) != 8) return ReadError.KvDumpTruncated;
    const hlen = std.mem.readInt(u64, &len_le, .little);
    const buf = try gpa.alloc(u8, 8 + hlen);
    defer gpa.free(buf);
    @memcpy(buf[0..8], &len_le);
    if (try f.readPositionalAll(io, buf[8..], 8) != hlen) return ReadError.KvDumpTruncated;
    const size = try f.length(io);
    return .{ .bytes = size, .hdr_xxh64 = xxh64(buf) };
}
```

- [ ] **Step 2.2 : `--selftest-kvdump-io <dir>` = le gate DC1, host-only.** Dans le binaire,
  early-return AVANT `checkVram`/chargement des poids. Contenu :

```
1. tenseurs synthétiques : a = [24]f32 (valeurs 0..23), b = [6]i32 — shapes {2,3,4} et {6}
2. write(<dir>/self.kvdump) avec manifest {format, l_max="9999", step_next="6", a_xxh64, b_xxh64}
3. readHeader + readTensorInto des 2 tenseurs dans des buffers neufs
4. PASS si : bytes bit-identiques (memcmp), meta round-trip (l_max=="9999")
5. MUTANT INTÉGRÉ : réécrire le fichier avec 1 octet flippé dans les données de `a`
   (réouverture + writePositionalAll d'1 octet à data_base+3), relire →
   readTensorInto DOIT rendre error.KvDumpChecksumMismatch. S'il ne le rend pas : FAIL.
6. log final : "KVIO: round-trip PASS + mutant VU (ChecksumMismatch)" / exit 1 sinon
```

- [ ] **Step 2.3 : BUILD.bazel** — `"kvdump.zig"` dans les 3 `srcs`.

- [ ] **Step 2.4 : build + run DC1 + commit**

```bash
./bazel.sh build -c opt --@rules_zig//zig/settings:mode=release_fast --@zml//platforms:cuda=true //examples/rqz:gemma4_g12auto
$B1 $W/model.safetensors $W/tokenizer.json --selftest-kvdump-io /tmp/kvio > kvio.out.log 2> kvio.err.log
grep "KVIO:" kvio.err.log
```
Attendu : `KVIO: round-trip PASS + mutant VU (ChecksumMismatch)`. Rapatrier le log →
`docs/evidence/kvdump/`. Tag `gate/dc1-pass` (le volet Python complémentaire arrive Task 7).
Commit : `kvdump : module std-only + selftest host DC1 (round-trip + mutant vu)`.

---

## Task 3 : flags, refus de combinaisons, usage

**Files :**
- Modify: `zml_runner/gemma4_g12auto.zig` — struct `Args` (`:145-184`), parsing (zone
  `--seed` `:349-352`), usage (`:186-201`), zone fail-fast de `run()` (motif `checkVram`,
  appel `:1744` — les refus se placent AVANT lui, avec les gardes early existantes)

- [ ] **Step 3.1 : deux champs `Args`**

```zig
    dump_cache: ?[]const u8 = null, // --dump-cache <fichier> : état E1-E4 en fin de generateOnce
    load_cache: ?[]const u8 = null, // --load-cache <fichier> : reprise sans prefill
```

Parsing : même motif que `--out-ids`. Usage : ajouter
`"[--dump-cache F] [--load-cache F (+ --max-tokens | --oracle ; exclut --prompt/--repl)] "`.

- [ ] **Step 3.2 : les CINQ refus — zone fail-fast (motif `checkVram`), AVANT compile**

```zig
    if (args.load_cache != null and args.repl) {
        log.err("--load-cache + --repl non supporté (spec §3 : sémantique multi-tour absente)", .{});
        return error.LoadCacheReplUnsupported;
    }
    if (args.dump_cache != null and args.repl) {
        log.err("--dump-cache + --repl non supporté v1 (spec §4.5 : un flag inopérant serait un mensonge)", .{});
        return error.DumpCacheReplUnsupported;
    }
    if (args.load_cache != null and args.prompt != null) {
        log.err("--load-cache + --prompt : le contexte vient du dump, pas d'un prompt (spec §3)", .{});
        return error.LoadCacheWithPrompt;
    }
    if (args.dump_cache != null and args.seed != null) { // décision ouverte n°1 (proposition appliquée)
        log.err("--dump-cache + sampling armé non supporté v1 (état PRNG non sérialisé — spec §3, dette)", .{});
        return error.DumpWithSamplingArmed;
    }
```

(⚠ vérifier les noms exacts des champs `prompt`/`repl` dans `Args` — re-localiser. Le 5ᵉ refus
— `SequenceTooLong` sur `--max-tokens` trop grand — vit dans `loadCacheManifest`, Task 5.)

- [ ] **Step 3.3 : ⚠ LA GARDE `--prompt` REQUIS (`:1626-1630`) — finding bloquant n°2.**
  La garde existante « `--prompt` est requis (sauf `--repl`) » tuerait tout `--load-cache`
  avant même la lecture du fichier. ⚠ Sa forme réelle est un **`orelse blk:`** (pas un simple
  `if` — le pseudo-diff de la rév. 2 était ingreppable) : LIRE `:1626-1630` d'abord, puis
  étendre la branche d'exemption : là où `--repl` obtient un prompt vide/sentinelle au lieu
  de `error.MissingArgument`, `args.load_cache != null` doit obtenir la même chose (les ids
  viennent du manifest). Court-circuiter aussi la tokenisation (motif `promptToIds`,
  `:1655-1660`) quand `args.load_cache != null`. Le diff exact s'écrit sur pièce — le
  critère de réussite est mécanique : Step 5.4 démarre sans `--prompt`.

- [ ] **Step 3.4 : build + exercer les 4 refus de flags SUR LA VM (échec avant compile, ~1 s)**

```bash
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /tmp/x --repl; echo "exit=$?"
$B1 $W/model.safetensors $W/tokenizer.json --dump-cache /tmp/x --repl; echo "exit=$?"
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /tmp/x --prompt "a"; echo "exit=$?"
$B1 $W/model.safetensors $W/tokenizer.json --dump-cache /tmp/x --prompt "a" --seed 1 --max-tokens 2; echo "exit=$?"
```
Attendu : 4 messages qualifiés + exit ≠ 0. Sorties → `docs/evidence/kvdump/dc5_flags.log`.

- [ ] **Step 3.5 : commit** — `kvdump(flags) : --dump-cache/--load-cache, garde --prompt apprise, 4 refus exercés`

---

## Task 4 : le dump — point unique, APRÈS le gel des deltas ALLOC-LOOP

**Files :**
- Modify: `zml_runner/gemma4_g12auto.zig` — signature `generateOnce` (`:2146`), fin de
  fonction (motif `const gen_elapsed`, avant les `cache_buf.*.deinit()` `:2436-2439`), la
  ligne `ALLOC-LOOP:` (`:2447-2451`), les 3 sites d'appel (`:2090`, `:2100`, `:2129`)

- [ ] **Step 4.1 : signature** — paramètre `dump_cache_path: ?[]const u8`, threadé depuis
  `args.dump_cache` au site one-shot/oracle ; `null` aux sites REPL (le refus 3.2 garantit
  qu'on n'y arrive jamais avec le flag — le `null` est un invariant, pas un silence).

- [ ] **Step 4.2 : ⚠ GEL DES DELTAS AVANT LE DUMP — finding bloquant n°1.** Immédiatement
  après `const gen_elapsed = …untilNow…` et AVANT le bloc de dump :

```zig
    // D10 : deltas ALLOC-LOOP FIGÉS ICI, avant tout travail post-boucle (le dump alloue via
    // l'allocateur compté ; sans ce gel, la ligne ALLOC-LOOP l'imputerait à la boucle et
    // DC6 échouerait par construction — finding bloquant de revue kvdump).
    const al_alloc = counter.n_alloc - al0_alloc;
    const al_resize = counter.n_resize - al0_resize;
    const al_remap = counter.n_remap - al0_remap;
    const al_free = counter.n_free - al0_free;
    const al_bytes = counter.bytes_alloc - al0_bytes;
```

Et la ligne `ALLOC-LOOP:` (`:2447-2451`) passe des expressions `counter.* - al0_*` aux locales
`al_*` (5 substitutions, aucun autre changement de la ligne).

- [ ] **Step 4.3 : le bloc de dump** — inséré après le gel, avant `cache_buf.sl_k.deinit()`.
  (Validé par la revue : contrat de donation respecté — les buffers lus sont les SORTIES du
  dernier step, le swap `:2370-2375` précède tous les `break` ; invariant
  `ids_fed.len == step_next` correct pour les 4 `stop_reason`.)

```zig
    if (dump_cache_path) |dp| {
        // d2h des 4 buffers FINAUX vers les slices host EXISTANTES (zéro alloc de 2,6 GiB,
        // plafond B10 intact — spec §4.2). toSlice : le mécanisme D2H prouvé (D10/C2).
        try cache_buf.sl_k.toSlice(io, zml.Slice.init(cache_sym.sl_k.shape(), host.cache_sl_k));
        try cache_buf.sl_v.toSlice(io, zml.Slice.init(cache_sym.sl_v.shape(), host.cache_sl_v));
        try cache_buf.fl_k.toSlice(io, zml.Slice.init(cache_sym.fl_k.shape(), host.cache_fl_k));
        try cache_buf.fl_v.toSlice(io, zml.Slice.init(cache_sym.fl_v.shape(), host.cache_fl_v));

        if (generated.items.len == 0) {
            log.err("KVDUMP: aucun token généré — état sans fed_next, dump refusé", .{});
            return error.KvDumpInconsistentState;
        }
        const step_next: usize = step + 1;
        // ids_fed = ids ++ generated[0..len-1] (le dernier généré n'a PAS été feedé).
        // Invariant spec §4.2 : ids_fed.len == step_next — assertion dure, jamais silencieuse.
        const n_gen_fed = generated.items.len - 1;
        if (ids.len + n_gen_fed != step_next) {
            log.err("KVDUMP: invariant cassé ids({d})+gen_fed({d}) != step_next({d})", .{ ids.len, n_gen_fed, step_next });
            return error.KvDumpInconsistentState;
        }
        const ids_fed = try allocator.alloc(i32, step_next);
        defer allocator.free(ids_fed);
        for (ids, 0..) |t, k| ids_fed[k] = @intCast(t);
        for (0..n_gen_fed) |k| ids_fed[ids.len + k] = @intCast(generated.items[k]);
        const fed_next: i64 = generated.items[generated.items.len - 1];

        try dumpCacheFile(allocator, io, dp, host, ids_fed, step_next, fed_next, stop_reason, scfg);

        // Contrat « cache ZÉROS par génération » (:2162) restauré pour l'appel suivant.
        @memset(host.cache_sl_k, 0);
        @memset(host.cache_sl_v, 0);
        @memset(host.cache_fl_k, 0);
        @memset(host.cache_fl_v, 0);
    }
```

- [ ] **Step 4.4 : `dumpCacheFile`** — fonction du fichier (elle connaît `HostInputs` et les
  constantes), PLUS la clé manifest `sampling` (spec §4.1 rév. 2) :

```zig
fn dumpCacheFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, host: anytype, ids_fed: []const i32, step_next: usize, fed_next: i64, stop_reason: anytype, scfg: *const sampling.SamplingCfg) !void {
    const sl_shape = [_]i64{ @intCast(NUM_SLIDING_SLOTS), 1, KVH_SL, L_MAX, HD_S };
    const fl_shape = [_]i64{ @intCast(NUM_FULL_SLOTS), 1, KVH_FL, L_MAX, HD_F };
    const ids_shape = [_]i64{@intCast(ids_fed.len)};
    const ids_bytes = std.mem.sliceAsBytes(ids_fed);
    const fp = try kvdump.ckptFingerprint(allocator, io, g_ckpt_path); // cf note ci-dessous
    var buf: [16][64]u8 = undefined;
    const sampling_str = if (scfg.warpersArmed())
        try std.fmt.bufPrint(&buf[10], "T={d},top_k={d},top_p={d}", .{ scfg.temperature, scfg.top_k, scfg.top_p })
    else
        "off";
    const meta = [_]kvdump.MetaKV{
        .{ .k = "format", .v = kvdump.FORMAT },
        .{ .k = "l_max", .v = try std.fmt.bufPrint(&buf[0], "{d}", .{L_MAX}) },
        .{ .k = "step_next", .v = try std.fmt.bufPrint(&buf[1], "{d}", .{step_next}) },
        .{ .k = "fed_next", .v = try std.fmt.bufPrint(&buf[2], "{d}", .{fed_next}) },
        .{ .k = "stop_reason", .v = @tagName(stop_reason) },
        .{ .k = "ckpt_bytes", .v = try std.fmt.bufPrint(&buf[3], "{d}", .{fp.bytes}) },
        .{ .k = "ckpt_hdr_xxh64", .v = try std.fmt.bufPrint(&buf[4], "{x}", .{fp.hdr_xxh64}) },
        .{ .k = "gencfg_path", .v = g_gencfg_path_logged },
        .{ .k = "build_mode", .v = build_mode_banner },
        .{ .k = "sampling", .v = sampling_str },
        .{ .k = "sl_k_xxh64", .v = try std.fmt.bufPrint(&buf[5], "{x}", .{kvdump.xxh64(host.cache_sl_k)}) },
        .{ .k = "sl_v_xxh64", .v = try std.fmt.bufPrint(&buf[6], "{x}", .{kvdump.xxh64(host.cache_sl_v)}) },
        .{ .k = "fl_k_xxh64", .v = try std.fmt.bufPrint(&buf[7], "{x}", .{kvdump.xxh64(host.cache_fl_k)}) },
        .{ .k = "fl_v_xxh64", .v = try std.fmt.bufPrint(&buf[8], "{x}", .{kvdump.xxh64(host.cache_fl_v)}) },
        .{ .k = "ids_fed_xxh64", .v = try std.fmt.bufPrint(&buf[9], "{x}", .{kvdump.xxh64(ids_bytes)}) },
    };
    const tensors = [_]kvdump.TensorOut{
        .{ .name = "sl_k", .dtype = "F32", .shape = &sl_shape, .bytes = host.cache_sl_k },
        .{ .name = "sl_v", .dtype = "F32", .shape = &sl_shape, .bytes = host.cache_sl_v },
        .{ .name = "fl_k", .dtype = "F32", .shape = &fl_shape, .bytes = host.cache_fl_k },
        .{ .name = "fl_v", .dtype = "F32", .shape = &fl_shape, .bytes = host.cache_fl_v },
        .{ .name = "ids_fed", .dtype = "I32", .shape = &ids_shape, .bytes = ids_bytes },
    };
    try kvdump.write(allocator, io, path, &tensors, &meta);
    const total = 2 * host.cache_sl_k.len + 2 * host.cache_fl_k.len + ids_bytes.len;
    log.info("KVDUMP: {s} l_max={d} step_next={d} fed_next={d} ids={d} octets={d} xxh64_ok", .{ path, L_MAX, step_next, fed_next, ids_fed.len, total });
}
```

À résoudre en re-localisant (câblage, pas design) : `g_ckpt_path` = positionnel des poids
tenu par `run()` (le threader) ; `g_gencfg_path_logged` = le chemin que `GENCFG:` logue ;
`build_mode_banner` = la source unique de la bannière `BUILD: mode=` ; `warpersArmed()` =
la méthode réelle de `SamplingCfg` (grep `pathArmed`/`drawArmed` `sampling.zig` et prendre
celle qui dit « des warpers sont demandés », la nommer si elle manque).

- [ ] **Step 4.5 : build + premier dump réel (1280) + inspection Python**

```bash
$B1 $W/model.safetensors $W/tokenizer.json --prompt "$PROMPT" --max-tokens 16 --dump-cache $K/first.kvdump > t4.out.log 2> t4.err.log
grep "KVDUMP:" t4.err.log
$PY -c "
from safetensors import safe_open
with safe_open('$K/first.kvdump'.replace('\$K','/data/gemma4-zml-probe/kvdump'), framework='numpy') as f:
    print(sorted(f.keys())); print(f.metadata())"
```
Attendu : les 5 clés (`fl_k fl_v ids_fed sl_k sl_v`), manifest complet, et
`octets=880803840 + len(ids_fed)*4` (prédiction spec §2bis — **si l'octet diffère, STOP
diagnostic**).

- [ ] **Step 4.6 : commit** — `kvdump(dump) : deltas gelés + point unique + manifest complet + re-zéro du contrat`

---

## Task 5 : le restore — validations, chargement, reprise, chrono

**Files :**
- Modify: `zml_runner/gemma4_g12auto.zig` — `run()` (chemin one-shot, modèle du site
  `:2090`), `generateOnce` (signature, garde SLIDING_WINDOW `:2158`, init `fed`/`step`
  `:2219-2220`, ligne `PERF :` `:2481` env.)

- [ ] **Step 5.1 : `loadCacheManifest` + `loadCacheTensors` + `Resume`** — nouvelles
  fonctions de `gemma4_g12auto.zig` :

```zig
const Resume = struct { step_next: usize, fed_next: i64, ids_fed: []u32, t_load0: std.Io.Timestamp };
```

**⚠ EN DEUX PHASES (rév. 3, spec §4.3) — la compile ne doit être NI dans le chrono, NI avant
les validations de forme :**

`fn loadCacheManifest(allocator, io, path, ckpt_path, limit) !ManifestCheck` — **AVANT la
compile** (fail-fast, ne lit que le header, quelques Ko) :
1. `openFile` + `kvdump.readHeader` (le `file` + `Header` sont RETOURNÉS, gardés ouverts) ;
2. `format` == `kvdump.FORMAT` sinon `error.KvDumpBadFormat` ;
3. `l_max` (parseInt du meta) == `L_MAX` sinon `error.KvDumpVariantMismatch` ;
4. shapes/dtype des 4 entrées == shapes compilées (mêmes constantes que `dumpCacheFile`)
   sinon `error.KvDumpShapeMismatch` ;
5. `kvdump.ckptFingerprint(ckpt_path)` == `ckpt_bytes`/`ckpt_hdr_xxh64` sinon
   `error.KvDumpCheckpointMismatch` ;
6. invariant `ids_fed` (longueur d'entrée) == `step_next` sinon
   `error.KvDumpInconsistentState` ;
7. garde `step_next + limit > L_MAX` ⇒ `error.SequenceTooLong` (= refus DC5(g)) ;
8. si `gencfg_path` du manifest ≠ chemin résolu courant : **WARN logué**, pas un refus.

`fn loadCacheTensors(allocator, io, mc, host) !Resume` — **APRÈS la compile** :
9. `t_load0 = std.Io.Timestamp.now(io, .awake)` — **première ligne** (l'instrument de C-D :
   lecture des GiB incluse, compile exclue) ;
10. `readTensorInto` des 4 caches DANS `host.cache_*` (checksum intégré) + `ids_fed` (alloc
    i32→u32, possédé par l'appelant) avec son checksum ;
11. log `KVLOAD: {s} l_max={d} step_next={d} fed_next={d} ids={d} (reprise sans prefill)`.

- [ ] **Step 5.2 : `generateOnce` — paramètre `resume_state: ?Resume`** :

```zig
    // en tête de fonction (motif `var fed: i64 = @intCast(ids[0]);`, :2219-2220)
    var fed: i64 = @intCast(ids[0]);
    var step: usize = 0;
    if (resume_state) |rs| {
        step = rs.step_next;
        fed = rs.fed_next;
    }
```

```zig
    // garde :2158 — un état repris dépasse légitimement la fenêtre (spec §4.3) :
    if (resume_state == null and ids.len >= @as(usize, @intCast(SLIDING_WINDOW))) {
```

PLUS :
  - au **premier token généré** en mode resume (transition `in_gen_phase`, premier append à
    `generated`) : loguer
    `KVLOAD-PERF: chargement+h2d+reprise -> 1er token en {d:.3}s` mesuré depuis
    `resume_state.?.t_load0` (finding 8 : la lecture des 2,6 GiB DOIT être dans la fenêtre —
    un chrono qui l'exclut est biaisé vers le PASS) ;
  - en mode resume, la ligne `PERF :` est remplacée par
    `PERF-RESUME : reprise @step={d}, {d} tokens générés en {d:.3}s ({d:.1} tok/s)` (le champ
    « prefill » d'un run repris serait un mensonge).

- [ ] **Step 5.3 : câblage `run()` — ⚠ y compris les gardes DUPLIQUÉES (finding N3).**
  Chemin `--load-cache` :
  1. `loadCacheManifest` dans la zone fail-fast (avec les refus de Step 3.2, avant compile) ;
  2. compile + chargement des poids (chemin existant inchangé) ;
  3. `loadCacheTensors` → `resume` ; **`ids = resume.ids_fed` assigné ICI, AVANT la zone des
     pré-checks de `run()`** ;
  4. les pré-checks dupliqués de `run()` : `ids.len + limit > L_MAX` (`:1788`) **reste actif
     tel quel** (équivalent à la garde 7 du manifest — redondance saine) ;
     `ids.len >= SLIDING_WINDOW` (`:1793`) **conditionné** `args.load_cache == null and …`
     (même raison que `:2158` : la garde protège le prefill, qui n'a pas lieu) ; le check
     oracle `positions[0] == ids.len` (`:1767-1770`) **reste actif** — la fixture d'un run
     load+oracle DOIT être fabriquée avec `positions0 = step_next` du dump (Task 7.1/9.2) ;
  5. appel `generateOnce` (modèle `:2090`) avec `resume_state` non-nul,
     `oracle_ids`/`max_tokens` selon flags. Les autres sites passent `null`.

- [ ] **Step 5.4 : build + round-trip réel 1280 (sanity, pas un gate)**

```bash
$B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/first.kvdump --max-tokens 8 > t5.out.log 2> t5.err.log
grep -E "KVLOAD:|KVLOAD-PERF|PERF-RESUME" t5.err.log
```
Attendu : `KVLOAD:` avec le `step_next` du dump Task 4, `KVLOAD-PERF:` présent, 8 tokens de
texte en continuité plausible.

- [ ] **Step 5.5 : commit** — `kvdump(restore) : 10 validations + reprise sans prefill + chrono KVLOAD-PERF`

---

## Task 6 : DC2 — `--selftest-kvdump-eq` (équivalence intra-process, bit-exacte)

**Files :**
- Modify: `zml_runner/gemma4_g12auto.zig` — nouveau selftest (zone `--selftest-draw` `:1137`,
  dispatch, garde d'exclusivité `--repl` `:1580-1588`), `generateOnce` gagne
  `t5_capture: ?*std.ArrayList(Top5)` et `ids_capture: ?*std.ArrayList(i64)` (copies APRÈS
  la boucle, à côté du log `GENCFG: suppress a mordu` — hors fenêtre ALLOC-LOOP gelée)

- [ ] **Step 6.1 : captures** — si non-nuls : `appendSlice` de `gen_top5.items` et
  `generated.items` (post-boucle, post-gel — aucune interaction avec DC6).

- [ ] **Step 6.2 : l'orchestration `--selftest-kvdump-eq`** (exige `--prompt` ; le refus
  `LoadCacheWithPrompt` ne s'applique pas ici — le selftest appelle
  `loadCacheManifest`/`loadCacheTensors` directement, pas le flag) :

```
appel (1) : generateOnce(max_tokens=16, dump_cache=$K/dc2.kvdump, captures c1)
appel (2) : generateOnce(max_tokens=48, captures c2)              — référence, cache zéros
pré-condition A : stop_reason des deux appels == max_tokens, sinon
    "KVEQ: INEXECUTABLE — arrêt prématuré (stop={s})" + exit 3
pré-condition B : c1.ids[0..16] == c2.ids[0..16] bit-à-bit, sinon
    "KVEQ: INEXECUTABLE — déterminisme intra-process non vérifié" + exit 3
appel (3) : loadCacheManifest + loadCacheTensors($K/dc2.kvdump) ; generateOnce(resume, max_tokens=32, captures c3)
verdict   : pour j in 0..32 :
    c2.ids[16+j] == c3.ids[j]
    ET c2.t5[16+j].idx == c3.t5[j].idx  (les 5)
    ET bits(c2.t5[16+j].val) == bits(c3.t5[j].val)  (les 5, @bitCast u32)
  PASS : "KVEQ: 32/32 bit-identiques -> PASS"
  FAIL : "KVEQ: FAIL @gen={d} ref=({d},0x{x}) got=({d},0x{x})" + exit 1
en PASS : écrire $K/dc2_ref.json — {"ids":[32], "marges":[32]} de la référence (2), la marge
  décisionnelle de chaque step CALCULÉE DEPUIS LES Top5 CAPTURÉS + policy (val du rang retenu
  − val du rang non-supprimé suivant — ⚠ l'instrument logué du runner n'existe qu'en mode
  --oracle, `:2355-2364` : ici on la calcule des captures, finding 11). Formatage manuel.
```

⚠ Off-by-one : `gen_top5` reçoit s0 AVANT `generated` (commentaire `:2178-2181`) —
l'alignement ids↔top5 des captures est prouvé par la pré-condition B elle-même (si
l'alignement est faux, le 16/16 échoue déjà : c'est le canari).

- [ ] **Step 6.3 : run DC2 (1280)**

```bash
$B1 $W/model.safetensors $W/tokenizer.json --selftest-kvdump-eq --prompt "$PROMPT" > dc2.out.log 2> dc2.err.log
grep -E "KVEQ:" dc2.err.log
```
Attendu : pré-conditions ok puis `KVEQ: 32/32 bit-identiques -> PASS`.
FAIL ⇒ STOP (C-B est « certain » : un FAIL = bug, pas un tie).

- [ ] **Step 6.4 : tag + commit** — `git tag gate/dc2-pass` ; rapatrier `dc2_ref.json`
  (~2 Ko) → `docs/evidence/kvdump/` ; `dc2.kvdump` (~840 MiB) RESTE sur la VM dans `$K/`
  (chemin unique, consommé par Tasks 7-9, pièce à conviction jusqu'au merge).
  Commit : `kvdump(DC2) : équivalence intra-process bit-exacte 32/32 (selftest orchestré)`.

---

## Task 7 : DC1-Python + DC4 + DC5 — mutants et refus sur dump RÉEL

**Files :**
- Create: `scripts/71_kvdump_inspect.py`

- [ ] **Step 7.1 : le script** — 5 sous-commandes argparse (bas niveau : 8 octets LE +
  header JSON + data ; JAMAIS l'API haut niveau safetensors en écriture, qui réordonne) :

```
inspect <f>                        : clés, metadata, recalcul des 5 xxh64 ; exit 0 ssi tous ok
mutate-flip <f> <out> <t> <off>    : flip d'1 octet dans les DONNÉES du tenseur t, manifest INTACT
make-zeroed <f> <out>              : 4 tenseurs de cache zérotés + xxh64 du manifest RECALCULÉS
                                     (manifest VALIDE : mordant DC4, pas un test de checksum)
set-meta <f> <out> <k> <v>         : réécrit UNE clé metadata (instruments DC5 b/d/f)
mutate-shape <f> <out> <t>         : réécrit le header en changeant la 1ʳᵉ dim du tenseur t
                                     (+1), data intacte — instrument DC5(e) (tranché rév. 3 :
                                     sous-commande ferme, pas de « sed binaire » improvisé)
make-fixture <n> <p0> <out>        : fixture oracle de n ids (tuilage de u9_ids.safetensors)
                                     avec `positions` tel que positions[0] == p0 — ⚠ la garde
                                     `:1767-1770` exige positions[0] == ids.len du run
                                     consommateur ; --oracle lit DEUX clés (`positions`, `fed`,
                                     `:1758-1772`) : reproduire les DEUX, mêmes dtypes que
                                     u9_ids (les inspecter d'abord)
```

- [ ] **Step 7.2 : DC1 volet Python — round-trip + mutant sur dump réel**

```bash
$PY $S71 inspect $K/dc2.kvdump; echo "exit=$?"
$PY $S71 mutate-flip $K/dc2.kvdump /tmp/dc1_mut.kvdump sl_v 12345
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /tmp/dc1_mut.kvdump --max-tokens 1 > dc1m.out.log 2> dc1m.err.log
grep -c "KvDumpChecksumMismatch" dc1m.err.log
```
Attendu : `exit=0` puis `1` (refus VU). `dc1m.err.log` → `docs/evidence/kvdump/`.

- [ ] **Step 7.3 : DC4 — le mordant**

```bash
$PY $S71 make-zeroed $K/dc2.kvdump /tmp/dc4_zero.kvdump
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /tmp/dc4_zero.kvdump --max-tokens 32 --dump-top5 > dc4.out.log 2> dc4.err.log
```
Verdict (Python, contre `dc2_ref.json`) : divergence d'ids dans les **4 premiers** tokens
(prédiction C-E). Continuation IDENTIQUE 32/32 ⇒ **STOP — le dispositif entier est vide**
(spec C-E). Diff → `docs/evidence/kvdump/dc4_divergence.txt`. Tag `gate/dc4-pass`.

- [ ] **Step 7.4 : DC5 — les 7 refus fichier/état, chacun VU**

```bash
# (a) variante : dump 1280 chargé par le binaire 4k
$B4 $W/model.safetensors $W/tokenizer.json --load-cache $K/dc2.kvdump --max-tokens 1        # KvDumpVariantMismatch
# (b) checkpoint
$PY $S71 set-meta $K/dc2.kvdump /tmp/dc5_ckpt.kvdump ckpt_hdr_xxh64 0
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /tmp/dc5_ckpt.kvdump --max-tokens 1 # KvDumpCheckpointMismatch
# (c) troncature
head -c 1000 $K/dc2.kvdump > /tmp/dc5_trunc.kvdump
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /tmp/dc5_trunc.kvdump --max-tokens 1 # KvDumpTruncated
# (d) format
$PY $S71 set-meta $K/dc2.kvdump /tmp/dc5_fmt.kvdump format g12-kvdump-v9
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /tmp/dc5_fmt.kvdump --max-tokens 1  # KvDumpBadFormat
# (e) shape
$PY $S71 mutate-shape $K/dc2.kvdump /tmp/dc5_shape.kvdump sl_k
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /tmp/dc5_shape.kvdump --max-tokens 1 # KvDumpShapeMismatch
# (f) état incohérent
$PY $S71 set-meta $K/dc2.kvdump /tmp/dc5_step.kvdump step_next 7
$B1 $W/model.safetensors $W/tokenizer.json --load-cache /tmp/dc5_step.kvdump --max-tokens 1 # KvDumpInconsistentState
# (g) plus de place
$B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/dc2.kvdump --max-tokens 999999   # SequenceTooLong
```
Chaque sortie archivée → `docs/evidence/kvdump/dc5_<cas>.log`. Tag `gate/dc5-pass`.
Commit : `kvdump(DC1py/DC4/DC5) : script 71, mordant prouvé, 7 refus fichier + 4 refus flags VUS`.

---

## Task 8 : DC3 — inter-process, borné par la bistabilité

**Files :** aucun changement de code.

- [ ] **Step 8.1 : process neuf, restore du dump DC2, continuation libre 32 + top5**

```bash
$B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/dc2.kvdump --max-tokens 32 --dump-top5 > dc3.out.log 2> dc3.err.log
```

- [ ] **Step 8.2 : verdict (Python, contre `dc2_ref.json`)** — règle spec C-C : argmax du 1er
  step identique (marge publiée) ; première divergence éventuelle à marge ≤ 1,873e-3 ⇒ tie
  publié, PASS ; première divergence à marge > 1,873e-3 ⇒ **FAIL, STOP** (signature d'état
  corrompu — DC4 a montré la tête que ça a). `n_match/32` publié à titre informatif.

- [ ] **Step 8.3 : tag `gate/dc3-pass`** (verdict complet, ties inclus, dans le doc Task 10).

---

## Task 9 : DC0 + DC6 + DC7 — graphe, interdit D10, le gain

- [ ] **Step 9.1 : DC0** — re-dump HLO des DEUX variantes (procédure Task 1), md5 == témoins ;
  `git diff main -- zml_runner/engine.zig` vide. Tag `gate/dc0-pass`.

- [ ] **Step 9.2 : DC6 — ⚠ runs de 260 tokens via `--oracle`, fixtures PAR RUN (findings 12
  + N2).** À 32 tokens, `RSS-DELTA` émet `INEXECUTABLE` (`:2492-2494`). La garde
  `positions[0] == ids.len` (`:1767-1770`) impose une fixture PAR géométrie de run :

```bash
# longueur tokenisée du prompt témoin (npt) — mesurée, pas devinée :
$B1 $W/model.safetensors $W/tokenizer.json --prompt "$PROMPT" --ids-only > /tmp/npt.log 2>&1
# npt = le nombre d'ids affiché (lire /tmp/npt.log) ; step_next du dump dc2 :
$PY $S71 inspect $K/dc2.kvdump | grep step_next   # → sn
# fixtures :
$PY $S71 make-fixture 260 <npt> /tmp/fx260a.safetensors   # runs 1-2 (prompt)
$PY $S71 make-fixture 260 <sn>  /tmp/fx260b.safetensors   # run 3 (load : ids.len == step_next)
# les trois runs :
$B1 $W/model.safetensors $W/tokenizer.json --prompt "$PROMPT" --oracle /tmp/fx260a.safetensors > dc6_nu.out.log 2> dc6_nu.err.log
$B1 $W/model.safetensors $W/tokenizer.json --prompt "$PROMPT" --oracle /tmp/fx260a.safetensors --dump-cache /tmp/dc6.kvdump > dc6_dump.out.log 2> dc6_dump.err.log
$B1 $W/model.safetensors $W/tokenizer.json --load-cache $K/dc2.kvdump --oracle /tmp/fx260b.safetensors > dc6_load.out.log 2> dc6_load.err.log
grep -h "ALLOC-LOOP:\|RSS-DELTA:" dc6_nu.err.log dc6_dump.err.log dc6_load.err.log
```
(⚠ vérifier le nom réel du flag « ids seulement » — motif `--ids-only`, early-return
`:919-945` cité par la spec generation-config §4.1 ; s'il a un autre nom, prendre celui du
usage string.)
PASS : les trois `ALLOC-LOOP:` identiques (post-D10 : `alloc=0 resize=0 remap=0 free=0`) ;
les trois `RSS-DELTA` chiffrés (pas INEXECUTABLE) et sous le plafond B10
(`docs/D10_RESULTS.md`). ⚠ Le verdict A1 des runs oracle sortira en mismatch (fixture
factice) : les lignes greppées sont émises AVANT lui — un exit ≠ 0 avec les lignes présentes
est acceptable et noté. Tag `gate/dc6-pass`.

- [ ] **Step 9.3 : DC7 — le gain à 4k (⚠ ~8 min de run + ~2,62 GiB sur /data)**

```bash
$PY $S71 make-fixture 3900 <npt> /tmp/fx3900.safetensors   # npt : mesuré au Step 9.2
$B4 $W/model.safetensors $W/tokenizer.json --prompt "$PROMPT" --oracle /tmp/fx3900.safetensors --dump-cache $K/dc7_4k.kvdump > dc7a.out.log 2> dc7a.err.log
$B4 $W/model.safetensors $W/tokenizer.json --load-cache $K/dc7_4k.kvdump --max-tokens 1 > dc7b.out.log 2> dc7b.err.log
grep -E "PERF :|KVDUMP:" dc7a.err.log ; grep -E "KVLOAD-PERF" dc7b.err.log
```
Publier : temps de calcul du run (i) (ligne `PERF :`, compile exclue — l'A1 mismatch final
est attendu et sans effet, `PERF :` et `KVDUMP:` sont émis avant lui) ; `KVLOAD-PERF:` du
run (ii) (lecture 2,6 GiB INCLUSE). Verdict C-D : gain ≥ ×30 attendu, < ×5 = kill, entre
les deux = publié comme écart. Tag `gate/dc7-pass`. Commit.

- [ ] **Step 9.4 : ménage VM** — supprimer `/tmp/dc*_*.kvdump`, `/tmp/fx*.safetensors` ;
  les dumps de `$K/` (dc2, dc7) : sur GO Régis uniquement, APRÈS merge (pièces à conviction).

---

## Task 10 : doc de résultats, dettes, PLANNING, PR

- [ ] **Step 10.1 : `docs/KVDUMP_RESULTS.md`** — verdicts des 8 gates (tableau), les 5 claims
  jugées contre leurs prédictions §2bis (valeurs mesurées TELLES QUELLES), périmètre de la
  claim (12B, argmax, un fichier par état, pas de multi-tour), dettes : E2B, REPL, sampling
  armé (PRNG non sérialisé), 8k non exercé, compression, reprise-avec-prompt-neuf, clé
  `sampling` informative (un restore avec d'autres warpers diverge légitimement).
- [ ] **Step 10.2 : PLANNING.md** — section chantier + dettes datées.
- [ ] **Step 10.3 : PR `kv-dump-restore` → `main`** — description = résumé du doc de
  résultats. Merge sur GO Régis (`--no-ff`, convention du repo).
- [ ] **Step 10.4 : mémoire** — MAJ `~/dev/Ma_MEMOIRE/memory/project_gemma4_zml_probe.md`
  (section datée) + index MEMORY.md (⚠ plafond 200 l / 25 000 c — MESURER avant d'ajouter).

---

## Récapitulatif des gates → tags

| Gate | Tag | Pièce archivée |
|---|---|---|
| DC0 graphe intact | `gate/dc0-pass` | `hlo_witness.md5` + re-mesure |
| DC1 round-trip + mutants (selftest host + Python réel) | `gate/dc1-pass` | `kvio.err.log`, `dc1m.err.log` |
| DC2 intra-process bit-exact | `gate/dc2-pass` | `dc2_ref.json` |
| DC3 inter-process borné | `gate/dc3-pass` | verdict + ties dans KVDUMP_RESULTS |
| DC4 mordant | `gate/dc4-pass` | `dc4_divergence.txt` |
| DC5 refus bruyants (7 fichier/état + 4 flags) | `gate/dc5-pass` | `dc5_*.log` |
| DC6 interdit D10 (3× ALLOC-LOOP gelés + RSS ≥ 220 tok) | `gate/dc6-pass` | 3 paires de lignes |
| DC7 gain (oracle 3900, KVLOAD-PERF lecture incluse) | `gate/dc7-pass` | `PERF :` + `KVLOAD-PERF:` |

**Ordre et dépendances** : 0 → 1 → 2 (DC1-host) → 3 → 4 → 5 → 6 (produit `dc2.kvdump` +
`dc2_ref.json`, consommés par 7, 8, 9.2, 9.3) → 7 → 8 → 9 → 10. Aucune tâche ne se lance si
la précédente a un gate FAIL non instruit.
