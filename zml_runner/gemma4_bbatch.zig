// Runner AUTONOME BATCHÉ (batch statique B lanes) — spec docs/superpowers/specs/
// 2026-07-12-batching-flash-attn-design.md §3.2, plan Tasks 2/3/4.
// Généralisation par lane de `gemma4_gen_auto.zig` (L3 in-graph, token→token) : le graphe compilé
// `BBStep.forward` est OP-POUR-OP celui de `StepTok` (gather embed_tokens + gather eptl +
// Model.forwardStep + topK) — SEULES les shapes d'entrée changent (tok {B,1}, cache {slot,B,1,…}).
// Le moteur `engine.zig` est INTACT (shape-polymorphe depuis le gate T0 : les 5 sites dérivent
// dim(.b)/dim(.s) des entrées) — B est un paramètre RUNTIME, jamais comptime : un binaire unique
// sert tout le sweep (doctrine de custody G2.3 §7.1, sha256 constant).
//
// CONTRAINTE V1 (spec §2.2, verrou du moteur) : `ctrl.step` est un scalaire UNIQUE (pickStep tire
// UNE ligne des tables Packed pour tout le batch) et `pos_u` est l'index `.k` scalaire du
// scatterSlices du cache pour TOUTES les lanes → les positions sont partagées ⇒ tous les prompts
// DOIVENT avoir la même longueur tokenisée (sinon error.PromptLengthMismatch). Longueurs
// hétérogènes (positions [b], masques par lane, padding) = hors périmètre.
//
// INVARIANT VRAM (spec §2.8) : les tables Packed (masques, cos/sin, positions) restent à b=1 —
// JAMAIS étendues ×B. Le `mask.broad(scores.shape())` d'engine.zig est rank-égal : ZML broadcaste
// alors par POSITIONS (tensor.zig:2183-2195), ce qui diffuse correctement b:1→B ici parce que
// l'ordre des axes coïncide. Invariant fragile → PROUVÉ par le sous-test 4 de --selftest-batch,
// pas supposé.
//
// CLI : gemma4_bbatch <model.safetensors> <tokenizer.json> --prompts <fichier> [--oracles f1,f2,…]
//       [--replicate N] [--max-tokens N] [--ids-only] [--selftest-batch <fixture>]
//       [--force-vram] [--allow-cpu] [--no-prealloc]
// ⚠ GPU : builder/lancer avec `--@zml//platforms:cuda=true`, sinon repli CPU SILENCIEUX.
const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const SLIDING_WINDOW: i64 = 512;
const HD_F: i64 = 512; // dim cos/sin full (= config.global_head_dim)
const HD_S: i64 = 256; // dim cache sliding (= engine.HD_SLIDING)
const D: i64 = 1536;
const LF: i64 = 8960;
// Slots de cache producteurs (engine.zig: isFull(i)=(i+1)%5==0, FIRST_KV_SHARED=15) : parmi les 15
// premières couches, 3 sont "full" (4,9,14) et 12 "sliding" — mêmes comptes que
// SLIDING_PRODUCERS/FULL_PRODUCERS de scripts/49_gen_custom_oracle.py:41-42.
const NUM_SLIDING_SLOTS: usize = 12;
const NUM_FULL_SLOTS: usize = 3;
// Variante d'attention : COMPTIME (elle change le graphe). Un second main (`gemma4_bbs.zig`)
// déclare `pub const ATTN = .sdpa` et réutilise TOUT ce fichier — pattern e1/e2 du repo, sans
// dupliquer le runner. Root == ce fichier → `.manual` (le défaut neutre du gate S1).
const ATTN: engine.AttnKind = if (@hasDecl(@import("root"), "ATTN")) @import("root").ATTN else .manual;
const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX, .attn = ATTN });
const PackedLong = engine.Packed(true);

// K du topK in-graph — CONSIGNÉ dans les logs (confound pré-enregistré du gate B4 : gen_auto est
// aussi à K=5, donc les runs appariés B=1 sont comparables sans re-run K, spec §3.2).
const K_TOPK: u32 = 5;

// BOS (id 2) : PRÉFIXÉ explicitement — l'encoder ZML (iree) n'ajoute AUCUN token spécial.
const BOS_ID: u32 = 2;

// Chat template Gemma — VÉRITÉ = repr() HF mesuré : '<bos><|turn>user\nPROMPT<turn|>\n<|turn>model\n'
// ⚠ tokens de tour : <|turn> (105) / <turn|> (106) — PAS <start_of_turn>/<end_of_turn>.
fn renderChatTemplate(allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "<|turn>user\n{s}<turn|>\n<|turn>model\n", .{prompt});
}

const Args = struct {
    ckpt: []const u8,
    tokjson_path: []const u8,
    prompts_path: ?[]const u8 = null,
    oracles: ?[][]const u8 = null,
    replicate: usize = 1,
    max_tokens: ?usize = null,
    ids_only: bool = false,
    allow_cpu: bool = false,
    force_vram: bool = false,
    no_prealloc: bool = false,
    selftest_batch: ?[]const u8 = null,
};

const usage =
    "Usage: gemma4_bbatch <model.safetensors> <tokenizer.json> --prompts <fichier> " ++
    "[--oracles f1,f2,...] [--replicate N] [--max-tokens N] [--ids-only] " ++
    "[--selftest-batch <fixture>] [--force-vram] [--allow-cpu (débogage)] [--no-prealloc]\n" ++
    "  --prompts : un prompt par ligne (lignes vides ignorées) ; B = nb de lignes × --replicate.\n" ++
    "  --oracles : fixtures 49 APPARIÉES PAR INDEX aux lignes de --prompts (autant que de lignes).\n" ++
    "  --ids-only : tokenise et rapporte les longueurs par lane (outil de constitution du jeu).";

// Parsing à la main (motif gen_auto.zig:90-150). Type EXACT du retour de std.process.Args.toSlice :
// slice d'éléments sentinelle-terminés — chaque élément coerce vers []const u8, la slice entière NON.
fn parseArgs(gpa: std.mem.Allocator, process_args: []const [:0]const u8) !Args {
    if (process_args.len < 3) {
        log.err("{s}", .{usage});
        return error.MissingArgument;
    }
    var args: Args = .{ .ckpt = process_args[1], .tokjson_path = process_args[2] };

    var i: usize = 3;
    while (i < process_args.len) : (i += 1) {
        const a = process_args[i];
        if (std.mem.eql(u8, a, "--prompts")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--prompts attend une valeur", .{});
                return error.MissingArgument;
            }
            args.prompts_path = process_args[i];
        } else if (std.mem.eql(u8, a, "--oracles")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--oracles attend une valeur (liste séparée par des virgules)", .{});
                return error.MissingArgument;
            }
            var list: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, process_args[i], ',');
            while (it.next()) |f| {
                const t = std.mem.trim(u8, f, " \t");
                if (t.len == 0) continue;
                try list.append(gpa, t);
            }
            if (list.items.len == 0) {
                log.err("--oracles : liste vide", .{});
                return error.MissingArgument;
            }
            args.oracles = list.items;
        } else if (std.mem.eql(u8, a, "--replicate")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--replicate attend une valeur", .{});
                return error.MissingArgument;
            }
            args.replicate = std.fmt.parseInt(usize, process_args[i], 10) catch |err| {
                log.err("--replicate: valeur invalide '{s}' ({s})", .{ process_args[i], @errorName(err) });
                return err;
            };
            if (args.replicate == 0) {
                log.err("--replicate doit être >= 1", .{});
                return error.InvalidArgument;
            }
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--max-tokens attend une valeur", .{});
                return error.MissingArgument;
            }
            args.max_tokens = std.fmt.parseInt(usize, process_args[i], 10) catch |err| {
                log.err("--max-tokens: valeur invalide '{s}' ({s})", .{ process_args[i], @errorName(err) });
                return err;
            };
        } else if (std.mem.eql(u8, a, "--selftest-batch")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-batch attend une valeur (fixture)", .{});
                return error.MissingArgument;
            }
            args.selftest_batch = process_args[i];
        } else if (std.mem.eql(u8, a, "--ids-only")) {
            args.ids_only = true;
        } else if (std.mem.eql(u8, a, "--allow-cpu")) {
            args.allow_cpu = true;
        } else if (std.mem.eql(u8, a, "--force-vram")) {
            args.force_vram = true;
        } else if (std.mem.eql(u8, a, "--no-prealloc")) {
            args.no_prealloc = true;
        } else {
            log.err("argument inconnu: {s}\n{s}", .{ a, usage });
            return error.InvalidArgument;
        }
    }
    return args;
}

// Lecture du fichier de prompts : une ligne = un prompt. Les lignes VIDES sont ignorées ; AUCUN
// support de commentaire (un '#' resterait un prompt) — délibéré : l'appariement `--oracles` se
// fait PAR INDEX sur ces lignes, tout filtrage supplémentaire serait une source de désappariement
// silencieux. Les slices retournées pointent dans `buf` (à garder vivant).
fn readPromptLines(allocator: std.mem.Allocator, io: std.Io, path: []const u8, buf_out: *[]u8) ![][]const u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    const len: usize = @intCast(try file.length(io));
    const buf = try allocator.alloc(u8, len);
    errdefer allocator.free(buf);
    const n = try file.readPositionalAll(io, buf, 0);
    if (n != len) {
        log.err("--prompts : lecture courte {d}/{d} octets ({s})", .{ n, len, path });
        return error.ShortRead;
    }
    buf_out.* = buf;

    var lines: std.ArrayList([]const u8) = .empty;
    errdefer lines.deinit(allocator);
    var it = std.mem.splitScalar(u8, buf, '\n');
    while (it.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        if (t.len == 0) continue;
        try lines.append(allocator, t);
    }
    if (lines.items.len == 0) {
        log.err("--prompts : aucun prompt non vide dans {s}", .{path});
        return error.EmptyPromptFile;
    }
    // toOwnedSlice (pas `.items`) : l'appelant `free()` la slice — elle doit avoir EXACTEMENT la
    // taille de l'allocation (une ArrayList a de la capacité en trop).
    return lines.toOwnedSlice(allocator);
}

// ============================================================================================
// Inputs host (copie de gen_auto.zig:152-327) : cos/sin RoPE full, masques additifs, positions,
// tables {L_MAX,…}. SEULE différence : les caches zéro sont dimensionnés ×B.
// ============================================================================================

// Masques additifs f32 : 0 = visible, -floatMax = masqué (== torch.finfo(float32).min).
const MASK_MIN: f32 = -std.math.floatMax(f32);

// RoPE "proportional" (couches full) — formule COPIÉE de HF _compute_proportional_rope_parameters :
// head_dim=512, base=1e6, rope_proportion=0.25 → rope_angles=64, nope_angles=192 ; emb = concat
// (freqs, freqs) — DUPLICATION de la moitié, pas d'entrelacement ; attention_scaling=1.0 (no-op).
const ROPE_FULL_THETA: f32 = 1_000_000.0;
const ROPE_FULL_HEAD_DIM: f32 = 512.0;
const ROPE_FULL_ANGLES: usize = 64;
const ROPE_FULL_HALF: usize = 256;

fn ropeFull(p: i64, cos_out: *[HD_F]f32, sin_out: *[HD_F]f32) void {
    var inv_freq: [ROPE_FULL_HALF]f32 = undefined;
    for (0..ROPE_FULL_HALF) |i| {
        if (i < ROPE_FULL_ANGLES) {
            const exp: f32 = @as(f32, @floatFromInt(2 * i)) / ROPE_FULL_HEAD_DIM;
            inv_freq[i] = 1.0 / std.math.pow(f32, ROPE_FULL_THETA, exp);
        } else {
            inv_freq[i] = 0.0; // nope_angles : pas de rotation
        }
    }
    const pf: f32 = @floatFromInt(p);
    for (0..ROPE_FULL_HALF) |i| {
        const angle: f32 = inv_freq[i] * pf;
        const c: f32 = @cos(angle);
        const s: f32 = @sin(angle);
        cos_out[i] = c;
        cos_out[i + ROPE_FULL_HALF] = c;
        sin_out[i] = s;
        sin_out[i + ROPE_FULL_HALF] = s;
    }
}

fn maskRows(p: i64, sliding_out: []f32, full_out: []f32) void {
    const lo = @max(0, p - (SLIDING_WINDOW - 1));
    for (0..@intCast(L_MAX)) |j| {
        const ji: i64 = @intCast(j);
        sliding_out[j] = if (ji > p or ji < lo) MASK_MIN else 0;
        full_out[j] = if (ji > p) MASK_MIN else 0;
    }
}

// Tables host indexées par STEP == POSITION ABSOLUE p (identité : `ctrl.step` vaut la position).
// Les tables cos/sin/masques/positions restent à b=1 (INVARIANT VRAM, cf tête de fichier) ; SEULS
// les caches sont dimensionnés ×B (coût marginal ≈ 38 Mo f32 par lane à kmax=1024).
const HostInputs = struct {
    cos_full: []f32, // {L_MAX, HD_F}
    sin_full: []f32,
    masks_sliding: []f32, // {L_MAX, L_MAX}
    masks_full: []f32,
    positions: []i32, // {L_MAX}
    embeds_zero: []u8, // {L_MAX,1,1,D} bf16 — factice (jamais lu par forwardStep)
    embptls_zero: []u8, // {L_MAX,1,1,LF} bf16 — idem
    cache_sl_k: []u8, // {NUM_SLIDING_SLOTS, B, 1, L_MAX, HD_S} f32, zéros
    cache_sl_v: []u8,
    cache_fl_k: []u8, // {NUM_FULL_SLOTS, B, 1, L_MAX, HD_F} f32, zéros
    cache_fl_v: []u8,

    fn init(allocator: std.mem.Allocator, b: usize) !HostInputs {
        const l_max: usize = @intCast(L_MAX);
        const hd_f: usize = @intCast(HD_F);
        const hd_s: usize = @intCast(HD_S);

        const cos_full = try allocator.alloc(f32, l_max * hd_f);
        errdefer allocator.free(cos_full);
        const sin_full = try allocator.alloc(f32, l_max * hd_f);
        errdefer allocator.free(sin_full);
        const masks_sliding = try allocator.alloc(f32, l_max * l_max);
        errdefer allocator.free(masks_sliding);
        const masks_full = try allocator.alloc(f32, l_max * l_max);
        errdefer allocator.free(masks_full);
        const positions = try allocator.alloc(i32, l_max);
        errdefer allocator.free(positions);

        var p: i64 = 0;
        while (p < L_MAX) : (p += 1) {
            const idx: usize = @intCast(p);
            positions[idx] = @intCast(p);
            var cos_row: [HD_F]f32 = undefined;
            var sin_row: [HD_F]f32 = undefined;
            ropeFull(p, &cos_row, &sin_row);
            @memcpy(cos_full[idx * hd_f .. (idx + 1) * hd_f], &cos_row);
            @memcpy(sin_full[idx * hd_f .. (idx + 1) * hd_f], &sin_row);
            maskRows(p, masks_sliding[idx * l_max .. (idx + 1) * l_max], masks_full[idx * l_max .. (idx + 1) * l_max]);
        }

        const embeds_zero = try allocator.alloc(u8, l_max * @as(usize, @intCast(D)) * 2);
        errdefer allocator.free(embeds_zero);
        @memset(embeds_zero, 0);
        const embptls_zero = try allocator.alloc(u8, l_max * @as(usize, @intCast(LF)) * 2);
        errdefer allocator.free(embptls_zero);
        @memset(embptls_zero, 0);
        const cache_sl_k = try allocator.alloc(u8, NUM_SLIDING_SLOTS * b * l_max * hd_s * 4);
        errdefer allocator.free(cache_sl_k);
        @memset(cache_sl_k, 0);
        const cache_sl_v = try allocator.alloc(u8, NUM_SLIDING_SLOTS * b * l_max * hd_s * 4);
        errdefer allocator.free(cache_sl_v);
        @memset(cache_sl_v, 0);
        const cache_fl_k = try allocator.alloc(u8, NUM_FULL_SLOTS * b * l_max * hd_f * 4);
        errdefer allocator.free(cache_fl_k);
        @memset(cache_fl_k, 0);
        const cache_fl_v = try allocator.alloc(u8, NUM_FULL_SLOTS * b * l_max * hd_f * 4);
        errdefer allocator.free(cache_fl_v);
        @memset(cache_fl_v, 0);

        return .{
            .cos_full = cos_full,
            .sin_full = sin_full,
            .masks_sliding = masks_sliding,
            .masks_full = masks_full,
            .positions = positions,
            .embeds_zero = embeds_zero,
            .embptls_zero = embptls_zero,
            .cache_sl_k = cache_sl_k,
            .cache_sl_v = cache_sl_v,
            .cache_fl_k = cache_fl_k,
            .cache_fl_v = cache_fl_v,
        };
    }

    fn deinit(self: *HostInputs, allocator: std.mem.Allocator) void {
        allocator.free(self.cos_full);
        allocator.free(self.sin_full);
        allocator.free(self.masks_sliding);
        allocator.free(self.masks_full);
        allocator.free(self.positions);
        allocator.free(self.embeds_zero);
        allocator.free(self.embptls_zero);
        allocator.free(self.cache_sl_k);
        allocator.free(self.cache_sl_v);
        allocator.free(self.cache_fl_k);
        allocator.free(self.cache_fl_v);
    }
};

// Lit un tenseur ENTIER d'une fixture, host-side, SANS Platform (copie gen_auto.zig:334-353) :
// dtype du header vérifié AVANT lecture, compte d'octets vérifié APRÈS (fichier tronqué →
// error.ShortRead, pas des zéros silencieux).
fn readFixtureAlloc(comptime T: type, comptime want_dtype: zml.DataType, allocator: std.mem.Allocator, io: std.Io, reg: *const zml.safetensors.TensorRegistry, file: *std.Io.File, name: []const u8) ![]T {
    const t = reg.tensors.get(name) orelse {
        log.err("tensor introuvable dans la fixture: {s}", .{name});
        return error.MissingTensor;
    };
    const dt = t.shape.dtype();
    if (dt != want_dtype) {
        log.err("{s}: dtype fixture = {s} ≠ attendu = {s}", .{ name, @tagName(dt), @tagName(want_dtype) });
        return error.DtypeMismatch;
    }
    const size: usize = @intCast(t.byteSize());
    const out = try allocator.alloc(T, size / @sizeOf(T));
    errdefer allocator.free(out);
    const got = try file.readPositionalAll(io, std.mem.sliceAsBytes(out), t.offset);
    if (got != size) {
        log.err("{s}: lecture courte — {d}/{d} octets (fixture tronquée ?)", .{ name, got, size });
        return error.ShortRead;
    }
    return out;
}

// Oracle d'une lane : fixture 49 (fed = séquence de référence) + son manifest sidecar JSON (SEUL
// porteur des `prompt_ids` — la fixture safetensors ne les contient pas). Les deux sont vérifiés
// contre le prompt tokenisé de la ligne appariée (garde anti faux-PASS sur fixture désappariée).
const Oracle = struct {
    fed: []i32,
    prompt_ids: []u32,
    seq_len: i32, // positions[0] de la fixture

    fn deinit(self: *Oracle, allocator: std.mem.Allocator) void {
        allocator.free(self.fed);
        allocator.free(self.prompt_ids);
    }
};

fn loadOracle(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Oracle {
    var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
    defer reg.deinit();
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    const positions_fx = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "positions");
    defer allocator.free(positions_fx);
    if (positions_fx.len == 0) {
        log.err("--oracles : fixture 'positions' vide ({s})", .{path});
        return error.EmptyFixture;
    }
    const fed_fx = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "fed");
    errdefer allocator.free(fed_fx);
    if (fed_fx.len == 0) {
        log.err("--oracles : fixture 'fed' vide — un PASS à 0 step serait vacueux ({s})", .{path});
        return error.EmptyFixture;
    }

    // Manifest sidecar OBLIGATOIRE : `<fixture>.manifest.json` (écrit par scripts/49, clé
    // "prompt_ids"). Son absence est une ERREUR, pas un repli silencieux : sans lui, une fixture
    // appariée à la MAUVAISE ligne mais de même longueur passerait la garde positions[0].
    const manifest_path = try std.fmt.allocPrint(allocator, "{s}.manifest.json", .{path});
    defer allocator.free(manifest_path);
    var mf = std.Io.Dir.cwd().openFile(io, manifest_path, .{ .mode = .read_only }) catch |err| {
        log.err("--oracles : manifest sidecar illisible ({s}) : {s} — requis (porte les prompt_ids)", .{ manifest_path, @errorName(err) });
        return error.MissingManifest;
    };
    defer mf.close(io);
    const mlen: usize = @intCast(try mf.length(io));
    const mtext = try allocator.alloc(u8, mlen);
    defer allocator.free(mtext);
    const mread = try mf.readPositionalAll(io, mtext, 0);
    if (mread != mlen) return error.ShortRead;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, mtext, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const root = parsed.value.object;
    const arr = (root.get("prompt_ids") orelse {
        log.err("--oracles : clé 'prompt_ids' absente de {s}", .{manifest_path});
        return error.MissingManifest;
    }).array;
    const pids = try allocator.alloc(u32, arr.items.len);
    errdefer allocator.free(pids);
    for (arr.items, 0..) |v, i| pids[i] = @intCast(v.integer);

    return .{ .fed = fed_fx, .prompt_ids = pids, .seq_len = positions_fx[0] };
}

// ============================================================================================
// Garde GPU — variante bbatch (spec §3.2) : PAS de seuil VRAM fixe (le plafond est un OUTPUT du
// banc, pas un input : la garde 20 GiB de gen_auto est calibrée B=1 et reste intouchée). bbatch
// refuse seulement s'il y a CONTENTION : d'autres process compute listés par nvidia-smi (piège
// opérationnel n°1 : Ollama ~22 Go). Logique nvidia-smi DUPLIQUÉE depuis checkVram
// (gen_auto.zig:683-736), PAS extraite en module partagé — gen_auto reste intact d'un octet.
// Best-effort : nvidia-smi absent/cassé → warn + continue (l'OOM reste le filet).
// ============================================================================================
fn parseFreeMiB(stdout: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, stdout, '\n');
    const first = lines.next() orelse return null;
    const trimmed = std.mem.trim(u8, first, " \t\r");
    if (trimmed.len == 0) return null;
    return std.fmt.parseInt(u64, trimmed, 10) catch null;
}

fn checkContention(gpa: std.mem.Allocator, io: std.Io) !void {
    const apps = std.process.run(gpa, io, .{
        .argv = &.{ "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader" },
    }) catch |err| {
        log.warn("garde de contention sautée : nvidia-smi indisponible ({s}) — machine sans GPU ?", .{@errorName(err)});
        return;
    };
    defer gpa.free(apps.stdout);
    defer gpa.free(apps.stderr);
    switch (apps.term) {
        .exited => |code| if (code != 0) {
            log.warn("garde de contention sautée : nvidia-smi exit={d}", .{code});
            return;
        },
        else => {
            log.warn("garde de contention sautée : nvidia-smi terminé anormalement", .{});
            return;
        },
    }

    var busy = false;
    var it = std.mem.splitScalar(u8, apps.stdout, '\n');
    while (it.next()) |line| {
        const l = std.mem.trim(u8, line, " \t\r");
        if (l.len == 0) continue;
        if (!busy) log.err("GPU occupé — process compute déjà présents (mesure VRAM/perf invalide) :", .{});
        log.err("  {s}", .{l});
        busy = true;
    }

    // VRAM libre : INFORMATIF seulement (aucun seuil — le plafond B est ce que le banc mesure).
    if (std.process.run(gpa, io, .{
        .argv = &.{ "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits" },
    })) |res| {
        defer gpa.free(res.stdout);
        defer gpa.free(res.stderr);
        if (parseFreeMiB(res.stdout)) |free_mib| {
            const gib10 = free_mib * 10 / 1024;
            log.info("VRAM libre au lancement : {d}.{d} GiB ({d} MiB) — aucun seuil appliqué (plafond = output du banc)", .{ gib10 / 10, gib10 % 10, free_mib });
        }
    } else |err| {
        log.warn("VRAM libre indisponible ({s})", .{@errorName(err)});
    }

    if (busy) {
        log.err("Libérer d'abord : `ollama ps` puis `ollama stop <modèle>` (réversible), ou --force-vram pour tenter quand même", .{});
        return error.GpuBusy;
    }
}

// Init plateforme (motif gen_auto.zig:901-916 + --no-prealloc de g23_sweep.zig:196). `--no-prealloc`
// coupe la préallocation BFC : nvidia-smi mesure alors la VRAM RÉELLEMENT utilisée et non la réserve
// (0.90 × libre) — INDISPENSABLE au relevé du pic VRAM par point de sweep (méthode G3).
fn initPlatform(allocator: std.mem.Allocator, io: std.Io, no_prealloc: bool, allow_cpu: bool) !*zml.Platform {
    const cuda_opts: zml.platform.CreateOptions = .{ .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = !no_prealloc, .memory_fraction = 0.90 } } } };
    const platform: *zml.Platform = blk: {
        if (zml.Platform.init(allocator, io, .cuda, cuda_opts)) |p| break :blk p else |_| {}
        log.warn("CUDA indisponible (libpjrt_cuda absent ?) — repli sur Platform.auto (probablement CPU).", .{});
        break :blk try zml.Platform.auto(allocator, io, .{});
    };
    errdefer platform.deinit(allocator);
    log.info("backend = {s} (cible : cuda) ; preallocate={}", .{ @tagName(platform.target), !no_prealloc });
    // Garde CUDA DURE (incident du 10 juil : binaire buildé sans `--@zml//platforms:cuda=true` →
    // libpjrt_cuda absent des runfiles → repli CPU SILENCIEUX, run qui rampe des heures).
    // ⚠ --allow-cpu ne FORCE pas le CPU : l'init .cuda est tentée d'abord, le flag ne fait que
    // tolérer le repli.
    if (platform.target != .cuda and !allow_cpu) {
        log.err("backend = {s} ≠ cuda — repli CPU refusé (rebuilder/lancer avec --@zml//platforms:cuda=true, ou --allow-cpu pour du débogage)", .{@tagName(platform.target)});
        return error.CudaRequired;
    }
    return platform;
}

// ============================================================================================
// Gate B1 — `--selftest-batch <fixture>` : spike des PRIMITIVES batchées (B=2), valeurs exactes vs
// référence host. Mini-graphes séparés (motif SgFwd, gen_auto.zig:495-516) — pas besoin du modèle
// complet. Ce mode est DISPATCHÉ AVANT l'exigence de `--prompts` (contrairement au
// `--selftest-gather` de gen_auto qui, lui, requiert un --prompt factice) : le selftest est
// autonome. Il PASSE par la garde de contention + la garde CUDA dure (c'est du travail GPU).
// Noms de structs COURTS (piège quota comptime @typeName sur pjrt structSize).
// ============================================================================================
const EMB_KEY = "model.language_model.embed_tokens.weight"; // clés ABSOLUES (root view, pas de withPrefix)
const EPTL_KEY = "model.language_model.embed_tokens_per_layer.weight";

const BbTabs = struct {
    emb: zml.Tensor, // {voc,d} bf16 BRUT
    eptl: zml.Tensor, // {voc,lf} bf16 BRUT

    fn init(base: zml.io.TensorStore.View) BbTabs {
        return .{
            .emb = base.createTensor(EMB_KEY, .{ .voc, .d }, null),
            .eptl = base.createTensor(EPTL_KEY, .{ .voc, .lf }, null),
        };
    }
    fn load(self: *const BbTabs, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(BbTabs) {
        return zml.io.load(BbTabs, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

// Sous-test 1 — gather batché : MÊME code que `BBStep` (GatherOpts `.{}` OBLIGATOIRE en 2e arg).
// Compilé DEUX FOIS (tok {2,1} et tok {1,1}) : la référence est le gather 1-lane, pas une constante.
const BbGat = struct {
    pub fn forward(emb: zml.Tensor, eptl: zml.Tensor, tok: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        return .{ emb.gather(.{ .voc = tok }, .{}), eptl.gather(.{ .voc = tok }, .{}) };
    }
};

// Sous-test 2 — scatterSlices batché : LE point jamais exercé dans ce repo (spec §7). Mêmes opts
// que runLayerGen (engine.zig:404) : slot scalaire + `.k = pos` scalaire PARTAGÉ par les lanes ;
// l'update porte `.b=2` avec des valeurs DISTINCTES par lane → chaque lane doit recevoir SA ligne.
const BbSct = struct {
    pub fn forward(cache: zml.Tensor, upd: zml.Tensor, pos: zml.Tensor) zml.Tensor {
        const so = zml.Tensor.ScatterOpts{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override };
        const slot = zml.Tensor.scalar(@as(u32, 0), .u32);
        return cache.scatterSlices(.{ .slot = slot, .k = pos }, upd, so);
    }
};

// Sous-test 3 — topK batché : forme struct `.{ .voc = .voc }` EXIGÉE (un enum literal seul ne matche
// ni la branche .int ni la branche .struct de topK, tensor.zig:3098). Vérifie le layout D2H {b,K}.
const BbTop = struct {
    pub fn forward(logits: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const t5 = logits.topK(.{ .voc = .voc }, K_TOPK, .{});
        return .{ t5.values, t5.indices };
    }
};

// Sous-test 4 — broad du masque RANK-ÉGAL (spec §2.8) : à rank égal, ZML broadcaste par POSITIONS
// et non par tags (tensor.zig:2183-2195). C'est EXACTEMENT le `scores.add(mask.broad(scores.shape()))`
// d'engine.zig:465 : masque à b=1, scores à b=B. Ce test PROUVE que les deux lanes reçoivent le
// masque au lieu de le supposer (invariant fragile : il ne tient que parce que l'ordre des axes coïncide).
const BbBrd = struct {
    pub fn forward(scores: zml.Tensor, mask: zml.Tensor) zml.Tensor {
        return scores.add(mask.broad(scores.shape()));
    }
};

fn selftestBatch(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, ckpt_path: []const u8, fixture_path: []const u8) !void {
    log.info("B1 — selftest des primitives batchées (B=2) : gather / scatterSlices / topK / broad", .{});

    // ---- Fixture d'abord (host-only, fail-fast avant tout travail GPU) : on n'en tire que 2 ids
    // `fed` distincts + leurs lignes d'embeddings de référence (bit-exact, motif SG).
    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg_fx.deinit();
    var file_fx = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
    defer file_fx.close(io);
    const fed_fx = try readFixtureAlloc(i32, .i32, allocator, io, &reg_fx, &file_fx, "fed");
    defer allocator.free(fed_fx);
    const embeds_fx = try readFixtureAlloc(u16, .bf16, allocator, io, &reg_fx, &file_fx, "embeds");
    defer allocator.free(embeds_fx);
    const embptls_fx = try readFixtureAlloc(u16, .bf16, allocator, io, &reg_fx, &file_fx, "embptls");
    defer allocator.free(embptls_fx);

    const d_u: usize = @intCast(D);
    const lf_u: usize = @intCast(LF);
    if (fed_fx.len < 2) {
        log.err("B1 : la fixture doit fournir >= 2 tokens `fed` (B=2), got {d}", .{fed_fx.len});
        return error.EmptyFixture;
    }
    if (embeds_fx.len < 2 * d_u or embptls_fx.len < 2 * lf_u) {
        log.err("B1 : shape fixture inattendue (embeds.len={d}, embptls.len={d})", .{ embeds_fx.len, embptls_fx.len });
        return error.UnexpectedShape;
    }

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt_path);
    defer reg_ck.deinit();
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    defer store_ck.deinit();
    const tabs: BbTabs = .init(store_ck.view());
    const tabs_buf = try tabs.load(allocator, io, platform, &store_ck, &.{sharding});

    // ======================= Sous-test 1 : gather {2,1} vs gather 1-lane =======================
    const tok2_sym = zml.Tensor.init(.{ 2, 1 }, .u32).withTags(.{ .b, .s });
    const tok1_sym = zml.Tensor.init(.{ 1, 1 }, .u32).withTags(.{ .b, .s });
    var exe_g2 = try platform.compileFn(allocator, io, BbGat.forward, .{ tabs.emb, tabs.eptl, tok2_sym }, .{ .shardings = &.{sharding} });
    defer exe_g2.deinit();
    var exe_g1 = try platform.compileFn(allocator, io, BbGat.forward, .{ tabs.emb, tabs.eptl, tok1_sym }, .{ .shardings = &.{sharding} });
    defer exe_g1.deinit();

    // Bits du token PRÉSERVÉS (@bitCast, pas @intCast) : un `fed` négatif d'une fixture corrompue ne
    // doit pas déclencher un piège de cast — le gather (ou le mismatch) qualifiera l'anomalie.
    var tok2_host = [2]u32{ @bitCast(fed_fx[0]), @bitCast(fed_fx[1]) };
    var tok2_buf = try zml.Buffer.fromBytes(io, platform, tok2_sym.shape(), sharding, std.mem.sliceAsBytes(&tok2_host));
    defer tok2_buf.deinit();

    var a_g2 = try exe_g2.args(allocator);
    defer a_g2.deinit(allocator);
    var r_g2 = try exe_g2.results(allocator);
    defer r_g2.deinit(allocator);
    a_g2.set(.{ tabs_buf.emb, tabs_buf.eptl, tok2_buf });
    exe_g2.call(a_g2, &r_g2);
    var g2_emb, var g2_eptl = r_g2.get(struct { zml.Buffer, zml.Buffer });
    defer g2_emb.deinit();
    defer g2_eptl.deinit();

    var g2_emb_s = try g2_emb.toSliceAlloc(allocator, io);
    defer g2_emb_s.free(allocator);
    var g2_eptl_s = try g2_eptl.toSliceAlloc(allocator, io);
    defer g2_eptl_s.free(allocator);
    const g2_emb_bits = g2_emb_s.items(u16);
    const g2_eptl_bits = g2_eptl_s.items(u16);
    // Garde longueur = shape ET dtype d'un coup : un gather upcasté bf16→f32 doublerait len et
    // produirait un « mismatch » trompeur au lieu d'une erreur qualifiée.
    if (g2_emb_bits.len != 2 * d_u or g2_eptl_bits.len != 2 * lf_u) {
        log.err("B1/gather : longueurs D2H inattendues (emb={d}≠{d}, eptl={d}≠{d}) — dtype/shape du gather a dérivé ?", .{ g2_emb_bits.len, 2 * d_u, g2_eptl_bits.len, 2 * lf_u });
        return error.UnexpectedShape;
    }

    for (0..2) |lane| {
        var tok1_host = [1]u32{@bitCast(fed_fx[lane])};
        var tok1_buf = try zml.Buffer.fromBytes(io, platform, tok1_sym.shape(), sharding, std.mem.sliceAsBytes(&tok1_host));
        defer tok1_buf.deinit();
        var a_g1 = try exe_g1.args(allocator);
        defer a_g1.deinit(allocator);
        var r_g1 = try exe_g1.results(allocator);
        defer r_g1.deinit(allocator);
        a_g1.set(.{ tabs_buf.emb, tabs_buf.eptl, tok1_buf });
        exe_g1.call(a_g1, &r_g1);
        var g1_emb, var g1_eptl = r_g1.get(struct { zml.Buffer, zml.Buffer });
        defer g1_emb.deinit();
        defer g1_eptl.deinit();
        var g1_emb_s = try g1_emb.toSliceAlloc(allocator, io);
        defer g1_emb_s.free(allocator);
        var g1_eptl_s = try g1_eptl.toSliceAlloc(allocator, io);
        defer g1_eptl_s.free(allocator);
        const ref_emb = g1_emb_s.items(u16);
        const ref_eptl = g1_eptl_s.items(u16);
        if (ref_emb.len != d_u or ref_eptl.len != lf_u) {
            log.err("B1/gather : longueurs D2H 1-lane inattendues (emb={d}, eptl={d})", .{ ref_emb.len, ref_eptl.len });
            return error.UnexpectedShape;
        }
        const lane_emb = g2_emb_bits[lane * d_u .. (lane + 1) * d_u];
        const lane_eptl = g2_eptl_bits[lane * lf_u .. (lane + 1) * lf_u];
        const fx_emb = embeds_fx[lane * d_u .. (lane + 1) * d_u];
        const fx_eptl = embptls_fx[lane * lf_u .. (lane + 1) * lf_u];
        // BIT-EXACT sur les u16 bruts (bf16 = 2 octets), AUCUNE tolérance : c'est le garde-fou
        // contre un scaling/upcast accidenté. Double référence : le gather 1-lane (primaire) ET la
        // ligne de la fixture HF (croisée) — les deux doivent coïncider à la lane près.
        for (0..d_u) |i| {
            if (lane_emb[i] != ref_emb[i]) {
                log.err("B1/gather FAIL (emb) lane={d} tok={d} idx={d} : batché=0x{x} 1-lane=0x{x}", .{ lane, fed_fx[lane], i, lane_emb[i], ref_emb[i] });
                return error.BatchGatherMismatch;
            }
            if (lane_emb[i] != fx_emb[i]) {
                log.err("B1/gather FAIL (emb vs fixture) lane={d} tok={d} idx={d} : batché=0x{x} fixture=0x{x}", .{ lane, fed_fx[lane], i, lane_emb[i], fx_emb[i] });
                return error.BatchGatherMismatch;
            }
        }
        for (0..lf_u) |i| {
            if (lane_eptl[i] != ref_eptl[i]) {
                log.err("B1/gather FAIL (eptl) lane={d} tok={d} idx={d} : batché=0x{x} 1-lane=0x{x}", .{ lane, fed_fx[lane], i, lane_eptl[i], ref_eptl[i] });
                return error.BatchGatherMismatch;
            }
            if (lane_eptl[i] != fx_eptl[i]) {
                log.err("B1/gather FAIL (eptl vs fixture) lane={d} tok={d} idx={d} : batché=0x{x} fixture=0x{x}", .{ lane, fed_fx[lane], i, lane_eptl[i], fx_eptl[i] });
                return error.BatchGatherMismatch;
            }
        }
    }
    log.info("B1/1 gather PASS — tok={{{d},{d}}} : 2 lanes × 2 tables bit-exact vs gather 1-lane ET vs fixture", .{ fed_fx[0], fed_fx[1] });

    // ================= Sous-test 2 : scatterSlices batché (pos scalaire partagé) =================
    const SCT_K: usize = 8;
    const SCT_HD: usize = 4;
    const SCT_POS: u32 = 3;
    const cache_sym = zml.Tensor.init(.{ 1, 2, 1, SCT_K, SCT_HD }, .f32).withTags(.{ .slot, .b, .h, .k, .hd });
    const upd_sym = zml.Tensor.init(.{ 2, 1, 1, SCT_HD }, .f32).withTags(.{ .b, .h, .k, .hd });
    const pos_sym = zml.Tensor.init(.{}, .u32);
    var exe_s = try platform.compileFn(allocator, io, BbSct.forward, .{ cache_sym, upd_sym, pos_sym }, .{ .shardings = &.{sharding} });
    defer exe_s.deinit();

    var cache_host = [_]f32{0} ** (1 * 2 * 1 * SCT_K * SCT_HD);
    var upd_host: [2 * SCT_HD]f32 = undefined;
    for (0..SCT_HD) |i| {
        upd_host[i] = 1.0; // lane 0
        upd_host[SCT_HD + i] = 2.0; // lane 1 — valeurs DISTINCTES : une lane qui reçoit la valeur
        // de l'autre (ou les deux la même) est le mode de panne exact que ce test doit attraper.
    }
    var cache_buf_t = try zml.Buffer.fromBytes(io, platform, cache_sym.shape(), sharding, std.mem.sliceAsBytes(&cache_host));
    defer cache_buf_t.deinit();
    var upd_buf = try zml.Buffer.fromBytes(io, platform, upd_sym.shape(), sharding, std.mem.sliceAsBytes(&upd_host));
    defer upd_buf.deinit();
    var pos_buf = try zml.Buffer.scalar(io, platform, SCT_POS, .u32, sharding);
    defer pos_buf.deinit();

    var a_s = try exe_s.args(allocator);
    defer a_s.deinit(allocator);
    var r_s = try exe_s.results(allocator);
    defer r_s.deinit(allocator);
    a_s.set(.{ cache_buf_t, upd_buf, pos_buf });
    exe_s.call(a_s, &r_s);
    var sct_out = r_s.get(zml.Buffer);
    defer sct_out.deinit();
    var sct_s = try sct_out.toSliceAlloc(allocator, io);
    defer sct_s.free(allocator);
    const sct = sct_s.items(f32);
    if (sct.len != cache_host.len) {
        log.err("B1/scatter : longueur D2H {d} ≠ {d} attendue", .{ sct.len, cache_host.len });
        return error.UnexpectedShape;
    }
    // Layout {slot=1,b=2,h=1,k=8,hd=4} row-major → index = b*(K*HD) + k*HD + hd.
    for (0..2) |lane| {
        const want: f32 = if (lane == 0) 1.0 else 2.0;
        for (0..SCT_K) |k| {
            for (0..SCT_HD) |h| {
                const got = sct[lane * SCT_K * SCT_HD + k * SCT_HD + h];
                const expect: f32 = if (k == SCT_POS) want else 0.0;
                if (got != expect) {
                    log.err("B1/scatter FAIL — lane={d} k={d} hd={d} : got={d} attendu={d} (pos scalaire={d})", .{ lane, k, h, got, expect, SCT_POS });
                    return error.BatchScatterMismatch;
                }
            }
        }
    }
    log.info("B1/2 scatterSlices PASS — pos scalaire={d} partagé : lane0=1.0 et lane1=2.0 écrites à k={d}, zéros ailleurs", .{ SCT_POS, SCT_POS });

    // ============ Sous-test 3 : topK {b=2,s=1,voc=16} — layout D2H {b,K} + dtype i32 ============
    const VOC_T: usize = 16;
    const logits_sym = zml.Tensor.init(.{ 2, 1, VOC_T }, .f32).withTags(.{ .b, .s, .voc });
    var exe_t = try platform.compileFn(allocator, io, BbTop.forward, .{logits_sym}, .{ .shardings = &.{sharding} });
    defer exe_t.deinit();

    // Valeurs TOUTES DISTINCTES par lane (7 est premier avec 16 → permutation) et argmax distincts
    // entre lanes : v[l][j] = ((7*j + 3*l) % 16). Zéro tie ⇒ le tri est déterministe, l'attendu est
    // calculable exactement host-side (pas de dépendance à la politique de tie-break d'XLA).
    var logits_host: [2 * VOC_T]f32 = undefined;
    for (0..2) |l| {
        for (0..VOC_T) |j| logits_host[l * VOC_T + j] = @floatFromInt((7 * j + 3 * l) % VOC_T);
    }
    var logits_buf = try zml.Buffer.fromBytes(io, platform, logits_sym.shape(), sharding, std.mem.sliceAsBytes(&logits_host));
    defer logits_buf.deinit();
    var a_t = try exe_t.args(allocator);
    defer a_t.deinit(allocator);
    var r_t = try exe_t.results(allocator);
    defer r_t.deinit(allocator);
    a_t.set(.{logits_buf});
    exe_t.call(a_t, &r_t);
    var t_val, var t_idx = r_t.get(struct { zml.Buffer, zml.Buffer });
    defer t_val.deinit();
    defer t_idx.deinit();
    var tval_s = try t_val.toSliceAlloc(allocator, io);
    defer tval_s.free(allocator);
    var tidx_s = try t_idx.toSliceAlloc(allocator, io);
    defer tidx_s.free(allocator);
    // dtype des indices : `topK` délègue à `sort`, dont les indices viennent de Tensor.arange(…, .i32)
    // (tensor.zig:2977) — VÉRIFIÉ, jamais supposé (même standard que la boucle réelle).
    if (tidx_s.dtype() != .i32) {
        log.err("B1/topK : indices dtype={s} ≠ i32 attendu", .{@tagName(tidx_s.dtype())});
        return error.UnexpectedDtype;
    }
    const tvals = tval_s.items(f32);
    const tidxs = tidx_s.items(i32);
    const k_u: usize = K_TOPK;
    if (tvals.len != 2 * k_u or tidxs.len != 2 * k_u) {
        log.err("B1/topK : layout D2H inattendu (values.len={d}, indices.len={d}, attendu {d} = b×K)", .{ tvals.len, tidxs.len, 2 * k_u });
        return error.UnexpectedShape;
    }
    for (0..2) |l| {
        for (0..k_u) |r| {
            // rang r ⇒ valeur attendue 15-r ; l'indice j tel que (7j + 3l) % 16 == 15-r.
            const want_val: usize = VOC_T - 1 - r;
            var want_idx: ?usize = null;
            for (0..VOC_T) |j| {
                if ((7 * j + 3 * l) % VOC_T == want_val) want_idx = j;
            }
            const wi = want_idx.?;
            const got_idx: usize = @intCast(tidxs[l * k_u + r]);
            const got_val = tvals[l * k_u + r];
            if (got_idx != wi or got_val != @as(f32, @floatFromInt(want_val))) {
                log.err("B1/topK FAIL — lane={d} rang={d} : got idx={d} val={d} ; attendu idx={d} val={d}", .{ l, r, got_idx, got_val, wi, want_val });
                return error.BatchTopkMismatch;
            }
        }
    }
    log.info("B1/3 topK PASS — layout D2H {{b={d}, K={d}}} (stride {d} par lane), indices i32, argmax distincts par lane", .{ 2, k_u, k_u });

    // =================== Sous-test 4 : broad du masque rank-égal (b=1 → b=2) ===================
    const BRD_K: usize = 8;
    const scores_sym = zml.Tensor.init(.{ 2, 1, 1, BRD_K }, .f32).withTags(.{ .b, .h, .q, .k });
    const mask_sym = zml.Tensor.init(.{ 1, 1, 1, BRD_K }, .f32).withTags(.{ .b, .h, .q, .k });
    var exe_b = try platform.compileFn(allocator, io, BbBrd.forward, .{ scores_sym, mask_sym }, .{ .shardings = &.{sharding} });
    defer exe_b.deinit();

    // Masque additif de forme causale/fenêtrée (0 visible / -100 masqué) — valeur FINIE (pas
    // MASK_MIN) pour que l'addition reste bit-exacte en f32 et l'attendu vérifiable à l'égalité
    // stricte ; ce qu'on prouve ici c'est la DIFFUSION du masque aux 2 lanes, pas la saturation.
    var scores_host: [2 * BRD_K]f32 = undefined;
    var mask_host: [BRD_K]f32 = undefined;
    for (0..BRD_K) |j| mask_host[j] = if (j < 4) 0.0 else -100.0;
    for (0..2) |l| {
        for (0..BRD_K) |j| scores_host[l * BRD_K + j] = @floatFromInt(l * 10 + j);
    }
    var scores_buf = try zml.Buffer.fromBytes(io, platform, scores_sym.shape(), sharding, std.mem.sliceAsBytes(&scores_host));
    defer scores_buf.deinit();
    var mask_buf = try zml.Buffer.fromBytes(io, platform, mask_sym.shape(), sharding, std.mem.sliceAsBytes(&mask_host));
    defer mask_buf.deinit();
    var a_b = try exe_b.args(allocator);
    defer a_b.deinit(allocator);
    var r_b = try exe_b.results(allocator);
    defer r_b.deinit(allocator);
    a_b.set(.{ scores_buf, mask_buf });
    exe_b.call(a_b, &r_b);
    var brd_out = r_b.get(zml.Buffer);
    defer brd_out.deinit();
    var brd_s = try brd_out.toSliceAlloc(allocator, io);
    defer brd_s.free(allocator);
    const brd = brd_s.items(f32);
    if (brd.len != 2 * BRD_K) {
        log.err("B1/broad : longueur D2H {d} ≠ {d}", .{ brd.len, 2 * BRD_K });
        return error.UnexpectedShape;
    }
    for (0..2) |l| {
        for (0..BRD_K) |j| {
            const expect = scores_host[l * BRD_K + j] + mask_host[j];
            const got = brd[l * BRD_K + j];
            if (got != expect) {
                log.err("B1/broad FAIL — lane={d} k={d} : got={d} attendu={d} (le masque b=1 n'a PAS été diffusé à cette lane)", .{ l, j, got, expect });
                return error.BatchBroadMismatch;
            }
        }
    }
    log.info("B1/4 broad PASS — masque b=1 rank-égal diffusé aux 2 lanes (positions), mask=[0×4, -100×4] appliqué à l'identique", .{});

    log.info("B1 PASS — 4/4 sous-tests (gather, scatterSlices batché, topK {{b,K}}, broad rank-égal) exacts à B=2", .{});
}

// ============================================================================================
// L3 — table `Tabs` (embed_tokens_per_layer, ~4,7 Go bf16) : SEULE table ajoutée au device —
// `embed_tokens` est DÉJÀ device-résident dans `Model` (lm_head tied, engine.zig:511), le gather
// le réutilise. Nom court OBLIGATOIRE (quota comptime @typeName).
// ============================================================================================
const Tabs = struct {
    eptl: zml.Tensor, // {voc,lf} bf16 BRUT (scaling ×16 déjà dans forwardStep)

    fn init(base: zml.io.TensorStore.View) Tabs {
        return .{ .eptl = base.createTensor("embed_tokens_per_layer.weight", .{ .voc, .lf }, null) };
    }
    fn load(self: *const Tabs, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Tabs) {
        return zml.io.load(Tabs, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

// BBStep : copie OP-POUR-OP de `StepTok` (gen_auto.zig:742-755) — gather + forwardStep + topK sont
// tous shape-polymorphes, le code est IDENTIQUE, SEULES les shapes d'entrée changent (tok {B,1},
// cache {slot,B,1,L_MAX,HD}). ⚠ CROSS-REF : si le dtype/shape des indices de gather change ici,
// changer À L'IDENTIQUE le tok_sym du selftest (BbGat) — sinon B1 resterait vert en validant autre
// chose que ce que le runtime fait.
const BBStep = struct {
    pub fn forward(model: Model, tabs: Tabs, tok: zml.Tensor, p: PackedLong, cache: engine.Cache, ctrl: engine.Ctrl) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        const e = model.embed_tokens.gather(.{ .voc = tok }, .{}); // {b,s,d} bf16 brut — GatherOpts `.{}` OBLIGATOIRE
        const el = tabs.eptl.gather(.{ .voc = tok }, .{}); // {b,s,lf} bf16 brut
        const logits, const slk, const slv, const flk, const flv = model.forwardStep(e, el, p, cache, ctrl);
        // Forme struct à un champ EXIGÉE par `Tensor.topK` (tensor.zig:3098) : `.{ .voc = .voc }`,
        // PAS `.topK(.voc, …)` (un enum literal seul ne matche aucune branche → échec de compile).
        const t5 = logits.topK(.{ .voc = .voc }, K_TOPK, .{});
        return .{ t5.values, t5.indices, slk, slv, flk, flv };
    }
};

// top-K par lane, rapatrié du device (K_TOPK entrées) : top1 = next token ; le K entier sert au
// diagnostic --oracles (marge top1−top2 : vigilance ties d'argmax, spec §4).
const Top5 = struct { idx: [K_TOPK]usize, val: [K_TOPK]f32 };

const StopReason = enum { running, oracle, eot, max_tokens, l_max };

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000); // piège quota comptime (35 couches inline × traçage)
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const args = try parseArgs(arena.allocator(), process_args);

    // === Mode --selftest-batch (gate B1) : AUTONOME — dispatché AVANT l'exigence de --prompts
    // (contrairement à gen_auto dont --selftest-gather réclame un --prompt factice). C'est du
    // travail GPU : il passe par la garde de contention + la garde CUDA dure, comme un run normal.
    if (args.selftest_batch) |fx| {
        if (args.force_vram) {
            log.warn("--force-vram : garde de contention sautée (OOM/mesure polluée possibles, assumé)", .{});
        } else {
            try checkContention(allocator, io);
        }
        const platform = try initPlatform(allocator, io, args.no_prealloc, args.allow_cpu);
        defer platform.deinit(allocator);
        const sharding = try zml.sharding.replicatedSharding(platform);
        try selftestBatch(allocator, io, platform, sharding, args.ckpt, fx);
        return;
    }

    const prompts_path = args.prompts_path orelse {
        log.err("--prompts est requis (hors --selftest-batch)\n{s}", .{usage});
        return error.MissingArgument;
    };

    // === Tokenisation par lane (host-only) ===
    var prompts_buf: []u8 = &.{};
    const prompts = try readPromptLines(allocator, io, prompts_path, &prompts_buf);
    defer allocator.free(prompts_buf);
    defer allocator.free(prompts);
    const n_base = prompts.len;

    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, args.tokjson_path);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    // EOT_ID — MESURÉ depuis le tokenizer (jamais hardcodé) : encode "<turn|>" et exige EXACTEMENT
    // 1 id. Un compte ≠ 1 = tokenizer/template différent de celui mesuré (id=106) → BLOCKED plutôt
    // qu'un repli silencieux. Il sert AUSSI de token de bourrage des lanes finies (borné vocab).
    var eot_tok = try encoder.encodeAlloc(allocator, "<turn|>");
    defer eot_tok.deinit(allocator);
    if (eot_tok.items.len != 1) {
        log.err("EOT: '<turn|>' encode en {d} tokens (attendu 1) — ids={any}", .{ eot_tok.items.len, eot_tok.items });
        return error.EotNotSingleToken;
    }
    const eot_id: u32 = eot_tok.items[0];
    log.info("EOT_ID = {d} (mesuré depuis le tokenizer)", .{eot_id});

    // ids par PROMPT DE BASE (avant --replicate) : template chat + BOS préfixé.
    const base_ids = try allocator.alloc([]u32, n_base);
    // Pré-initialisé à vide AVANT le `defer` : sur un échec d'encodage à mi-boucle, le defer ne doit
    // pas libérer des slices non initialisées (les entrées produites font toujours len ≥ 1 — BOS).
    for (base_ids) |*s| s.* = &.{};
    defer {
        for (base_ids) |ids| {
            if (ids.len != 0) allocator.free(ids);
        }
        allocator.free(base_ids);
    }
    for (prompts, 0..) |prompt, i| {
        encoder.reset(); // l'encoder iree est un automate à ÉTAT : reset() entre deux prompts.
        const rendered = try renderChatTemplate(arena.allocator(), prompt);
        var tok = try encoder.encodeAlloc(allocator, rendered);
        defer tok.deinit(allocator);
        var ids: std.ArrayList(u32) = try .initCapacity(allocator, tok.items.len + 1);
        errdefer ids.deinit(allocator);
        try ids.append(allocator, BOS_ID);
        try ids.appendSlice(allocator, tok.items);
        base_ids[i] = try ids.toOwnedSlice(allocator);
    }

    // === --ids-only : OUTIL de constitution du jeu de prompts de même longueur (spec §3.2) — il
    // sort AVANT la garde d'uniformité (c'est précisément lui qui sert à la satisfaire) et avant
    // tout travail GPU. ===
    if (args.ids_only) {
        for (base_ids, 0..) |ids, i| {
            log.info("lane {d} : len={d} ids={any}", .{ i, ids.len, ids });
            // Round-trip détok : decode les ids APRÈS bos (partie produite par l'encoder, gabarit de
            // chat INCLUS) puis re-encode. Décodeur FRAIS par lane (automate iree à état).
            var decoder = try tokenizer.decoder();
            defer decoder.deinit();
            var text_rt = try decoder.decodeAlloc(allocator, ids[1..]);
            defer text_rt.deinit(allocator);
            encoder.reset();
            var reenc = try encoder.encodeAlloc(allocator, text_rt.items);
            defer reenc.deinit(allocator);
            if (std.mem.eql(u32, reenc.items, ids[1..])) {
                log.info("  round-trip détok lane {d} : PASS", .{i});
            } else {
                log.err("  round-trip détok lane {d} : FAIL — got={any} want={any}", .{ i, reenc.items, ids[1..] });
                return error.RoundTripFailed;
            }
        }
        // Récap trié par longueur : lecture directe des candidats à regrouper (contrainte V1 =
        // longueurs identiques).
        const order = try allocator.alloc(usize, n_base);
        defer allocator.free(order);
        for (order, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order, base_ids, struct {
            fn lt(ctx: [][]u32, a: usize, b: usize) bool {
                return ctx[a].len < ctx[b].len;
            }
        }.lt);
        log.info("récap (trié par longueur) :", .{});
        for (order) |i| log.info("  len={d}  lane {d}  prompt=\"{s}\"", .{ base_ids[i].len, i, prompts[i] });
        return;
    }

    // === CONTRAINTE V1 : longueurs tokenisées IDENTIQUES (verrou du moteur, cf tête de fichier) ===
    const ids_len = base_ids[0].len;
    var uniform = true;
    for (base_ids) |ids| {
        if (ids.len != ids_len) uniform = false;
    }
    if (!uniform) {
        log.err("error.PromptLengthMismatch — toutes les lanes doivent avoir la MÊME longueur tokenisée (position partagée : ctrl.step et pos_u sont des scalaires uniques pour tout le batch). Longueurs :", .{});
        for (base_ids, 0..) |ids, i| log.err("  lane {d} : len={d}  prompt=\"{s}\"", .{ i, ids.len, prompts[i] });
        log.err("Ajuster les textes jusqu'à égalité — utiliser `--ids-only` (outil de constitution du jeu).", .{});
        return error.PromptLengthMismatch;
    }

    // === --oracles : fixtures 49 APPARIÉES PAR INDEX aux lignes de --prompts. Lues AVANT tout
    // travail GPU (fail-fast). La fixture ne porte PAS les ids du prompt : la garde est double —
    // positions[0] == ids.len (fixture) ET prompt_ids == ids tokenisés (manifest sidecar), sinon une
    // fixture désappariée de même longueur donnerait un faux PASS. ===
    var oracles: ?[]Oracle = null;
    defer if (oracles) |ors| {
        for (ors) |*o| o.deinit(allocator);
        allocator.free(ors);
    };
    if (args.oracles) |paths| {
        if (paths.len != n_base) {
            log.err("error.OracleCountMismatch — {d} fixtures --oracles pour {d} lignes de --prompts (appariement PAR INDEX exigé)", .{ paths.len, n_base });
            return error.OracleCountMismatch;
        }
        const ors = try allocator.alloc(Oracle, paths.len);
        errdefer allocator.free(ors);
        for (paths, 0..) |path, i| {
            ors[i] = try loadOracle(allocator, io, path);
            const o = ors[i];
            if (o.seq_len != @as(i32, @intCast(ids_len))) {
                log.err("--oracles lane {d} : positions[0]={d} (seq_len fixture) != ids.len={d} (prompt rendu) — fixture désappariée ({s})", .{ i, o.seq_len, ids_len, path });
                return error.OraclePromptMismatch;
            }
            if (!std.mem.eql(u32, o.prompt_ids, base_ids[i])) {
                log.err("--oracles lane {d} : prompt_ids du manifest != ids tokenisés ({s})", .{ i, path });
                log.err("  manifest={any}", .{o.prompt_ids});
                log.err("  tokenisé={any}", .{base_ids[i]});
                return error.OraclePromptMismatch;
            }
            if (o.fed.len != ors[0].fed.len) {
                log.err("--oracles : fed.len hétérogène (lane {d} : {d} ≠ lane 0 : {d}) — regénérer les fixtures avec le MÊME --n-tokens", .{ i, o.fed.len, ors[0].fed.len });
                return error.OracleCountMismatch;
            }
        }
        oracles = ors;
        log.info("--oracles : {d} fixtures appariées, {d} steps de génération attendus par lane (prompts vérifiés : positions[0] + prompt_ids du manifest)", .{ paths.len, ors[0].fed.len });
    }

    // === B = nombre de lanes = lignes × --replicate (RUNTIME : le moteur est shape-polymorphe,
    // aucun mismatch compile/runtime possible). Lane i ↔ prompt i % n_base (donc ↔ oracle i % n_base
    // : ça donne le spot-check lane 0 des points de sweep B≥8 sans code supplémentaire). ===
    const b_lanes: usize = n_base * args.replicate;
    const b_dim: i64 = @intCast(b_lanes);
    log.info("B = {d} lanes ({d} prompt(s) × --replicate {d}), ids.len = {d} (uniforme)", .{ b_lanes, n_base, args.replicate, ids_len });

    const max_tokens: usize = args.max_tokens orelse 200;
    const limit: usize = if (oracles) |ors| ors[0].fed.len else max_tokens;
    if (oracles != null and args.max_tokens != null) {
        log.warn("--oracles actif : --max-tokens={d} ignoré (limite = fed.len = {d})", .{ args.max_tokens.?, limit });
    }

    // Gardes PAR LANE (mêmes asserts que l'oracle 49) — ids.len étant uniforme, elles portent sur
    // toutes les lanes à la fois.
    if (ids_len + limit > @as(usize, @intCast(L_MAX))) {
        log.err("garde-fou : ids.len({d}) + limit({d}) > L_MAX({d})", .{ ids_len, limit, L_MAX });
        return error.SequenceTooLong;
    }
    if (ids_len >= @as(usize, @intCast(SLIDING_WINDOW))) {
        log.err("garde-fou : ids.len({d}) >= SLIDING_WINDOW({d})", .{ ids_len, SLIDING_WINDOW });
        return error.PromptTooLong;
    }

    // === Garde de contention + backend ===
    if (args.force_vram) {
        log.warn("--force-vram : garde de contention sautée (OOM/mesure polluée possibles, assumé)", .{});
    } else {
        try checkContention(allocator, io);
    }
    const platform = try initPlatform(allocator, io, args.no_prealloc, args.allow_cpu);
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);
    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const model: Model = try .init(arena.allocator(), base);

    // === Symboliques : SEULES les shapes tok/cache portent B. Les tables Packed restent à b=1
    // (INVARIANT VRAM, spec §2.8) — les étendre ×B gaspillerait la VRAM pour rien, le broadcast
    // rank-égal (prouvé en B1/4) les diffuse aux lanes. ===
    const tok_sym = zml.Tensor.init(.{ b_dim, 1 }, .u32).withTags(.{ .b, .s });
    const packed_sym = PackedLong{
        .embeds = zml.Tensor.init(.{ L_MAX, 1, 1, D }, .bf16).withTags(.{ .step, .b, .s, .d }),
        .embptls = zml.Tensor.init(.{ L_MAX, 1, 1, LF }, .bf16).withTags(.{ .step, .b, .s, .lf }),
        .cos_full = zml.Tensor.init(.{ L_MAX, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .sin_full = zml.Tensor.init(.{ L_MAX, 1, 1, HD_F }, .f32).withTags(.{ .step, .b, .s, .hd }),
        .masks_sliding = zml.Tensor.init(.{ L_MAX, 1, 1, 1, L_MAX }, .f32).withTags(.{ .step, .b, .h, .q, .k }),
        .masks_full = zml.Tensor.init(.{ L_MAX, 1, 1, 1, L_MAX }, .f32).withTags(.{ .step, .b, .h, .q, .k }),
        .positions = zml.Tensor.init(.{L_MAX}, .i32).withTags(.{.step}),
    };
    const cache_sym = engine.Cache{
        .sl_k = zml.Tensor.init(.{ NUM_SLIDING_SLOTS, b_dim, 1, L_MAX, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .sl_v = zml.Tensor.init(.{ NUM_SLIDING_SLOTS, b_dim, 1, L_MAX, HD_S }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_k = zml.Tensor.init(.{ NUM_FULL_SLOTS, b_dim, 1, L_MAX, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
        .fl_v = zml.Tensor.init(.{ NUM_FULL_SLOTS, b_dim, 1, L_MAX, HD_F }, .f32).withTags(.{ .slot, .b, .h, .k, .hd }),
    };
    const ctrl_sym: engine.Ctrl = .initSymbolic();

    log.info("Materializing weights + Packed (b=1) / Cache (b={d}) ...", .{b_lanes});
    const eng_buf = try model.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const tabs: Tabs = .init(base); // même view withPrefix que Model.init
    const tabs_buf = try tabs.load(arena.allocator(), io, platform, &store_ck, &.{sharding});

    var host = try HostInputs.init(allocator, b_lanes);
    defer host.deinit(allocator);
    // Bufferized assemblé champ-à-champ (motif E2) depuis les slices host.
    const pk_buf = zml.Bufferized(PackedLong){
        .embeds = try zml.Buffer.fromBytes(io, platform, packed_sym.embeds.shape(), sharding, host.embeds_zero),
        .embptls = try zml.Buffer.fromBytes(io, platform, packed_sym.embptls.shape(), sharding, host.embptls_zero),
        .cos_full = try zml.Buffer.fromBytes(io, platform, packed_sym.cos_full.shape(), sharding, std.mem.sliceAsBytes(host.cos_full)),
        .sin_full = try zml.Buffer.fromBytes(io, platform, packed_sym.sin_full.shape(), sharding, std.mem.sliceAsBytes(host.sin_full)),
        .masks_sliding = try zml.Buffer.fromBytes(io, platform, packed_sym.masks_sliding.shape(), sharding, std.mem.sliceAsBytes(host.masks_sliding)),
        .masks_full = try zml.Buffer.fromBytes(io, platform, packed_sym.masks_full.shape(), sharding, std.mem.sliceAsBytes(host.masks_full)),
        .positions = try zml.Buffer.fromBytes(io, platform, packed_sym.positions.shape(), sharding, std.mem.sliceAsBytes(host.positions)),
    };
    var cache_buf = zml.Bufferized(engine.Cache){
        .sl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_k.shape(), sharding, host.cache_sl_k),
        .sl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.sl_v.shape(), sharding, host.cache_sl_v),
        .fl_k = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_k.shape(), sharding, host.cache_fl_k),
        .fl_v = try zml.Buffer.fromBytes(io, platform, cache_sym.fl_v.shape(), sharding, host.cache_fl_v),
    };
    store_ck.deinit();
    reg_ck.deinit();
    mem_probe.logMem(io, "post-load (poids + Packed/Cache sur device)"); // ⚠ RSS HOST, pas VRAM

    log.info("Compiling BBStep.forward (gather+forwardStep+topK, 35 couches, B={d}) ...", .{b_lanes});
    const t_compile: std.Io.Timestamp = .now(io, .awake);
    var exe = try platform.compileFn(allocator, io, BBStep.forward, .{ model, tabs, tok_sym, packed_sym, cache_sym, ctrl_sym }, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    log.info("  compile: {f} (consigné au manifest du sweep : un graphe par valeur de B)", .{t_compile.untilNow(io, .awake)});
    mem_probe.logMem(io, "post-compile (go/no-go)");

    // === Boucle prefill-par-decode BATCHÉE ===
    // Toutes les lanes partagent ids_len ⇒ la frontière prefill/génération est COMMUNE
    // (in_gen_phase = step + 1 >= ids_len), comme le `ctrl.step` scalaire l'impose.
    const generated = try allocator.alloc(std.ArrayList(i64), b_lanes);
    defer {
        for (generated) |*g| g.deinit(allocator);
        allocator.free(generated);
    }
    const gen_top5 = try allocator.alloc(std.ArrayList(Top5), b_lanes);
    defer {
        for (gen_top5) |*g| g.deinit(allocator);
        allocator.free(gen_top5);
    }
    const active = try allocator.alloc(bool, b_lanes);
    defer allocator.free(active);
    const stop = try allocator.alloc(StopReason, b_lanes);
    defer allocator.free(stop);
    const eot_step = try allocator.alloc(?usize, b_lanes);
    defer allocator.free(eot_step);
    const fed = try allocator.alloc(i64, b_lanes);
    defer allocator.free(fed);
    const tok_host = try allocator.alloc(u32, b_lanes);
    defer allocator.free(tok_host);
    const tok_of = try allocator.alloc(i64, b_lanes); // token produit par lane AU STEP COURANT (B3)
    defer allocator.free(tok_of);
    for (0..b_lanes) |i| {
        generated[i] = .empty;
        gen_top5[i] = .empty;
        active[i] = true;
        stop[i] = .running;
        eot_step[i] = null;
        fed[i] = @intCast(base_ids[i % n_base][0]);
    }

    // Gate B3 (indépendance inter-lanes) : actif dès qu'il y a réplication — chaque lane i ≥ n_base
    // partage son prompt avec la lane i % n_base et DOIT produire exactement les mêmes ids, au même
    // step, avec le même step d'EOT. (À n_base=1 : « toutes les lanes == lane 0 », le pinning du gate.)
    const b3_on = args.replicate > 1;
    var b3_steps_ok: usize = 0;
    var b3_steps_cmp: usize = 0;
    var b3_fail: ?struct { step: usize, lane: usize, ref: usize, got: i64, want: i64, active_lane: bool, active_ref: bool } = null;

    const vocab = model.embed_tokens.dim(.voc);
    var step: usize = 0;
    const t0: std.Io.Timestamp = .now(io, .awake);
    var t_prefill_end: std.Io.Timestamp = t0; // capturé au DERNIER step de prefill (jamais lu à t0)
    while (true) : (step += 1) {
        // Bounds-check PAR LANE avant le cast : le `gather` XLA CLAMPE silencieusement les indices
        // hors-borne (divergence plausible mais fausse, pas un crash) et `@intCast` vers u32 est UB
        // en ReleaseFast hors plage. Portée : le chemin `fed` (host→device) ; les indices issus de
        // topK/arange sont ≥ 0 par construction.
        for (0..b_lanes) |i| {
            if (fed[i] < 0 or fed[i] >= vocab) {
                log.err("lane {d} : token hors vocab: {d} (vocab={d})", .{ i, fed[i], vocab });
                return error.TokenOutOfRange;
            }
            tok_host[i] = @intCast(fed[i]);
        }
        var tok_buf = try zml.Buffer.fromBytes(io, platform, tok_sym.shape(), sharding, std.mem.sliceAsBytes(tok_host));
        var step_buf = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(step)), .u32, sharding);
        const ctrl_buf = zml.Bufferized(engine.Ctrl){ .step = step_buf };

        var call_args = try exe.args(allocator);
        var call_results = try exe.results(allocator);
        call_args.set(.{ eng_buf, tabs_buf, tok_buf, pk_buf, cache_buf, ctrl_buf });
        exe.call(call_args, &call_results);
        var r_t5v, var r_t5i, const r_slk, const r_slv, const r_flk, const r_flv = call_results.get(struct {
            zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
        });

        const in_gen_phase = step + 1 >= ids_len;

        // D2H : top-K sort en {b, K} → extraction par lane (stride K).
        var t5v_s = try r_t5v.toSliceAlloc(allocator, io);
        defer t5v_s.free(allocator);
        var t5i_s = try r_t5i.toSliceAlloc(allocator, io);
        defer t5i_s.free(allocator);
        // dtype vérifié À CHAQUE step (coût nul : compare d'enum) plutôt que supposé — `topK`
        // délègue à `sort`, dont les indices viennent de Tensor.arange(…, .i32) (tensor.zig:2977).
        if (t5i_s.dtype() != .i32) {
            log.err("t5.indices : dtype={s} ≠ i32 attendu (topK/sort)", .{@tagName(t5i_s.dtype())});
            return error.UnexpectedDtype;
        }
        const t5i = t5i_s.items(i32);
        const t5v = t5v_s.items(f32);
        const k_u: usize = K_TOPK;
        if (t5i.len != b_lanes * k_u or t5v.len != b_lanes * k_u) {
            log.err("layout D2H topK inattendu : indices.len={d} values.len={d} (attendu {d} = B×K)", .{ t5i.len, t5v.len, b_lanes * k_u });
            return error.UnexpectedShape;
        }

        for (0..b_lanes) |i| {
            var top5: Top5 = undefined;
            for (0..k_u) |j| {
                top5.idx[j] = @intCast(t5i[i * k_u + j]);
                top5.val[j] = t5v[i * k_u + j];
            }
            tok_of[i] = @intCast(top5.idx[0]);
            // Sortie MASQUÉE pour une lane finie : elle continue de tourner dans le graphe (on lui
            // feed EOT, cf plus bas) mais ses tokens ne sont plus collectés.
            if (in_gen_phase and active[i]) {
                try generated[i].append(allocator, tok_of[i]);
                try gen_top5[i].append(allocator, top5);
            }
        }

        // Gate B3 — comparé AVANT la mise à jour de `active` : `active` reflète encore l'état des
        // lanes PENDANT ce step (un écart de step d'EOT se voit donc comme un écart d'activité).
        if (b3_on and in_gen_phase) {
            var all_ok = true;
            for (n_base..b_lanes) |i| {
                const ref = i % n_base;
                if (active[i] != active[ref]) {
                    all_ok = false;
                    if (b3_fail == null) b3_fail = .{ .step = step, .lane = i, .ref = ref, .got = tok_of[i], .want = tok_of[ref], .active_lane = active[i], .active_ref = active[ref] };
                    continue;
                }
                if (!active[i]) continue;
                if (tok_of[i] != tok_of[ref]) {
                    all_ok = false;
                    if (b3_fail == null) b3_fail = .{ .step = step, .lane = i, .ref = ref, .got = tok_of[i], .want = tok_of[ref], .active_lane = true, .active_ref = true };
                }
            }
            b3_steps_cmp += 1;
            if (all_ok) b3_steps_ok += 1;
        }

        // cache swap : deinit l'ancien, adopte le nouveau (motif gen_long_gpu).
        var old_cache = cache_buf;
        cache_buf = zml.Bufferized(engine.Cache){ .sl_k = r_slk, .sl_v = r_slv, .fl_k = r_flk, .fl_v = r_flv };
        old_cache.sl_k.deinit();
        old_cache.sl_v.deinit();
        old_cache.fl_k.deinit();
        old_cache.fl_v.deinit();

        r_t5v.deinit();
        r_t5i.deinit();
        tok_buf.deinit();
        step_buf.deinit();
        call_args.deinit(allocator);
        call_results.deinit(allocator);

        if ((step + 1) % 256 == 0) log.info("  ... step {d}", .{step + 1});

        // Fin du DERNIER step de prefill (frontière COMMUNE aux lanes — même ids_len).
        if (step + 1 == ids_len) t_prefill_end = .now(io, .awake);

        if (step + 1 < ids_len) {
            // Phase 1 (prefill) : les argmax sont IGNORÉS, on re-feed le prompt de chaque lane.
            for (0..b_lanes) |i| fed[i] = @intCast(base_ids[i % n_base][step + 1]);
            continue;
        }

        // Phase 2 (génération ; s0 est produit par le DERNIER step de prefill et déjà collecté).
        var any_active = false;
        for (0..b_lanes) |i| {
            if (active[i]) {
                if (oracles != null) {
                    if (generated[i].items.len >= limit) {
                        active[i] = false;
                        stop[i] = .oracle;
                    }
                } else if (tok_of[i] == @as(i64, @intCast(eot_id))) {
                    active[i] = false;
                    stop[i] = .eot;
                    eot_step[i] = generated[i].items.len - 1; // index du EOT dans `generated`
                } else if (generated[i].items.len >= max_tokens) {
                    active[i] = false;
                    stop[i] = .max_tokens;
                }
            }
            // Lane finie : on lui feed l'EOT (token de bourrage BORNÉ VOCAB — il doit rester un id
            // valide, le gather clampe sinon) ; sa sortie est masquée au tour suivant.
            fed[i] = if (active[i]) tok_of[i] else @as(i64, @intCast(eot_id));
            if (active[i]) any_active = true;
        }
        if (!any_active) break; // toutes les lanes ont terminé (oracle/EOT/max_tokens)
        if (step + 1 >= @as(usize, @intCast(L_MAX))) {
            for (0..b_lanes) |i| {
                if (active[i]) {
                    active[i] = false;
                    stop[i] = .l_max;
                }
            }
            log.warn("garde L_MAX atteinte (step={d}) — arrêt forcé de toutes les lanes", .{step});
            break;
        }
    }
    const elapsed = t0.untilNow(io, .awake);
    // `gen_elapsed` échantillonné ICI, AVANT les deinit de cache (leur coût device polluerait la
    // fenêtre gen_s) — convention L3 EXACTE (gen_auto.zig:1119-1130).
    const gen_elapsed = t_prefill_end.untilNow(io, .awake);
    cache_buf.sl_k.deinit();
    cache_buf.sl_v.deinit();
    cache_buf.fl_k.deinit();
    cache_buf.fl_v.deinit();

    const elapsed_s = @as(f64, @floatFromInt(elapsed.toNanoseconds())) / std.time.ns_per_s;
    const gen_s = @as(f64, @floatFromInt(gen_elapsed.toNanoseconds())) / std.time.ns_per_s;
    const pf_s = elapsed_s - gen_s;
    var total_gen: usize = 0;
    for (generated) |g| total_gen += g.items.len;
    // Nuance de mesure assumée (convention L3, à conserver pour rester comparable aux 110-113 tok/s
    // B=1) : s0 est produit par le dernier call de prefill mais compté dans `generated` — le 1er
    // token de gén coûte ~0 s dans la fenêtre gen_s. Négligeable dès ~48 steps.
    const pf_rate_lane = if (pf_s > 0) @as(f64, @floatFromInt(ids_len)) / pf_s else 0;
    const pf_rate_agg = if (pf_s > 0) @as(f64, @floatFromInt(ids_len * b_lanes)) / pf_s else 0;
    const gen_rate_agg = if (gen_s > 0) @as(f64, @floatFromInt(total_gen)) / gen_s else 0;
    log.info("BB PERF : prefill {d} steps en {d:.3}s ({d:.1} tok/s par lane, {d:.1} tok/s agrégé) ; génération agrégée {d} tokens en {d:.3}s ({d:.1} tok/s) [B={d}, K={d}, backend={s}]", .{ ids_len, pf_s, pf_rate_lane, pf_rate_agg, total_gen, gen_s, gen_rate_agg, b_lanes, K_TOPK, @tagName(platform.target) });
    log.info("BB PERF par lane :", .{});
    for (0..b_lanes) |i| {
        const n = generated[i].items.len;
        const rate = if (gen_s > 0) @as(f64, @floatFromInt(n)) / gen_s else 0;
        log.info("  lane {d} : {d} tokens ({d:.1} tok/s) — arrêt={s}{s}", .{ i, n, rate, @tagName(stop[i]), if (eot_step[i] != null) " (EOT)" else "" });
    }
    mem_probe.logMem(io, "post-run (RSS host ; la VRAM se lit à nvidia-smi, --no-prealloc)");

    // === Gate B3 : indépendance inter-lanes (mode --replicate) ===
    if (b3_on) {
        if (b3_fail) |f| {
            log.err("B3 FAIL — divergence inter-lanes au step {d} : lane {d} vs lane {d} (même prompt)", .{ f.step, f.lane, f.ref });
            if (f.active_lane != f.active_ref) {
                log.err("  écart d'activité (step d'EOT différent) : active[{d}]={} active[{d}]={}", .{ f.lane, f.active_lane, f.ref, f.active_ref });
            } else {
                log.err("  ids divergents : lane {d} → {d} ; lane {d} → {d}", .{ f.lane, f.got, f.ref, f.want });
            }
            for (0..b_lanes) |i| log.err("  lane {d} : {d} tokens, step EOT={?d}", .{ i, generated[i].items.len, eot_step[i] });
            return error.LaneDivergence;
        }
        log.info("B3 PASS — lanes identiques {d}/{d} steps de génération comparés ({d} groupe(s) de prompts × {d} réplicas ; steps d'EOT identiques)", .{ b3_steps_ok, b3_steps_cmp, n_base, args.replicate });
        for (0..b_lanes) |i| log.info("  lane {d} : {d} tokens, step EOT={?d}", .{ i, generated[i].items.len, eot_step[i] });
    }

    // === Gate B2 : fidélité par lane vs fixtures HF (48/48 attendus) ===
    if (oracles) |ors| {
        var all_pass = true;
        for (0..b_lanes) |i| {
            const fx = ors[i % n_base].fed;
            const gen = generated[i].items;
            var n_match: usize = 0;
            var first_fail: ?usize = null;
            const n = @min(gen.len, fx.len);
            for (0..n) |k| {
                if (gen[k] == @as(i64, @intCast(fx[k]))) {
                    n_match += 1;
                } else if (first_fail == null) {
                    first_fail = k;
                }
            }
            const len_ok = gen.len == fx.len;
            if (first_fail == null and len_ok) {
                log.info("B2 lane {d} : PASS — {d}/{d} argmax-match == fixture HF", .{ i, n_match, fx.len });
                continue;
            }
            all_pass = false;
            const ff = first_fail orelse n;
            log.err("B2 lane {d} : FAIL — {d}/{d} match, 1er mismatch au step gen={d}{s}", .{ i, n_match, fx.len, ff, if (!len_ok) " (ou longueurs différentes)" else "" });
            if (ff < fx.len) {
                const got: i64 = if (ff < gen.len) gen[ff] else -1;
                log.err("  step gen={d} : généré={d} attendu(fed)={d}", .{ ff, got, fx[ff] });
            }
            if (ff < gen_top5[i].items.len) {
                const t5 = gen_top5[i].items[ff];
                // Diagnostic pré-enregistré (spec §4) : l'argmax est trop grossier pour trancher —
                // une marge top1−top2 fine = tie plausible (bifurcation LÉGITIME due aux GEMM
                // différentes à B>1), une marge large = vrai bug. Le FAIL brut est publié dans les
                // deux cas ; la requalification se fait dans la doc, jamais dans le code.
                const margin = t5.val[0] - t5.val[1];
                log.err("  diagnostic top-{d} @ step gen={d} : idx={any} val={any} — marge top1−top2 = {e}", .{ K_TOPK, ff, t5.idx, t5.val, margin });
            }
        }
        if (!all_pass) return error.B2Mismatch;
        log.info("B2 PASS — {d}/{d} lanes à {d}/{d} argmax-match (fidélité par lane vs fixtures HF mono)", .{ b_lanes, b_lanes, ors[0].fed.len, ors[0].fed.len });
        return;
    }

    // === Mode libre : détok par lane → texte sur STDOUT (les logs vont sur stderr) ===
    var stdout_w = std.Io.File.stdout().writer(io, &.{});
    for (0..b_lanes) |i| {
        const gen = generated[i].items;
        // Strip du EOT FINAL si l'arrêt vient de l'EOS (le texte de la réponse ne contient pas le
        // token de fin de tour) ; sinon tout `generated` est du texte.
        const n_text = if (stop[i] == .eot and gen.len > 0) gen.len - 1 else gen.len;
        const ids_u32 = try allocator.alloc(u32, n_text);
        defer allocator.free(ids_u32);
        // Conversion i64→u32 explicite à la frontière du décodeur (pas de reinterprétation de slice).
        for (gen[0..n_text], 0..) |t, k| ids_u32[k] = @intCast(t);
        // Décodeur FRAIS PAR LANE : l'automate iree est à ÉTAT — réutiliser un décodeur déjà
        // consommé ferait fuiter l'état d'une lane dans la suivante.
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        var text = try decoder.decodeAlloc(allocator, ids_u32);
        defer text.deinit(allocator);
        try stdout_w.interface.print("lane {d} ({d} tokens, arrêt={s}) : \"{s}\"\n", .{ i, gen.len, @tagName(stop[i]), text.items });
    }
    try stdout_w.interface.flush();
}
