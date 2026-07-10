// Runtime AUTONOME texte→texte (spec docs/GEN_AUTONOME_DESIGN.md).
// Gates : A0 tokenizer+template ; A1 prefill-par-decode 48/48 ; A2 long N/N ; A3 early-stop EOS.
// Le moteur engine.zig est INTACT — entrée compilée : forwardStep (embeds host token-dépendants).
//
// CLI : gemma4_gen_auto <model.safetensors> <tokenizer.json> --prompt "..." [--max-tokens N]
//       [--oracle fixture] [--ids-only] [--selftest-inputs f] [--selftest-gather f]
// Task 2 (gate A0) : parsing CLI, chargement tokenizer ZML natif, rendu du chat template,
// encodage, préfixage BOS explicite, mode `--ids-only` (log des ids finaux + round-trip détok).
// Task 3 : mode `--selftest-inputs <fixture>` — cos/sin RoPE full (`ropeFull`, formule
// "proportional" copiée de la source HF), masques additifs (`maskRows`), positions, et les
// tables host complètes {L_MAX,…} (cos_full/sin_full/masks_sliding/masks_full/positions +
// embeds/embptls/cache zéros, cf `HostInputs`) — validées vs la fixture 49.
// Task 4 (cette tranche) : gather embeds BRUTS en streaming (`EmbedGather`, pattern
// gchunk_auto.zig:137-230 adapté — offsets absolus sur le checkpoint, une ligne bf16 par token,
// scratch réutilisé, AUCUN scaling host) + mode `--selftest-gather <fixture>` (bit-exact vs
// embeds/embptls de la fixture A1). Les buffers device par step et la boucle autonome
// (forwardStep) restent Task 5.
const std = @import("std");
const log = std.log;
const zml = @import("zml");
const engine = @import("engine.zig");
const mem_probe = @import("mem_probe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const L_MAX: i64 = 1024;
const SLIDING_WINDOW: i64 = 512;
const HD_F: i64 = 512; // dim cos/sin full (= config.global_head_dim, cf ropeFull)
const HD_S: i64 = 256; // dim cache sliding (= engine.HD_SLIDING)
const D: i64 = 1536;
const LF: i64 = 8960;
// Slots de cache producteurs (cf engine.zig: isFull(i)=(i+1)%5==0, FIRST_KV_SHARED=15) : parmi les
// 15 premières couches, 3 sont "full" (4,9,14) et 12 "sliding" (les autres) — mêmes comptes que
// SLIDING_PRODUCERS/FULL_PRODUCERS de scripts/49_gen_custom_oracle.py:41-42.
const NUM_SLIDING_SLOTS: usize = 12;
const NUM_FULL_SLOTS: usize = 3;
const Model = engine.EngineModel(struct {}, .{ .two_masks = true, .kmax_sliding = L_MAX, .kmax_full = L_MAX });
const PackedLong = engine.Packed(true);

// BOS (id 2) : PRÉFIXÉ explicitement — l'encoder ZML (iree, cf zml/tokenizer/tokenizer.zig)
// n'ajoute AUCUN token spécial (constat Task 0 : ids ZML == ids HF sans template, modulo ce préfixe).
const BOS_ID: u32 = 2;

// Chat template Gemma — VÉRITÉ = repr() HF mesuré (10 juil) :
//   '<bos><|turn>user\nPROMPT<turn|>\n<|turn>model\n'
// ⚠ tokens de tour : <|turn> (id 105) / <turn|> (id 106) — PAS <start_of_turn>/<end_of_turn>.
// BOS (id 2) : PRÉFIXÉ en id (l'encoder ZML n'ajoute AUCUN token spécial) — le rendu texte
// commence donc APRÈS <bos>.
fn renderChatTemplate(allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "<|turn>user\n{s}<turn|>\n<|turn>model\n", .{prompt});
}

const Args = struct {
    ckpt: []const u8,
    tokjson_path: []const u8,
    prompt: ?[]const u8 = null,
    max_tokens: ?usize = null,
    oracle_path: ?[]const u8 = null,
    ids_only: bool = false,
    selftest_inputs: ?[]const u8 = null,
    selftest_gather: ?[]const u8 = null,
};

const usage =
    "Usage: gemma4_gen_auto <model.safetensors> <tokenizer.json> --prompt \"...\" " ++
    "[--max-tokens N] [--oracle fixture] [--ids-only] [--selftest-inputs f] [--selftest-gather f]";

// Parsing à la main (comme les runners existants, ex. gemma4_gen_long_gpu.zig --no-prealloc) :
// pas de lib de flags ici, juste un balayage séquentiel des positionnels puis des --flags.
// Type EXACT du retour de std.process.Args.toSlice (cf lib/std/process/Args.zig) : une slice
// d'éléments sentinelle-terminés — chaque élément coerce vers []const u8 mais la slice ENTIÈRE
// ne coerce PAS vers []const []const u8 (piège de typage, d'où la signature précise ici).
fn parseArgs(process_args: []const [:0]const u8) !Args {
    if (process_args.len < 3) {
        log.err("{s}", .{usage});
        return error.MissingArgument;
    }
    var args: Args = .{ .ckpt = process_args[1], .tokjson_path = process_args[2] };

    var i: usize = 3;
    while (i < process_args.len) : (i += 1) {
        const a = process_args[i];
        if (std.mem.eql(u8, a, "--prompt")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--prompt attend une valeur", .{});
                return error.MissingArgument;
            }
            args.prompt = process_args[i];
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
        } else if (std.mem.eql(u8, a, "--oracle")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--oracle attend une valeur", .{});
                return error.MissingArgument;
            }
            args.oracle_path = process_args[i];
        } else if (std.mem.eql(u8, a, "--ids-only")) {
            args.ids_only = true;
        } else if (std.mem.eql(u8, a, "--selftest-inputs")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-inputs attend une valeur", .{});
                return error.MissingArgument;
            }
            args.selftest_inputs = process_args[i];
        } else if (std.mem.eql(u8, a, "--selftest-gather")) {
            i += 1;
            if (i >= process_args.len) {
                log.err("--selftest-gather attend une valeur", .{});
                return error.MissingArgument;
            }
            args.selftest_gather = process_args[i];
        } else {
            log.err("argument inconnu: {s}\n{s}", .{ a, usage });
            return error.InvalidArgument;
        }
    }
    return args;
}

// ============================================================================================
// Task 3 — inputs host : cos/sin RoPE full, masques additifs, positions, tables {L_MAX,…}.
// ============================================================================================

// Masques additifs f32 : 0 = visible, -floatMax = masqué (== torch.finfo(float32).min — même
// valeur binaire : le plus grand f32 fini, négé, des deux côtés).
const MASK_MIN: f32 = -std.math.floatMax(f32);

// Coefficients RoPE "proportional" (couches full_attention) — formule COPIÉE de
// transformers/modeling_rope_utils.py::_compute_proportional_rope_parameters (lue sur la 3090,
// 10 juil, cf commit) :
//   head_dim = config.global_head_dim = 512 (head_dim_key="global_head_dim" pour full_attention,
//     modeling_gemma4.py:1098-1099) ; base = rope_theta = 1e6 ; rope_proportion =
//     partial_rotary_factor = 0.25 ; factor = rope_parameters_dict.get("factor", 1.0) = 1.0
//     (absent de rope_scaling.full_attention, confirmé AutoConfig) ;
//   rope_angles = int(rope_proportion * head_dim // 2) = int(0.25*512 // 2) = int(128.0//2) = 64 ;
//   inv_freq_rotated[i] = 1 / base**(arange(0,2*rope_angles,2)[i] / head_dim)
//                       = 1 / base**((2*i)/head_dim)  pour i in 0..rope_angles (64 valeurs) ;
//   nope_angles = head_dim//2 - rope_angles = 256-64 = 192 ;
//   inv_freq = concat(inv_freq_rotated, zeros(nope_angles)) → 256 valeurs (head_dim//2) ;
//   inv_freq /= factor (no-op, factor=1.0).
// forward() (modeling_gemma4.py:1141-1152) : freqs[i] = inv_freq[i] * p (i in 0..256) ;
//   emb = concat(freqs, freqs) — DUPLICATION DE LA MOITIÉ (pas d'entrelacement) → 512 valeurs ;
//   cos = cos(emb) * attention_scaling ; sin = sin(emb) * attention_scaling
//   (attention_scaling = 1.0 pour "proportional" — "Unused in this type of RoPE" — no-op).
const ROPE_FULL_THETA: f32 = 1_000_000.0;
const ROPE_FULL_HEAD_DIM: f32 = 512.0; // = HD_F
const ROPE_FULL_ANGLES: usize = 64; // rope_angles
const ROPE_FULL_HALF: usize = 256; // head_dim // 2 (= HD_F / 2)

// cos/sin full pour la position p — formule ci-dessus (partial 0.25, proportional, theta 1e6).
// PRÉCISION — investigation mesurée (10 juil, fixtures courte p≤68 ET longue p≤1023, cf commits) :
// la transcription EST correcte (vérifiée valeur par valeur contre la fixture ET contre numpy),
// mais un résidu subsiste sur quelques (position, indice de fréquence) précis, QUELLE QUE SOIT la
// stratégie de calcul essayée (f64 bout-en-bout arrondi seulement à la fin ; f32 via
// `std.math.pow` ; f32 via `@exp2(exp*@log2(base))` — même ordre de grandeur les trois fois).
// Root cause : `pow()` de Zig (LLVM/libm, CPU) et de PyTorch arrondissent chacun CORRECTEMENT
// mais PAS IDENTIQUEMENT `base**exp` — 1 ULP d'écart sur certains inv_freq[i] (confirmé
// bit-à-bit : inv_freq[12]≈0.523, bits 3f05f6ee vs 3f05f6ef ; le calcul en précision arbitraire
// montre qu'aucune des deux valeurs n'est "la bonne", les deux sont à ~2 ULP du réel).
// AMPLIFICATION LINÉAIRE EN p (mesurée sur la fixture longue) : l'erreur d'angle vaut
//   Δangle ≈ Δinv_freq×p + arrondi f32 du produit inv_freq×p (±ULP à l'échelle de l'angle,
//   elle-même ∝ p puisque angle ≈ inv_freq×p) ≈ 2 ULP × p × ~6e-8 ≈ 1.2e-7 × p,
// que sin/cos propagent à pente ≤ 1. Vérifié numériquement au pire point mesuré (k=587, p=612,
// i=12, sin) : Δinv_freq = 1 ULP = 5.96e-8 → Δangle = 6.10e-5 (= 3.65e-5 de Δinv×p + arrondis de
// produit opposés, ULP(320 rad) = 3.05e-5), × pente |cos|≈0.983 → 6.00e-5 observé — REPRODUIT
// exactement par numpy f32 sur les deux angles candidats (borne 2-ULP : 7.3e-5, cohérente). Aux
// positions courtes (p≤68) le même mécanisme donnait le plancher 2^-18 = 3.81e-6. C'est un
// plancher de précision float32 inter-implémentations (même famille de piège que "pas de
// bit-à-bit inter-compiles XLA-GPU", mémoire ZML) — PAS une erreur de formule. `std.math.pow`
// est gardé (le plus standard, pas de bricolage ad hoc) ; le SELFTEST compare avec une tolérance
// DÉPENDANTE DE LA POSITION (cf `cosSinTol`), pas une constante.
fn ropeFull(p: i64, cos_out: *[HD_F]f32, sin_out: *[HD_F]f32) void {
    var inv_freq: [ROPE_FULL_HALF]f32 = undefined;
    for (0..ROPE_FULL_HALF) |i| {
        if (i < ROPE_FULL_ANGLES) {
            const exp: f32 = @as(f32, @floatFromInt(2 * i)) / ROPE_FULL_HEAD_DIM;
            inv_freq[i] = 1.0 / std.math.pow(f32, ROPE_FULL_THETA, exp);
        } else {
            inv_freq[i] = 0.0; // nope_angles : pas de rotation (angle constant nul quel que soit p)
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

// Masques additifs f32 : 0 = visible, -floatMax = masqué (== torch.finfo(float32).min).
fn maskRows(p: i64, sliding_out: []f32, full_out: []f32) void {
    const lo = @max(0, p - (SLIDING_WINDOW - 1));
    for (0..@intCast(L_MAX)) |j| {
        const ji: i64 = @intCast(j);
        sliding_out[j] = if (ji > p or ji < lo) MASK_MIN else 0;
        full_out[j] = if (ji > p) MASK_MIN else 0;
    }
}

// Tables host complètes, indexées par STEP == POSITION ABSOLUE p in 0..L_MAX-1 (identité — cf
// PLAN Task 5 : `ctrl.step` vaudra la position courante, et `pickStep(p.cos_full, step)` ira
// chercher la ligne p). Conçues comme des slices host simples (pas de Buffer/Platform) : Task 5
// les enveloppera avec `zml.Buffer.fromBytes` (mêmes shapes que `engine.Packed`/`engine.Cache`).
const HostInputs = struct {
    cos_full: []f32, // {L_MAX, HD_F}
    sin_full: []f32, // {L_MAX, HD_F}
    masks_sliding: []f32, // {L_MAX, L_MAX}
    masks_full: []f32, // {L_MAX, L_MAX}
    positions: []i32, // {L_MAX} = 0..L_MAX-1
    embeds_zero: []u8, // {L_MAX, 1, 1, D} bf16, zéros — factice, non consommé par forwardStep
    embptls_zero: []u8, // {L_MAX, 1, 1, LF} bf16, zéros — idem
    cache_sl_k: []u8, // {NUM_SLIDING_SLOTS, 1, 1, L_MAX, HD_S} f32, zéros
    cache_sl_v: []u8,
    cache_fl_k: []u8, // {NUM_FULL_SLOTS, 1, 1, L_MAX, HD_F} f32, zéros
    cache_fl_v: []u8,

    fn init(allocator: std.mem.Allocator) !HostInputs {
        const l_max: usize = @intCast(L_MAX);
        const hd_f: usize = @intCast(HD_F);

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
        const cache_sl_k = try allocator.alloc(u8, NUM_SLIDING_SLOTS * l_max * @as(usize, @intCast(HD_S)) * 4);
        errdefer allocator.free(cache_sl_k);
        @memset(cache_sl_k, 0);
        const cache_sl_v = try allocator.alloc(u8, NUM_SLIDING_SLOTS * l_max * @as(usize, @intCast(HD_S)) * 4);
        errdefer allocator.free(cache_sl_v);
        @memset(cache_sl_v, 0);
        const cache_fl_k = try allocator.alloc(u8, NUM_FULL_SLOTS * l_max * hd_f * 4);
        errdefer allocator.free(cache_fl_k);
        @memset(cache_fl_k, 0);
        const cache_fl_v = try allocator.alloc(u8, NUM_FULL_SLOTS * l_max * hd_f * 4);
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

// Lit un tenseur ENTIER de la fixture, host-side, SANS Platform : lecture positionnelle directe
// dans le fichier à `tensor.offset` (octets absolus, cf gemma4_gchunk_auto.zig:220-223) sur
// `tensor.byteSize()` octets. Durci : dtype du header vérifié AVANT lecture (une fixture au
// mauvais dtype serait sinon réinterprétée silencieusement), compte d'octets lus vérifié APRÈS
// (fichier tronqué → error.ShortRead, pas des zéros silencieux).
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

// Tolérance cos/sin DÉPENDANTE DE LA POSITION (dérivation complète : note `ropeFull`) — PAS une
// constante : l'erreur d'angle inter-implémentations croît linéairement, Δangle ≲ 2 ULP × p ×
// ULP(inv_freq≈0.5)=6e-8 ≈ 1.2e-7×p, propagée par sin/cos à pente ≤ 1. tol(p) = 1e-5 + 1.5e-7×p
// couvre cette enveloppe avec marge (mesuré : 3.81e-6 @ p≤68 vs tol 2.0e-5 ; 6.00e-5 @ p=612 vs
// tol 1.02e-4 ; borne théorique ~1.2e-4 @ p=1011 vs tol 1.6e-4) sans masquer une VRAIE régression
// de formule (qui produirait des écarts de plusieurs ordres de grandeur, pas ~2 ULP).
fn cosSinTol(p: i32) f32 {
    return 1e-5 + 1.5e-7 * @as(f32, @floatFromInt(p));
}

// --selftest-inputs <fixture> : charge la fixture 49 (gen_custom.safetensors, n_decode steps) et
// compare, pour chaque step k, la position p = positions[k] LUE DE LA FIXTURE :
//   - cos/sin : lignes de la table `HostInputs` à l'index p (donc `ropeFull` ET la construction
//     de la table {L_MAX,…} sont toutes les deux exercées) vs cos_full/sin_full[k] — écart par
//     step ≤ cosSinTol(p) (tolérance position-dépendante, cf dérivation sur `cosSinTol`).
//   - masques : idem vs masks_sliding/masks_full[k] — égalité BIT-EXACTE (valeurs ∈ {0, MASK_MIN}).
//   - positions : continuité (positions[k] == positions[0] + k, i.e. p = seq_len + k avec
//     seq_len = positions[0], lu de la fixture — pas besoin du manifest JSON séparé).
fn selftestInputs(allocator: std.mem.Allocator, io: std.Io, fixture_path: []const u8) !void {
    var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg.deinit();
    var file = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
    defer file.close(io);

    const positions_fx = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "positions");
    defer allocator.free(positions_fx);
    const cos_fx = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "cos_full");
    defer allocator.free(cos_fx);
    const sin_fx = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "sin_full");
    defer allocator.free(sin_fx);
    const masks_sliding_fx = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "masks_sliding");
    defer allocator.free(masks_sliding_fx);
    const masks_full_fx = try readFixtureAlloc(f32, .f32, allocator, io, &reg, &file, "masks_full");
    defer allocator.free(masks_full_fx);

    const n_decode: usize = positions_fx.len;
    const hd_f: usize = @intCast(HD_F);
    const l_max: usize = @intCast(L_MAX);
    if (cos_fx.len != n_decode * hd_f or sin_fx.len != n_decode * hd_f) {
        log.err("SELFTEST INPUTS : shape cos/sin inattendue (cos.len={d} sin.len={d} n_decode*HD_F={d})", .{ cos_fx.len, sin_fx.len, n_decode * hd_f });
        return error.UnexpectedShape;
    }
    if (masks_sliding_fx.len != n_decode * l_max or masks_full_fx.len != n_decode * l_max) {
        log.err("SELFTEST INPUTS : shape masques inattendue (sliding.len={d} full.len={d} n_decode*L_MAX={d})", .{ masks_sliding_fx.len, masks_full_fx.len, n_decode * l_max });
        return error.UnexpectedShape;
    }

    var host = try HostInputs.init(allocator);
    defer host.deinit(allocator);

    var max_abs: f32 = 0;
    var max_ratio: f32 = 0; // max sur les steps de (écart / cosSinTol(p)) — critère de PASS : ≤ 1
    var max_at: struct { k: usize, i: usize, kind: u8, host: f32, fx: f32, tol: f32 } = .{ .k = 0, .i = 0, .kind = 'c', .host = 0, .fx = 0, .tol = 0 };
    var masks_bitexact = true;
    var positions_ok = true;
    const seq_len = positions_fx[0];

    for (0..n_decode) |k| {
        const p = positions_fx[k];
        if (p != seq_len + @as(i32, @intCast(k))) positions_ok = false;
        if (p < 0 or p >= L_MAX) {
            log.err("SELFTEST INPUTS : position hors table à step {d} (p={d})", .{ k, p });
            return error.PositionOutOfRange;
        }
        const pi: usize = @intCast(p);
        const tol = cosSinTol(p);

        const cos_row = host.cos_full[pi * hd_f .. (pi + 1) * hd_f];
        const sin_row = host.sin_full[pi * hd_f .. (pi + 1) * hd_f];
        const cos_fx_row = cos_fx[k * hd_f .. (k + 1) * hd_f];
        const sin_fx_row = sin_fx[k * hd_f .. (k + 1) * hd_f];
        for (0..hd_f) |i| {
            const dc = @abs(cos_row[i] - cos_fx_row[i]);
            const ds = @abs(sin_row[i] - sin_fx_row[i]);
            if (dc > max_abs) max_abs = dc;
            if (ds > max_abs) max_abs = ds;
            if (dc / tol > max_ratio) {
                max_ratio = dc / tol;
                max_at = .{ .k = k, .i = i, .kind = 'c', .host = cos_row[i], .fx = cos_fx_row[i], .tol = tol };
            }
            if (ds / tol > max_ratio) {
                max_ratio = ds / tol;
                max_at = .{ .k = k, .i = i, .kind = 's', .host = sin_row[i], .fx = sin_fx_row[i], .tol = tol };
            }
        }

        const sl_row = host.masks_sliding[pi * l_max .. (pi + 1) * l_max];
        const fl_row = host.masks_full[pi * l_max .. (pi + 1) * l_max];
        const sl_fx_row = masks_sliding_fx[k * l_max .. (k + 1) * l_max];
        const fl_fx_row = masks_full_fx[k * l_max .. (k + 1) * l_max];
        for (0..l_max) |j| {
            if (sl_row[j] != sl_fx_row[j]) masks_bitexact = false;
            if (fl_row[j] != fl_fx_row[j]) masks_bitexact = false;
        }
    }

    const cos_sin_ok = max_ratio <= 1.0;
    if (cos_sin_ok and masks_bitexact and positions_ok) {
        log.info("SELFTEST INPUTS PASS ({d} steps, cos/sin max_abs={e} max_ratio={d:.3} de tol(p), masks bit-exact, positions ==)", .{ n_decode, max_abs, max_ratio });
    } else {
        log.err("SELFTEST INPUTS FAIL — cos/sin max_abs={e} max_ratio={d:.3} (ok={}) masks_bitexact={} positions_ok={}", .{ max_abs, max_ratio, cos_sin_ok, masks_bitexact, positions_ok });
        if (!cos_sin_ok) {
            const p0: usize = @intCast(positions_fx[0]);
            log.err("  1er step : p={d} cos_host[0..8]={any} cos_fx[0..8]={any}", .{ p0, host.cos_full[p0 * hd_f .. p0 * hd_f + 8], cos_fx[0..8] });
            log.err("  1er step : sin_host[0..8]={any} sin_fx[0..8]={any}", .{ host.sin_full[p0 * hd_f .. p0 * hd_f + 8], sin_fx[0..8] });
            log.err("  pire ratio à step k={d} (p={d}) index i={d} kind={c} : host={d} fx={d} écart={e} tol(p)={e}", .{ max_at.k, positions_fx[max_at.k], max_at.i, max_at.kind, max_at.host, max_at.fx, @abs(max_at.host - max_at.fx), max_at.tol });
        }
        return error.SelftestInputsFailed;
    }
}

// ============================================================================================
// Task 4 — gather embeds bruts en streaming (offsets positionnels sur le CHECKPOINT).
// ============================================================================================
//
// Adapté de gemma4_gchunk_auto.zig:137-230 : au lieu de matérialiser embed_tokens (~1,6 Go) et
// embed_tokens_per_layer (~4,7 Go) en RAM, on lit UNE ligne bf16 par token DIRECTEMENT dans le
// fichier `model.safetensors`, à l'offset absolu `tensor.offset + tok*row_bytes` (layout
// row-major : {voc,d} et {voc,lf}). RAW — AUCUN scaling host : les facteurs ×√1536 (embed_tokens,
// cf `engine.zig` EMBED_SCALE/`.scale(EMBED_SCALE)`) et ×16 (embed_tokens_per_layer, PLE) sont
// appliqués IN-GRAPH par `forwardStep` (engine.zig:644-645) — les appliquer ici causerait un
// double-scaling silencieux (spec DESIGN §3.3).
const EMB_KEY = "model.language_model.embed_tokens.weight";
const EPTL_KEY = "model.language_model.embed_tokens_per_layer.weight";

// Table d'embeddings en streaming : ouvre le checkpoint une fois, résout les offsets/dtypes des
// deux tables via le registry (header seul — pas de chargement de poids), puis `gather(tok)`
// relit une ligne de chacune dans des scratch buffers réutilisés (pas d'alloc en boucle).
//
// Conçu pour Task 5 : `emb_scratch`/`eptl_scratch` ([]u16) sont déjà EXACTEMENT les bits bruts
// d'une ligne {d}/{lf} bf16 (B=S=1) — Task 5 n'aura qu'à les envelopper avec
// `zml.Buffer.fromBytes(io, platform, shape{1,1,D|LF}.bf16, sharding, std.mem.sliceAsBytes(scratch))`
// par step, comme gchunk_auto.zig:226-227 le fait déjà pour son propre scratch (là en []u8).
const EmbedGather = struct {
    file: std.Io.File,
    emb_base_off: u64, // offset absolu (octets) du début de la table embed_tokens
    eptl_base_off: u64, // idem embed_tokens_per_layer
    emb_row_bytes: usize, // = D * sizeOf(bf16)
    eptl_row_bytes: usize, // = LF * sizeOf(bf16)
    num_rows: i64, // = vocab, DÉRIVÉ du header (byteSize/row_bytes) — borne du bounds-check de gather
    // Scratch typés []u16 (bits bruts bf16), PAS []u8 : reformer un []u16 depuis des octets bruts
    // exige un pointeur aligné à 2, ce qu'un []u8 alloué par le GPA ne garantit pas — l'allouer
    // directement en []u16 lève le problème à la racine (élément naturellement aligné) tout en
    // restant directement comparable en bits aux lectures fixture (elles aussi []u16, cf
    // readFixtureAlloc(u16, .bf16, ...)). readPositionalAll reçoit `std.mem.sliceAsBytes(...)`
    // (u16→u8 : toujours valide, alignement plus faible).
    emb_scratch: []u16, // {D} bf16 brut — réutilisé à chaque step
    eptl_scratch: []u16, // {LF} bf16 brut — idem

    fn init(allocator: std.mem.Allocator, io: std.Io, ckpt_path: []const u8) !EmbedGather {
        var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt_path);
        defer reg.deinit();
        const emb_t = reg.tensors.get(EMB_KEY) orelse {
            log.err("tensor introuvable dans le checkpoint: {s}", .{EMB_KEY});
            return error.MissingTensor;
        };
        const eptl_t = reg.tensors.get(EPTL_KEY) orelse {
            log.err("tensor introuvable dans le checkpoint: {s}", .{EPTL_KEY});
            return error.MissingTensor;
        };
        const emb_dtype = emb_t.shape.dtype();
        const eptl_dtype = eptl_t.shape.dtype();
        if (emb_dtype != .bf16 or eptl_dtype != .bf16) {
            log.err("dtype embeddings inattendu (emb={s} eptl={s}, attendu bf16 les deux)", .{ @tagName(emb_dtype), @tagName(eptl_dtype) });
            return error.DtypeMismatch;
        }
        const emb_row_bytes: usize = @intCast(D * @as(i64, @intCast(emb_dtype.sizeOf())));
        const eptl_row_bytes: usize = @intCast(LF * @as(i64, @intCast(eptl_dtype.sizeOf())));

        // Géométrie des tables VALIDÉE contre le header : byteSize doit être un multiple exact de
        // row_bytes (sinon D/LF hardcodés ≠ shape réelle du checkpoint), et les deux comptes de
        // lignes doivent coïncider (même vocab). Garde-fou de revue : sans cette dérivation, un
        // checkpoint à la géométrie inattendue produirait des lignes décalées silencieusement.
        const emb_bytes: u64 = emb_t.byteSize();
        const eptl_bytes: u64 = eptl_t.byteSize();
        if (emb_bytes % @as(u64, emb_row_bytes) != 0 or eptl_bytes % @as(u64, eptl_row_bytes) != 0) {
            log.err("géométrie table invalide : emb {d} % {d} = {d}, eptl {d} % {d} = {d} (D/LF hardcodés ≠ checkpoint ?)", .{ emb_bytes, emb_row_bytes, emb_bytes % @as(u64, emb_row_bytes), eptl_bytes, eptl_row_bytes, eptl_bytes % @as(u64, eptl_row_bytes) });
            return error.BadTableGeometry;
        }
        const emb_rows: u64 = emb_bytes / @as(u64, emb_row_bytes);
        const eptl_rows: u64 = eptl_bytes / @as(u64, eptl_row_bytes);
        if (emb_rows != eptl_rows) {
            log.err("géométrie table invalide : emb {d} lignes ≠ eptl {d} lignes (vocab incohérent)", .{ emb_rows, eptl_rows });
            return error.BadTableGeometry;
        }
        log.info("EmbedGather : {d} lignes/table (vocab), emb {d} o/ligne, eptl {d} o/ligne", .{ emb_rows, emb_row_bytes, eptl_row_bytes });

        var file = try std.Io.Dir.cwd().openFile(io, ckpt_path, .{ .mode = .read_only });
        errdefer file.close(io);
        const emb_scratch = try allocator.alloc(u16, emb_row_bytes / 2);
        errdefer allocator.free(emb_scratch);
        const eptl_scratch = try allocator.alloc(u16, eptl_row_bytes / 2);
        errdefer allocator.free(eptl_scratch);

        return .{
            .file = file,
            .emb_base_off = emb_t.offset,
            .eptl_base_off = eptl_t.offset,
            .emb_row_bytes = emb_row_bytes,
            .eptl_row_bytes = eptl_row_bytes,
            .num_rows = @intCast(emb_rows),
            .emb_scratch = emb_scratch,
            .eptl_scratch = eptl_scratch,
        };
    }

    fn deinit(self: *EmbedGather, io: std.Io, allocator: std.mem.Allocator) void {
        allocator.free(self.emb_scratch);
        allocator.free(self.eptl_scratch);
        self.file.close(io);
    }

    // Lit la ligne du token `tok` dans les deux tables (read positionnel, offset absolu = base +
    // tok*row_bytes — cf gchunk_auto.zig:220-223) — RAW, aucun scaling. Résultat dans
    // self.emb_scratch/self.eptl_scratch (réutilisés, PAS de nouvelle alloc).
    fn gather(self: *EmbedGather, io: std.Io, tok: i64) !void {
        // Bounds-check EXPLICITE (revue) : tok ≥ vocab tomberait dans le tenseur SUIVANT du
        // checkpoint — la lecture réussirait (pas de ShortRead) et une ligne plausible mais
        // FAUSSE partirait dans le forward = divergence silencieuse, exactement la classe
        // d'erreur que la méthodo de ce repo existe pour empêcher. tok < 0 ne panique qu'en
        // ReleaseSafe (mode de build non épinglé) → check explicite obligatoire.
        if (tok < 0 or tok >= self.num_rows) {
            log.err("gather: token hors table — tok={d}, bornes [0, {d})", .{ tok, self.num_rows });
            return error.TokenOutOfRange;
        }
        const emb_off: u64 = self.emb_base_off + @as(u64, @intCast(tok)) * @as(u64, @intCast(self.emb_row_bytes));
        const eptl_off: u64 = self.eptl_base_off + @as(u64, @intCast(tok)) * @as(u64, @intCast(self.eptl_row_bytes));
        const got_emb = try self.file.readPositionalAll(io, std.mem.sliceAsBytes(self.emb_scratch), emb_off);
        if (got_emb != self.emb_row_bytes) {
            log.err("gather embeds: lecture courte tok={d} — {d}/{d} octets (checkpoint tronqué ?)", .{ tok, got_emb, self.emb_row_bytes });
            return error.ShortRead;
        }
        const got_eptl = try self.file.readPositionalAll(io, std.mem.sliceAsBytes(self.eptl_scratch), eptl_off);
        if (got_eptl != self.eptl_row_bytes) {
            log.err("gather embptls: lecture courte tok={d} — {d}/{d} octets (checkpoint tronqué ?)", .{ tok, got_eptl, self.eptl_row_bytes });
            return error.ShortRead;
        }
    }
};

// --selftest-gather <fixture> : pour chaque step k de la fixture A1 (fed[k] = token FED à ce
// step, cf 49_gen_custom_oracle.py:159/174-177), gather(fed[k]) sur le CHECKPOINT doit être
// BIT-EXACT à embeds[k]/embptls[k] de la fixture — les deux sont la même ligne bf16 brute
// (`emb_w[tid].to(bfloat16)` / `eptl_w[tid].view(...).to(bfloat16)`, NON re-scalée,
// 49_gen_custom_oracle.py:176-178). Comparaison en u16 bruts (bf16 = 2 octets, pas de
// tolérance) : c'est le SEUL garde-fou contre un scaling host accidentel tant que le gate A1
// (bout-en-bout, Task 5) n'est pas passé — ne JAMAIS l'affaiblir en tolérance (piège relevé en
// revue).
fn selftestGather(allocator: std.mem.Allocator, io: std.Io, ckpt_path: []const u8, fixture_path: []const u8) !void {
    var gather_tbl = try EmbedGather.init(allocator, io, ckpt_path);
    defer gather_tbl.deinit(io, allocator);

    var reg: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture_path);
    defer reg.deinit();
    var file = try std.Io.Dir.cwd().openFile(io, fixture_path, .{ .mode = .read_only });
    defer file.close(io);

    // `fed` est i32 dans la fixture (cf 49_gen_custom_oracle.py:203 : torch.int32) ; le gather
    // canonicalise en i64 (comme le sibling gchunk_auto) — cast explicite à la frontière.
    const fed = try readFixtureAlloc(i32, .i32, allocator, io, &reg, &file, "fed");
    defer allocator.free(fed);
    const embeds_fx = try readFixtureAlloc(u16, .bf16, allocator, io, &reg, &file, "embeds");
    defer allocator.free(embeds_fx);
    const embptls_fx = try readFixtureAlloc(u16, .bf16, allocator, io, &reg, &file, "embptls");
    defer allocator.free(embptls_fx);

    const n_decode = fed.len;
    if (n_decode == 0) {
        log.err("SELFTEST GATHER : fixture vide (fed.len=0) — un PASS à 0 step serait vacueux", .{});
        return error.EmptyFixture;
    }
    const d: usize = @intCast(D);
    const lf: usize = @intCast(LF);
    if (embeds_fx.len != n_decode * d) {
        log.err("SELFTEST GATHER : shape embeds inattendue (embeds.len={d} n_decode*D={d})", .{ embeds_fx.len, n_decode * d });
        return error.UnexpectedShape;
    }
    if (embptls_fx.len != n_decode * lf) {
        log.err("SELFTEST GATHER : shape embptls inattendue (embptls.len={d} n_decode*LF={d})", .{ embptls_fx.len, n_decode * lf });
        return error.UnexpectedShape;
    }

    for (0..n_decode) |k| {
        const tok: i64 = @intCast(fed[k]);
        try gather_tbl.gather(io, tok);
        const emb_fx_row = embeds_fx[k * d .. (k + 1) * d];
        const eptl_fx_row = embptls_fx[k * lf .. (k + 1) * lf];
        if (!std.mem.eql(u16, gather_tbl.emb_scratch, emb_fx_row)) {
            log.err("SELFTEST GATHER FAIL — embeds diverge au step {d} (tok={d}) : gathered[0..4]={any} fixture[0..4]={any}", .{ k, tok, gather_tbl.emb_scratch[0..4], emb_fx_row[0..4] });
            return error.SelftestGatherFailed;
        }
        if (!std.mem.eql(u16, gather_tbl.eptl_scratch, eptl_fx_row)) {
            log.err("SELFTEST GATHER FAIL — embptls diverge au step {d} (tok={d}) : gathered[0..4]={any} fixture[0..4]={any}", .{ k, tok, gather_tbl.eptl_scratch[0..4], eptl_fx_row[0..4] });
            return error.SelftestGatherFailed;
        }
    }

    log.info("SELFTEST GATHER PASS ({d} steps, bit-exact)", .{n_decode});
}

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000); // piège quota comptime (cf gemma4_gchunk_auto.zig:96)
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const args = try parseArgs(process_args);

    // === Task 3 : --selftest-inputs — indépendant du prompt/tokenizer/poids (fixture only) ===
    if (args.selftest_inputs) |fixture_path| {
        try selftestInputs(allocator, io, fixture_path);
        return;
    }

    // === Task 4 : --selftest-gather — host-only (registry + reads positionnels), pas de
    // Platform/device requis (device buffers = Task 5). args.ckpt = checkpoint, positionnel 1. ===
    if (args.selftest_gather) |fixture_path| {
        try selftestGather(allocator, io, args.ckpt, fixture_path);
        return;
    }

    const prompt_text = args.prompt orelse {
        log.err("--prompt est requis\n{s}", .{usage});
        return error.MissingArgument;
    };

    // === Gate A0 : tokenizer ZML natif + chat template Zig ===
    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, args.tokjson_path);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const rendered = try renderChatTemplate(arena.allocator(), prompt_text);
    var prompt_tok = try encoder.encodeAlloc(allocator, rendered);
    defer prompt_tok.deinit(allocator);

    var ids: std.ArrayList(u32) = try .initCapacity(allocator, prompt_tok.items.len + 1);
    defer ids.deinit(allocator);
    try ids.append(allocator, BOS_ID);
    try ids.appendSlice(allocator, prompt_tok.items);

    if (args.ids_only) {
        log.info("ids = {any}", .{ids.items});

        // Round-trip détok (Step 2.4) : decode les ids APRÈS bos (= prompt_tok.items, la partie
        // produite par l'encoder, gabarit de chat INCLUS — pas seulement le texte user) puis
        // re-encode. Plus fort que le PLAN Step 2.4 (qui n'exigeait que le prompt hors template) :
        // on assume la déviation, le round-trip couvre aussi les tokens de tour <|turn>/<turn|>.
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        var text_rt = try decoder.decodeAlloc(allocator, ids.items[1..]);
        defer text_rt.deinit(allocator);

        // reset() avant réutilisation : l'encoder iree est un automate à état (encode_state_t) ;
        // finalize() ne remet pas AT_INPUT_START, réutiliser encoder sans reset risquerait de
        // faire fuiter l'état du 1er encodage dans le round-trip.
        encoder.reset();
        var reenc = try encoder.encodeAlloc(allocator, text_rt.items);
        defer reenc.deinit(allocator);

        const round_trip_ok = std.mem.eql(u32, reenc.items, prompt_tok.items);
        if (round_trip_ok) {
            log.info("round-trip détok : PASS (decode -> re-encode == ids)", .{});
        } else {
            log.err("round-trip détok : FAIL — got={any} want={any}", .{ reenc.items, prompt_tok.items });
            return error.RoundTripFailed;
        }
        return;
    }

    // Task 5+ : boucle autonome prefill-par-decode (forwardStep, device buffers per-step),
    // --oracle, --max-tokens. Pas encore câblés (Task 3 = inputs host, Task 4 = gather streaming +
    // --selftest-gather, gate A0 déjà livrée) — le parsing ci-dessus est prêt à être branché,
    // args.ckpt inclus (pas encore chargé sur device).
    log.err("génération non implémentée (Task 5+) — utiliser --ids-only (A0), --selftest-inputs (Task 3) ou --selftest-gather (Task 4)", .{});
    return error.NotImplemented;
}
