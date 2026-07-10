// Runtime AUTONOME texte→texte (spec docs/GEN_AUTONOME_DESIGN.md).
// Gates : A0 tokenizer+template ; A1 prefill-par-decode 48/48 ; A2 long N/N ; A3 early-stop EOS.
// Le moteur engine.zig est INTACT — entrée compilée : forwardStep (embeds host token-dépendants).
//
// CLI : gemma4_gen_auto <model.safetensors> <tokenizer.json> --prompt "..." [--max-tokens N]
//       [--oracle fixture] [--ids-only] [--selftest-inputs f] [--selftest-gather f]
// Cette tranche (Task 2, gate A0) implémente : parsing CLI, chargement tokenizer ZML natif,
// rendu du chat template, encodage, préfixage BOS explicite, et le mode `--ids-only`
// (log des ids finaux + round-trip détok, puis exit). Les autres flags sont parsés mais
// stubbés (Task 3+ les branchera sur le moteur).
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
            args.max_tokens = try std.fmt.parseInt(usize, process_args[i], 10);
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

pub fn main(init: std.process.Init) !void {
    @setEvalBranchQuota(200000); // piège quota comptime (cf gemma4_gchunk_auto.zig:96)
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const args = try parseArgs(process_args);
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
        // produite par l'encoder) puis re-encode ; exclure le gabarit du template n'est PAS requis
        // (decode tout ce qui suit BOS suffit, cf PLAN Step 2.4).
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
        }
        return;
    }

    // Task 3+ : inputs host (cos/sin/masques/cache), boucle prefill-par-decode, --oracle, --max-tokens,
    // --selftest-inputs, --selftest-gather. Pas encore câblés dans ce squelette (gate A0 seule) — le
    // parsing ci-dessus est prêt à être branché, args.ckpt inclus (pas encore chargé, cf Task 3+).
    log.err("génération non implémentée (Task 3+) — utiliser --ids-only pour la gate A0", .{});
    return error.NotImplemented;
}
