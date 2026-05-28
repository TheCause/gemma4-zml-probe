// P5.2.A — Policy lookup ZML host-side, sans attention math.
//
// Objectif : prouver que le runtime Zig consomme la policy table YOCO P5.1
// (fixtures/yoco_policy_table.json) et re-derive la meme table en independant.
//
// Entree CLI : <path-to-yoco_policy_table.json>
//
// Sortie :
//   1. 7 cas fixes (layer_idx -> {layer_type, is_reader, target_kv_layer})
//   2. Validation table 35 entrees, Zig recompute vs JSON oracle.
//
// Interdits stricts P5.2.A :
//   - QKV / RoPE / matmul attention / cache reel
//   - aucune dependance //zml (pas de Tensor, pas de graph)
//
// Logique recompute (miroir verbatim Transformers + vLLM, cf P5.0 § 6.2) :
//   first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers
//   is_reader = (num_kv_shared_layers > 0) and (i >= first)
//   target = i si producer, sinon last j < first avec layer_types[j] == layer_types[i]

const std = @import("std");
const log = std.log;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const LayerType = enum {
    full_attention,
    sliding_attention,

    fn fromStr(s: []const u8) ?LayerType {
        if (std.mem.eql(u8, s, "full_attention")) return .full_attention;
        if (std.mem.eql(u8, s, "sliding_attention")) return .sliding_attention;
        return null;
    }

    fn toStr(self: LayerType) []const u8 {
        return switch (self) {
            .full_attention => "full_attention",
            .sliding_attention => "sliding_attention",
        };
    }
};

const PolicyEntry = struct {
    layer_idx: u32,
    layer_type: LayerType,
    is_reader: bool,
    target_kv_layer: u32,
};

const Policy = struct {
    num_hidden_layers: u32,
    num_kv_shared_layers: u32,
    first_kv_shared_layer_idx: u32,
    layer_types: []const LayerType,
    entries: []PolicyEntry,

    /// Recompute pure Zig depuis (n, k, layer_types). Miroir verbatim de
    /// `compute_policy_table` (scripts/13_yoco_policy_table.py) et de
    /// `Gemma4TextAttention.__init__` (modeling_gemma4.py:777-782) /
    /// `Gemma4Attention.__init__` (vllm gemma4.py:469-489).
    fn build(
        allocator: std.mem.Allocator,
        n: u32,
        k: u32,
        layer_types: []const LayerType,
    ) !Policy {
        if (layer_types.len != n) return error.LayerTypesLenMismatch;
        const first: u32 = if (k <= n) n - k else return error.InvalidConfig;

        const entries = try allocator.alloc(PolicyEntry, n);
        errdefer allocator.free(entries);

        for (layer_types, 0..) |t, i_usize| {
            const i: u32 = @intCast(i_usize);
            const is_reader = (k > 0) and (i >= first);
            var target: u32 = i;
            if (is_reader) {
                // last j < first avec layer_types[j] == t
                var found = false;
                var j: u32 = first;
                while (j > 0) {
                    j -= 1;
                    if (layer_types[j] == t) {
                        target = j;
                        found = true;
                        break;
                    }
                }
                if (!found) return error.NoProducerOfType;
            }
            entries[i_usize] = .{
                .layer_idx = i,
                .layer_type = t,
                .is_reader = is_reader,
                .target_kv_layer = target,
            };
        }

        return .{
            .num_hidden_layers = n,
            .num_kv_shared_layers = k,
            .first_kv_shared_layer_idx = first,
            .layer_types = layer_types,
            .entries = entries,
        };
    }

    fn deinit(self: *Policy, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        self.entries = &[_]PolicyEntry{};
    }

    /// API runtime : input = layer_idx, output = PolicyEntry.
    /// O(1), pure lookup, aucun calcul. Pour P5.2.B+, c'est ce que
    /// l'attention forward consultera avant de choisir le path producer/reader.
    fn lookup(self: Policy, layer_idx: u32) PolicyEntry {
        std.debug.assert(layer_idx < self.num_hidden_layers);
        return self.entries[layer_idx];
    }
};

/// Parse partielle de la fixture P5.1 — on garde la table + le minimum config.
const JsonEntry = struct {
    layer_idx: u32,
    layer_type: []const u8,
    is_reader: bool,
    target_kv_layer: u32,
};

const FixedCase = struct {
    layer_idx: u32,
    label: []const u8,
};

const FIXED_CASES = [_]FixedCase{
    .{ .layer_idx = 0, .label = "sliding producer (own)" },
    .{ .layer_idx = 4, .label = "full producer (own)" },
    .{ .layer_idx = 13, .label = "sliding writer" },
    .{ .layer_idx = 14, .label = "full writer" },
    .{ .layer_idx = 15, .label = "sliding reader -> target 13" },
    .{ .layer_idx = 19, .label = "full reader -> target 14" },
    .{ .layer_idx = 34, .label = "last full reader -> target 14" },
};

fn parseFixture(allocator: std.mem.Allocator, json_text: []const u8) !struct {
    num_hidden_layers: u32,
    num_kv_shared_layers: u32,
    first_kv_shared_layer_idx: u32,
    layer_types: []LayerType,
    json_entries: []JsonEntry,
    parsed: std.json.Parsed(std.json.Value),
} {
    var parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json_text,
        .{ .allocate = .alloc_always },
    );
    errdefer parsed.deinit();

    const root = parsed.value.object;
    const config = root.get("config").?.object;
    const n: u32 = @intCast(config.get("num_hidden_layers").?.integer);
    const k: u32 = @intCast(config.get("num_kv_shared_layers").?.integer);
    const first: u32 = @intCast(config.get("first_kv_shared_layer_idx").?.integer);

    const table_arr = root.get("table").?.array;
    if (table_arr.items.len != n) return error.TableLenMismatch;

    const layer_types = try allocator.alloc(LayerType, n);
    errdefer allocator.free(layer_types);

    const entries = try allocator.alloc(JsonEntry, n);
    errdefer allocator.free(entries);

    for (table_arr.items, 0..) |item, i| {
        const obj = item.object;
        const li: u32 = @intCast(obj.get("layer_idx").?.integer);
        const lt_str = obj.get("layer_type").?.string;
        const is_reader = obj.get("is_reader").?.bool;
        const tgt: u32 = @intCast(obj.get("target_kv_layer").?.integer);

        if (li != i) return error.LayerIdxOutOfOrder;
        const lt = LayerType.fromStr(lt_str) orelse return error.UnknownLayerType;
        layer_types[i] = lt;
        entries[i] = .{
            .layer_idx = li,
            .layer_type = lt_str,
            .is_reader = is_reader,
            .target_kv_layer = tgt,
        };
    }

    return .{
        .num_hidden_layers = n,
        .num_kv_shared_layers = k,
        .first_kv_shared_layer_idx = first,
        .layer_types = layer_types,
        .json_entries = entries,
        .parsed = parsed,
    };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_policy_lookup <path-to-yoco_policy_table.json>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.A — policy lookup ZML host-side (no attention math)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    const dir = std.Io.Dir.cwd();
    const file = try dir.openFile(io, fixture_path, .{});
    defer file.close(io);

    const len_u64 = try file.length(io);
    const len: usize = @intCast(len_u64);
    const json_text = try allocator.alloc(u8, len);
    defer allocator.free(json_text);

    const n_read = try file.readPositionalAll(io, json_text, 0);
    if (n_read != len) {
        log.err("short read on fixture: got {d} expected {d}", .{ n_read, len });
        return error.ShortRead;
    }

    var fixture = try parseFixture(allocator, json_text);
    defer {
        allocator.free(fixture.layer_types);
        allocator.free(fixture.json_entries);
        fixture.parsed.deinit();
    }

    log.info("  num_hidden_layers         : {}", .{fixture.num_hidden_layers});
    log.info("  num_kv_shared_layers      : {}", .{fixture.num_kv_shared_layers});
    log.info("  first_kv_shared_layer_idx : {}", .{fixture.first_kv_shared_layer_idx});

    // === Build Zig policy ===
    var policy = try Policy.build(
        allocator,
        fixture.num_hidden_layers,
        fixture.num_kv_shared_layers,
        fixture.layer_types,
    );
    defer policy.deinit(allocator);

    if (policy.first_kv_shared_layer_idx != fixture.first_kv_shared_layer_idx) {
        log.err(
            "first_kv_shared_layer_idx mismatch : zig={} json={}",
            .{ policy.first_kv_shared_layer_idx, fixture.first_kv_shared_layer_idx },
        );
        return error.FirstKvSharedMismatch;
    }

    // === Pretty-print 7 cas fixes ===
    log.info("", .{});
    log.info("Fixed-case lookup table:", .{});
    log.info(
        "  layer_idx | layer_type        | is_reader | target_kv_layer | label",
        .{},
    );
    log.info(
        "  ----------|-------------------|-----------|-----------------|------",
        .{},
    );
    for (FIXED_CASES) |c| {
        const e = policy.lookup(c.layer_idx);
        log.info("  {d:>9} | {s:<17} | {s:<9} | {d:>15} | {s}", .{
            e.layer_idx,
            e.layer_type.toStr(),
            if (e.is_reader) "true" else "false",
            e.target_kv_layer,
            c.label,
        });
    }

    // === Validation pleine table 35 entrees Zig vs JSON ===
    log.info("", .{});
    log.info("Validating Zig recompute against JSON oracle (35 entries):", .{});

    var fails: u32 = 0;
    for (policy.entries, fixture.json_entries) |z, j| {
        var local_fail = false;
        if (z.layer_idx != j.layer_idx) {
            log.err("  [layer {d}] layer_idx zig={d} json={d}", .{ j.layer_idx, z.layer_idx, j.layer_idx });
            local_fail = true;
        }
        if (!std.mem.eql(u8, z.layer_type.toStr(), j.layer_type)) {
            log.err("  [layer {d}] layer_type zig={s} json={s}", .{ j.layer_idx, z.layer_type.toStr(), j.layer_type });
            local_fail = true;
        }
        if (z.is_reader != j.is_reader) {
            log.err("  [layer {d}] is_reader zig={} json={}", .{ j.layer_idx, z.is_reader, j.is_reader });
            local_fail = true;
        }
        if (z.target_kv_layer != j.target_kv_layer) {
            log.err(
                "  [layer {d}] target_kv_layer zig={d} json={d}",
                .{ j.layer_idx, z.target_kv_layer, j.target_kv_layer },
            );
            local_fail = true;
        }
        if (local_fail) fails += 1;
    }

    if (fails > 0) {
        log.err("BLOCK: {d}/35 entries differ between Zig recompute and JSON", .{fails});
        return error.PolicyMismatch;
    }
    log.info("  35/35 entries match (Zig recompute == JSON oracle)", .{});

    // === Sanity invariants additionnelles ===
    var producers: u32 = 0;
    var readers: u32 = 0;
    var full_writer: ?u32 = null;
    var sliding_writer: ?u32 = null;
    for (policy.entries) |e| {
        if (e.is_reader) {
            readers += 1;
            const expected_target: u32 = switch (e.layer_type) {
                .full_attention => 14,
                .sliding_attention => 13,
            };
            if (e.target_kv_layer != expected_target) {
                log.err(
                    "invariant: reader {d} ({s}) target {d} != {d}",
                    .{ e.layer_idx, e.layer_type.toStr(), e.target_kv_layer, expected_target },
                );
                return error.ReaderTargetMismatch;
            }
        } else {
            producers += 1;
            if (e.target_kv_layer != e.layer_idx) {
                log.err(
                    "invariant: producer {d} target {d} != self",
                    .{ e.layer_idx, e.target_kv_layer },
                );
                return error.ProducerNotIdentity;
            }
            switch (e.layer_type) {
                .full_attention => full_writer = e.layer_idx,
                .sliding_attention => sliding_writer = e.layer_idx,
            }
        }
    }

    log.info("", .{});
    log.info("Sanity invariants:", .{});
    log.info("  producers         : {d} (expect 15)", .{producers});
    log.info("  readers           : {d} (expect 20)", .{readers});
    log.info("  full writer       : {?} (expect 14)", .{full_writer});
    log.info("  sliding writer    : {?} (expect 13)", .{sliding_writer});

    if (producers != 15 or readers != 20) return error.ProducerReaderCountMismatch;
    if (full_writer == null or full_writer.? != 14) return error.FullWriterMismatch;
    if (sliding_writer == null or sliding_writer.? != 13) return error.SlidingWriterMismatch;

    log.info("", .{});
    log.info("P5.2.A PASS: policy lookup ZML host-side validated end-to-end", .{});
    log.info("  (no QKV, no RoPE, no matmul, no cache — pure routing table)", .{});
}
