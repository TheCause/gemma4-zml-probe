// P5.2.B — Producer/read routing mock host-side.
//
// Objectif : valider que la policy table (oracle JSON, validee en P5.2.A)
// resoud correctement vers des slots KV factices producer_kv[15], sans
// allouer le moindre cache reel ni faire aucun calcul Q/K/V.
//
// Entree CLI : <path-to-yoco_policy_table.json>
//
// Conception :
//   - producer_kv = [_]FakeKvSlot{ .{0,1000}, .{1,1001}, ..., .{14,1014} }
//   - Pour chaque entry du JSON (35 layers) : slot = producer_kv[entry.target_kv_layer]
//   - Verifie :
//       * slot.producer_layer == entry.target_kv_layer (sanite mock)
//       * producers 0..14 : !is_reader et slot.producer_layer == layer_idx
//       * readers sliding : is_reader et slot.producer_layer == 13
//       * readers full    : is_reader et slot.producer_layer == 14
//   - Aggregation finale :
//       * 15/15 producer self-routes
//       * 16/16 sliding reader -> 13
//       * 4/4  full reader -> 14
//
// Interdits stricts P5.2.B :
//   - Q/K/V projection
//   - RoPE
//   - attention matmul, scores, softmax
//   - sliding mask
//   - cache reel (uniquement FakeKvSlot opaque)
//
// Note : reutilise la lecture JSON et les patterns Zig 0.16-dev capitalises
//        en P5.2.A. Pas d'import croise du runner P5.2.A — sous-gates
//        independants par design.

const std = @import("std");
const log = std.log;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const FakeKvSlot = struct {
    producer_layer: u8,
    marker: u32,
};

/// 15 slots factices pre-construits a l'init. Aucune allocation cache reelle.
/// marker = 1000 + producer_layer pour traceabilite.
fn buildProducerKv() [15]FakeKvSlot {
    var slots: [15]FakeKvSlot = undefined;
    for (&slots, 0..) |*s, i| {
        s.* = .{
            .producer_layer = @intCast(i),
            .marker = @intCast(1000 + i),
        };
    }
    return slots;
}

const RoutingMode = enum {
    producer_self,
    reader_shared,
};

const ResolvedEntry = struct {
    layer_idx: u32,
    layer_type: []const u8,
    is_reader: bool,
    target_kv_layer: u32,
    resolved_slot_producer_layer: u8,
    resolved_slot_marker: u32,
    mode: RoutingMode,
};

const FixedCase = struct {
    layer_idx: u32,
    expected_slot: u8,
    label: []const u8,
};

const FIXED_CASES = [_]FixedCase{
    .{ .layer_idx = 0, .expected_slot = 0, .label = "producer sliding -> slot 0" },
    .{ .layer_idx = 4, .expected_slot = 4, .label = "producer full -> slot 4" },
    .{ .layer_idx = 13, .expected_slot = 13, .label = "producer sliding writer -> slot 13" },
    .{ .layer_idx = 14, .expected_slot = 14, .label = "producer full writer -> slot 14" },
    .{ .layer_idx = 15, .expected_slot = 13, .label = "reader sliding -> slot 13" },
    .{ .layer_idx = 18, .expected_slot = 13, .label = "reader sliding -> slot 13" },
    .{ .layer_idx = 19, .expected_slot = 14, .label = "reader full -> slot 14" },
    .{ .layer_idx = 24, .expected_slot = 14, .label = "reader full -> slot 14" },
    .{ .layer_idx = 34, .expected_slot = 14, .label = "reader full -> slot 14" },
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_routing_mock <path-to-yoco_policy_table.json>", .{});
        return error.MissingArgument;
    }
    const fixture_path = process_args[1];

    log.info("P5.2.B — producer/read routing mock (no attention math, no Q/K/V)", .{});
    log.info("Loading fixture from {s}", .{fixture_path});

    // === Read JSON fixture ===
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

    var parsed = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        json_text,
        .{ .allocate = .alloc_always },
    );
    defer parsed.deinit();

    const root = parsed.value.object;
    const config = root.get("config").?.object;
    const num_hidden_layers: u32 = @intCast(config.get("num_hidden_layers").?.integer);
    const first_kv_shared_layer_idx: u32 = @intCast(config.get("first_kv_shared_layer_idx").?.integer);

    const table_arr = root.get("table").?.array;
    if (table_arr.items.len != num_hidden_layers) return error.TableLenMismatch;

    log.info("  num_hidden_layers         : {d}", .{num_hidden_layers});
    log.info("  first_kv_shared_layer_idx : {d}", .{first_kv_shared_layer_idx});

    // === Build mock producer_kv[15] ===
    const producer_kv = buildProducerKv();
    log.info("", .{});
    log.info("Mock producer_kv[15] (no real cache allocation, opaque slots):", .{});
    for (producer_kv) |s| {
        log.info("  slot[{d:>2}] = .{{ .producer_layer = {d:>2}, .marker = {d} }}", .{
            s.producer_layer,
            s.producer_layer,
            s.marker,
        });
    }
    if (producer_kv.len != 15) return error.ProducerKvSizeMismatch;
    if (first_kv_shared_layer_idx != 15) {
        log.err("expected first_kv_shared_layer_idx=15, got {d}", .{first_kv_shared_layer_idx});
        return error.FirstKvSharedMismatch;
    }

    // === Resolve each layer via JSON entry -> producer_kv ===
    const resolved = try allocator.alloc(ResolvedEntry, num_hidden_layers);
    defer allocator.free(resolved);

    for (table_arr.items, 0..) |item, i| {
        const obj = item.object;
        const li: u32 = @intCast(obj.get("layer_idx").?.integer);
        const lt = obj.get("layer_type").?.string;
        const is_reader = obj.get("is_reader").?.bool;
        const tgt: u32 = @intCast(obj.get("target_kv_layer").?.integer);

        if (li != i) return error.LayerIdxOutOfOrder;
        if (tgt >= producer_kv.len) {
            log.err("[layer {d}] target_kv_layer {d} out of producer_kv bounds (<{d})", .{ li, tgt, producer_kv.len });
            return error.TargetOutOfBounds;
        }

        const slot = producer_kv[tgt];
        const mode: RoutingMode = if (is_reader) .reader_shared else .producer_self;
        resolved[i] = .{
            .layer_idx = li,
            .layer_type = lt,
            .is_reader = is_reader,
            .target_kv_layer = tgt,
            .resolved_slot_producer_layer = slot.producer_layer,
            .resolved_slot_marker = slot.marker,
            .mode = mode,
        };
    }

    // === Pretty-print 9 cas fixes ===
    log.info("", .{});
    log.info("Fixed-case routing:", .{});
    log.info(
        "  layer | type              | is_reader | target | slot.producer_layer | mode            | label",
        .{},
    );
    log.info(
        "  ------|-------------------|-----------|--------|---------------------|-----------------|------",
        .{},
    );
    for (FIXED_CASES) |c| {
        const r = resolved[c.layer_idx];
        const mode_str: []const u8 = switch (r.mode) {
            .producer_self => "producer_self",
            .reader_shared => "reader_shared",
        };
        log.info("  {d:>5} | {s:<17} | {s:<9} | {d:>6} | {d:>19} | {s:<15} | {s}", .{
            r.layer_idx,
            r.layer_type,
            if (r.is_reader) "true" else "false",
            r.target_kv_layer,
            r.resolved_slot_producer_layer,
            mode_str,
            c.label,
        });
        if (r.resolved_slot_producer_layer != c.expected_slot) {
            log.err(
                "  FAIL: layer {d} expected slot {d}, got {d}",
                .{ c.layer_idx, c.expected_slot, r.resolved_slot_producer_layer },
            );
            return error.FixedCaseMismatch;
        }
    }

    // === Full 35-route validation ===
    log.info("", .{});
    log.info("Full routing validation (35 entries):", .{});

    var route_fails: u32 = 0;
    var producer_self_routes: u32 = 0;
    var sliding_reader_routes_to_13: u32 = 0;
    var full_reader_routes_to_14: u32 = 0;
    var sliding_writer: ?u32 = null;
    var full_writer: ?u32 = null;

    for (resolved) |r| {
        // Sanity du mock : slot lu == slot demande
        if (r.resolved_slot_producer_layer != r.target_kv_layer) {
            log.err("[layer {d}] slot.producer_layer {d} != target_kv_layer {d}", .{
                r.layer_idx, r.resolved_slot_producer_layer, r.target_kv_layer,
            });
            route_fails += 1;
            continue;
        }
        // Marker coherent avec slot
        const expected_marker: u32 = 1000 + r.target_kv_layer;
        if (r.resolved_slot_marker != expected_marker) {
            log.err("[layer {d}] marker {d} != {d}", .{
                r.layer_idx, r.resolved_slot_marker, expected_marker,
            });
            route_fails += 1;
            continue;
        }

        // Invariants par classe
        if (r.layer_idx < first_kv_shared_layer_idx) {
            // Producer
            if (r.is_reader) {
                log.err("[layer {d}] expected producer (is_reader=false), got is_reader=true", .{r.layer_idx});
                route_fails += 1;
                continue;
            }
            if (r.resolved_slot_producer_layer != r.layer_idx) {
                log.err("[layer {d}] producer self-route expected slot={d}, got {d}", .{
                    r.layer_idx, r.layer_idx, r.resolved_slot_producer_layer,
                });
                route_fails += 1;
                continue;
            }
            producer_self_routes += 1;
            // Track writers (dernier producer de chaque type = writer designe)
            if (std.mem.eql(u8, r.layer_type, "full_attention")) {
                full_writer = r.layer_idx;
            } else if (std.mem.eql(u8, r.layer_type, "sliding_attention")) {
                sliding_writer = r.layer_idx;
            }
        } else {
            // Reader
            if (!r.is_reader) {
                log.err("[layer {d}] expected reader (is_reader=true), got false", .{r.layer_idx});
                route_fails += 1;
                continue;
            }
            const expected_target: u32 = if (std.mem.eql(u8, r.layer_type, "sliding_attention"))
                13
            else if (std.mem.eql(u8, r.layer_type, "full_attention"))
                14
            else {
                log.err("[layer {d}] unknown layer_type {s}", .{ r.layer_idx, r.layer_type });
                route_fails += 1;
                continue;
            };
            if (r.resolved_slot_producer_layer != expected_target) {
                log.err("[layer {d}] reader ({s}) expected slot {d}, got {d}", .{
                    r.layer_idx, r.layer_type, expected_target, r.resolved_slot_producer_layer,
                });
                route_fails += 1;
                continue;
            }
            if (expected_target == 13) {
                sliding_reader_routes_to_13 += 1;
            } else {
                full_reader_routes_to_14 += 1;
            }
        }
    }

    log.info("  producer_self_routes        : {d}/15", .{producer_self_routes});
    log.info("  sliding_reader_routes_to_13 : {d}/16", .{sliding_reader_routes_to_13});
    log.info("  full_reader_routes_to_14    : {d}/4", .{full_reader_routes_to_14});
    log.info("  writers stable               : sliding={?} (expect 13), full={?} (expect 14)", .{
        sliding_writer, full_writer,
    });

    if (route_fails > 0) {
        log.err("BLOCK: {d}/35 route checks failed", .{route_fails});
        return error.RoutingFailed;
    }
    if (producer_self_routes != 15) {
        log.err("expected 15/15 producer self-routes, got {d}", .{producer_self_routes});
        return error.ProducerSelfRouteCount;
    }
    if (sliding_reader_routes_to_13 != 16) {
        log.err("expected 16/16 sliding reader routes to 13, got {d}", .{sliding_reader_routes_to_13});
        return error.SlidingReaderRouteCount;
    }
    if (full_reader_routes_to_14 != 4) {
        log.err("expected 4/4 full reader routes to 14, got {d}", .{full_reader_routes_to_14});
        return error.FullReaderRouteCount;
    }
    if (sliding_writer == null or sliding_writer.? != 13) {
        log.err("sliding writer expected 13, got {?}", .{sliding_writer});
        return error.SlidingWriterMismatch;
    }
    if (full_writer == null or full_writer.? != 14) {
        log.err("full writer expected 14, got {?}", .{full_writer});
        return error.FullWriterMismatch;
    }

    log.info("", .{});
    log.info("P5.2.B PASS: producer/read routing mock validated end-to-end", .{});
    log.info("  (no Q/K/V projection, no RoPE, no matmul, no sliding mask, no real cache)", .{});
}
