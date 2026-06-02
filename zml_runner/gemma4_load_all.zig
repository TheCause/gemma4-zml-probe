// P5.7.2 — ZML loader multi-couches : charge embed + final_norm + les 35 couches (no compute).
//
// Nouvelle complexité : indexation dynamique des 35 couches (slice []LayerW + prefix/withLayer,
// idiome qwen/llama) + scale mémoire (~4 GB bf16). Prouve que le loader adresse toutes les couches
// par index et que le corps du modèle tient en mémoire. Assert les shapes par-couche
// (full head_dim 512 → q_proj 4096 ; sliding 256 → 2048 ; reader → MLP double-wide 12288).
//
// Chargement uniforme des 17 tenseurs disque/couche (y compris K/V des readers, présents sur
// disque ; le skip YOCO runtime est une optim de P5.7.3+). PLE frontend (embed_tokens_per_layer
// 4.7 GB) différé. CLI : gemma4_load_all <path-to-model.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const NUM_LAYERS: usize = 35;
const View = zml.io.TensorStore.View;

// full_attention aux couches 4,9,14,19,24,29,34.
fn isFull(i: usize) bool {
    return (i + 1) % 5 == 0;
}
// readers : i >= first_kv_shared_layer_idx = 15.
fn isReader(i: usize) bool {
    return i >= 15;
}

const LayerW = struct {
    input_layernorm: zml.Tensor,
    q_proj: zml.Tensor,
    q_norm: zml.Tensor,
    k_proj: zml.Tensor,
    k_norm: zml.Tensor,
    v_proj: zml.Tensor,
    o_proj: zml.Tensor,
    post_attention_layernorm: zml.Tensor,
    pre_feedforward_layernorm: zml.Tensor,
    gate_proj: zml.Tensor,
    up_proj: zml.Tensor,
    down_proj: zml.Tensor,
    post_feedforward_layernorm: zml.Tensor,
    per_layer_input_gate: zml.Tensor,
    per_layer_projection: zml.Tensor,
    post_per_layer_input_norm: zml.Tensor,
    layer_scalar: zml.Tensor,

    pub fn init(v: View) LayerW {
        const sa = v.withPrefix("self_attn");
        const mlp = v.withPrefix("mlp");
        return .{
            .input_layernorm = v.createTensor("input_layernorm.weight", .{.d}, null),
            .q_proj = sa.createTensor("q_proj.weight", .{ .o, .d }, null),
            .q_norm = sa.createTensor("q_norm.weight", .{.hd}, null),
            .k_proj = sa.createTensor("k_proj.weight", .{ .o, .d }, null),
            .k_norm = sa.createTensor("k_norm.weight", .{.hd}, null),
            .v_proj = sa.createTensor("v_proj.weight", .{ .o, .d }, null),
            .o_proj = sa.createTensor("o_proj.weight", .{ .d, .m }, null),
            .post_attention_layernorm = v.createTensor("post_attention_layernorm.weight", .{.d}, null),
            .pre_feedforward_layernorm = v.createTensor("pre_feedforward_layernorm.weight", .{.d}, null),
            .gate_proj = mlp.createTensor("gate_proj.weight", .{ .f, .d }, null),
            .up_proj = mlp.createTensor("up_proj.weight", .{ .f, .d }, null),
            .down_proj = mlp.createTensor("down_proj.weight", .{ .d, .f }, null),
            .post_feedforward_layernorm = v.createTensor("post_feedforward_layernorm.weight", .{.d}, null),
            .per_layer_input_gate = v.createTensor("per_layer_input_gate.weight", .{ .p, .d }, null),
            .per_layer_projection = v.createTensor("per_layer_projection.weight", .{ .d, .p }, null),
            .post_per_layer_input_norm = v.createTensor("post_per_layer_input_norm.weight", .{.d}, null),
            .layer_scalar = v.createTensor("layer_scalar", .{.one}, null),
        };
    }
};

const Loader = struct {
    embed_tokens: zml.Tensor,
    final_norm: zml.Tensor,
    layers: []LayerW,

    pub fn init(allocator: std.mem.Allocator, base: View) !Loader {
        const layers = try allocator.alloc(LayerW, NUM_LAYERS);
        const layers_base = base.withPrefix("layers");
        for (layers, 0..) |*layer, i| {
            layer.* = LayerW.init(layers_base.withLayer(i));
        }
        return .{
            .embed_tokens = base.createTensor("embed_tokens.weight", .{ .voc, .d }, null),
            .final_norm = base.createTensor("norm.weight", .{.d}, null),
            .layers = layers,
        };
    }

    pub fn load(
        self: *const Loader,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(Loader) {
        return zml.io.load(Loader, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_load_all <path-to-model.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];

    log.info("P5.7.2 — ZML loader multi-couches (embed + final_norm + 35 couches), no compute", .{});
    log.info("Checkpoint: {s}", .{ckpt});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const base = store.view().withPrefix("model").withPrefix("language_model");
    const model: Loader = try .init(arena.allocator(), base);

    // === Assertions de shape par-couche (symbolic) ===
    log.info("Spot-check shapes par-couche (symbolic) :", .{});
    var n_tensors: usize = 2; // embed_tokens + final_norm
    for (model.layers, 0..) |layer, i| {
        const q_o = layer.q_proj.dim(.o);
        const mlp_f = layer.gate_proj.dim(.f);
        // attendu : full -> q_o 4096 ; sliding -> 2048 ; reader -> mlp 12288 ; producer -> 6144.
        const exp_q: i64 = if (isFull(i)) 4096 else 2048;
        const exp_f: i64 = if (isReader(i)) 12288 else 6144;
        if (q_o != exp_q) {
            log.err("layer {d}: q_proj.o={d} != attendu {d} (full={})", .{ i, q_o, exp_q, isFull(i) });
            return error.ShapeMismatch;
        }
        if (mlp_f != exp_f) {
            log.err("layer {d}: gate_proj.f={d} != attendu {d} (reader={})", .{ i, mlp_f, exp_f, isReader(i) });
            return error.ShapeMismatch;
        }
        n_tensors += 17;
        if (i == 4 or i == 13 or i == 15 or i == 34) {
            log.info("  L{d:>2} full={} reader={} q_proj.o={d} mlp.f={d} ✓", .{ i, isFull(i), isReader(i), q_o, mlp_f });
        }
    }
    log.info("Symbolic OK : {d} tenseurs déclarés (2 top-level + 35×17).", .{n_tensors});

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers (35 couches + embeddings depuis le checkpoint réel)...", .{});
    const buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    // Preuve d'ingestion par index : le slice est plein, chaque couche a ses 17 buffers.
    if (buffers.layers.len != NUM_LAYERS) return error.LayerCount;
    log.info("Buffers loaded : {d} couches matérialisées (slice plein).", .{buffers.layers.len});

    log.info("P5.7.2 PASS: ZML charge embed + final_norm + 35 couches ({d} tenseurs) depuis le checkpoint réel", .{n_tensors});
    log.info("  -> loader multi-couches (indexation dynamique []LayerW + scale corps modèle) fonctionnel", .{});
}
