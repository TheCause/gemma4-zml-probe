// P5.7.1 — ZML chargement depuis le checkpoint RÉEL (embeddings + 1 couche). Pas de slim-fixture.
//
// Prouve que ZML ingère weights/model.safetensors (bf16, 2011 tenseurs) par clé SÉLECTIVE :
// charge embed_tokens + final_norm + les 17 tenseurs disque de layer 13, vérifie les shapes
// (symbolic, asserts) et compare 4 petits tenseurs (layer_scalar, input_layernorm, q_norm, k_norm)
// convertis bf16->f32 vs référence PyTorch (bf16->f32 = exact, attendu bit-exact).
//
// Nouvelle complexité P5.7.1 : (1) chargement sélectif depuis le gros checkpoint multi-tenseurs,
// (2) dtype bf16 (les gates précédentes utilisaient des fixtures fp32). Les tenseurs déclarés mais
// non utilisés dans forward sont quand même chargés par model.load (prouve l'ingestion).
//
// CLI : gemma4_load_check <path-to-model.safetensors>

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const TOL: f32 = 1.0e-6; // bf16->f32 exact -> attendu 0.0

// Référence fp32 (bf16 lu, cf scripts/35_*.py).
const LAYER_SCALAR_REF: f32 = 0.0883789062;
const INPUT_LN_REF = [_]f32{ 17.6250000000, 0.4121093750, 0.4433593750, 21.6250000000, 0.3828125000, 17.7500000000, 0.6601562500, 0.4316406250 };
const Q_NORM_REF = [_]f32{ 0.9921875000, 0.9921875000, 0.9921875000, 0.9921875000, 0.9921875000, 0.9921875000, 0.9921875000, 0.9921875000 };
const K_NORM_REF = [_]f32{ 0.1259765625, 0.1259765625, 0.1259765625, 0.1259765625, 0.1259765625, 0.1259765625, 0.1259765625, 0.1259765625 };

const L = "model.language_model.layers.13";

const LoadFixture = struct {
    // Embeddings + final norm.
    embed_tokens: zml.Tensor,
    final_norm: zml.Tensor,
    // Layer 13 (17 tenseurs disque).
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

    pub fn init(store: zml.io.TensorStore.View) LoadFixture {
        return .{
            .embed_tokens = store.createTensor("model.language_model.embed_tokens.weight", .{ .voc, .d }, null),
            .final_norm = store.createTensor("model.language_model.norm.weight", .{.d}, null),
            .input_layernorm = store.createTensor(L ++ ".input_layernorm.weight", .{.d}, null),
            .q_proj = store.createTensor(L ++ ".self_attn.q_proj.weight", .{ .o, .d }, null),
            .q_norm = store.createTensor(L ++ ".self_attn.q_norm.weight", .{.hd}, null),
            .k_proj = store.createTensor(L ++ ".self_attn.k_proj.weight", .{ .o, .d }, null),
            .k_norm = store.createTensor(L ++ ".self_attn.k_norm.weight", .{.hd}, null),
            .v_proj = store.createTensor(L ++ ".self_attn.v_proj.weight", .{ .o, .d }, null),
            .o_proj = store.createTensor(L ++ ".self_attn.o_proj.weight", .{ .d, .m }, null),
            .post_attention_layernorm = store.createTensor(L ++ ".post_attention_layernorm.weight", .{.d}, null),
            .pre_feedforward_layernorm = store.createTensor(L ++ ".pre_feedforward_layernorm.weight", .{.d}, null),
            .gate_proj = store.createTensor(L ++ ".mlp.gate_proj.weight", .{ .f, .d }, null),
            .up_proj = store.createTensor(L ++ ".mlp.up_proj.weight", .{ .f, .d }, null),
            .down_proj = store.createTensor(L ++ ".mlp.down_proj.weight", .{ .d, .f }, null),
            .post_feedforward_layernorm = store.createTensor(L ++ ".post_feedforward_layernorm.weight", .{.d}, null),
            .per_layer_input_gate = store.createTensor(L ++ ".per_layer_input_gate.weight", .{ .p, .d }, null),
            .per_layer_projection = store.createTensor(L ++ ".per_layer_projection.weight", .{ .d, .p }, null),
            .post_per_layer_input_norm = store.createTensor(L ++ ".post_per_layer_input_norm.weight", .{.d}, null),
            .layer_scalar = store.createTensor(L ++ ".layer_scalar", .{.one}, null),
        };
    }

    pub fn load(
        self: *const LoadFixture,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(LoadFixture) {
        return zml.io.load(LoadFixture, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    /// Forward : convertit en f32 les 4 petits tenseurs à vérifier. Les autres sont chargés
    /// (preuve d'ingestion) mais non utilisés ici.
    pub fn forward(self: LoadFixture) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
        return .{
            self.layer_scalar.convert(.f32),
            self.input_layernorm.convert(.f32),
            self.q_norm.convert(.f32),
            self.k_norm.convert(.f32),
        };
    }
};

fn checkBlock(name: []const u8, data: []const f32, ref: []const f32) !f32 {
    var max_diff: f32 = 0.0;
    const n = @min(data.len, ref.len);
    for (0..n) |i| {
        const diff = @abs(data[i] - ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    log.info("  {s}: loaded[:{d}]={d:.6} ref[:{d}]={d:.6} max_diff={e:.3}", .{ name, n, data[0], n, ref[0], max_diff });
    if (max_diff > TOL) {
        log.err("BLOCK: {s} max_diff {e:.3} > tol {e:.1}", .{ name, max_diff, TOL });
        return error.LoadMismatch;
    }
    return max_diff;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 2) {
        log.err("Usage: gemma4_load_check <path-to-model.safetensors>", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];

    log.info("P5.7.1 — ZML chargement sélectif depuis checkpoint réel bf16 (embed + layer 13)", .{});
    log.info("Checkpoint: {s}", .{ckpt});

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: LoadFixture = .init(store.view());

    // Assertions de shape (symbolic) vs manifest P5.7.0.
    log.info("Shapes chargées (symbolic) :", .{});
    log.info("  embed_tokens : {f}", .{model.embed_tokens});
    log.info("  final_norm   : {f}", .{model.final_norm});
    log.info("  q_proj       : {f}", .{model.q_proj});
    log.info("  k_proj       : {f}", .{model.k_proj});
    log.info("  o_proj       : {f}", .{model.o_proj});
    log.info("  gate_proj    : {f}", .{model.gate_proj});
    log.info("  down_proj    : {f}", .{model.down_proj});
    log.info("  layer_scalar : {f}", .{model.layer_scalar});
    if (model.embed_tokens.dim(.voc) != 262144 or model.embed_tokens.dim(.d) != 1536) return error.ShapeEmbed;
    if (model.q_proj.dim(.o) != 2048 or model.q_proj.dim(.d) != 1536) return error.ShapeQProj;
    if (model.gate_proj.dim(.f) != 6144) return error.ShapeGate; // layer 13 producer = 6144

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing buffers (19 tenseurs depuis le checkpoint réel ~0.9 GB)...", .{});
    const buffers = try model.load(arena.allocator(), io, platform, &store, &.{replicated_sharding});
    log.info("Buffers loaded — 19 tenseurs ingérés depuis model.safetensors par clé sélective.", .{});

    var exe = try platform.compile(allocator, io, model, .forward, .{}, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{buffers});
    exe.call(args, &results);

    var ls_buf, var iln_buf, var qn_buf, var kn_buf = results.get(struct { zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer });
    defer ls_buf.deinit();
    defer iln_buf.deinit();
    defer qn_buf.deinit();
    defer kn_buf.deinit();

    var ls = try ls_buf.toSliceAlloc(allocator, io);
    defer ls.free(allocator);
    var iln = try iln_buf.toSliceAlloc(allocator, io);
    defer iln.free(allocator);
    var qn = try qn_buf.toSliceAlloc(allocator, io);
    defer qn.free(allocator);
    var kn = try kn_buf.toSliceAlloc(allocator, io);
    defer kn.free(allocator);

    log.info("Vérification valeurs (bf16->f32, attendu bit-exact) :", .{});
    var max_all: f32 = 0.0;
    const d1 = try checkBlock("layer_scalar", ls.items(f32), &.{LAYER_SCALAR_REF});
    const d2 = try checkBlock("input_layernorm", iln.items(f32)[0..8], &INPUT_LN_REF);
    const d3 = try checkBlock("q_norm", qn.items(f32)[0..8], &Q_NORM_REF);
    const d4 = try checkBlock("k_norm", kn.items(f32)[0..8], &K_NORM_REF);
    inline for (.{ d1, d2, d3, d4 }) |d| {
        if (d > max_all) max_all = d;
    }
    log.info("  -> max_diff global = {e:.3} (tol {e:.1})", .{ max_all, TOL });

    log.info("P5.7.1 PASS: ZML charge embeddings + layer 13 depuis le checkpoint réel bf16 (shapes + valeurs OK)", .{});
}
