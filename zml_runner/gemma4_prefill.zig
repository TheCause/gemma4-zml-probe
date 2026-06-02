// P5.7.5 — ZML MOTEUR PREFILL Gemma-4-E2B-it (forward complet, validé par composition producer+reader).
//
// Compose toutes les ops validées en un forward dispatché sliding/full avec KV-sharing YOCO, vs l'oracle
// hybride (fixtures/p5_7_5_hybrid.safetensors, contrat docs/P5_7_5_precision_contract.md).
//
// DEUX MODES (le forward 35-couches déroulé fp32 ne tient pas en VM 24 Go -> validation par moitiés) :
//  - "producer" (défaut) : embedding + PLE frontend + couches 0-14 (capture KV writers 13/14).
//    Compare taps 0/4/13/14 + STAGE0 embeds aux hidden oracle. Valide sliding+full producers + PLE.
//  - "reader" : depuis hidden_15 + KV partagé (fixture, fournis par l'oracle), couches 15-34 + final norm.
//    Compare taps 15/19/33 + last_hidden. Valide le chemin reader YOCO. Producer PASS + reader PASS
//    ⟹ forward complet validé (l'oracle fournit le hidden_15/KV que le producer reproduit bit-near).
//
// Corps fp32 (convert bf16->f32) ; embed_tokens / embed_tokens_per_layer pré-gatherés en slices (fixture)
// pour éviter l'OOM. Poids depuis un checkpoint slim (35 couches + frontend + norm).
//
// CLI : gemma4_prefill <model.safetensors> <p5_7_5_hybrid.safetensors> [producer|reader]

const std = @import("std");
const log = std.log;
const zml = @import("zml");
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };

const NUM_LAYERS: usize = 35;
const FIRST_KV_SHARED: usize = 15; // num_hidden_layers - num_kv_shared = 35-20
const SLIDING_WRITER: usize = 13;
const FULL_WRITER: usize = 14;

const B: i64 = 1;
const S: i64 = 4;
const D: i64 = 1536;
const NH: i64 = 8;
const KVH: i64 = 1;
const HD_SLIDING: i64 = 256;
const HD_FULL: i64 = 512;
const PLE_DIM: i64 = 256;
const LF: i64 = 8960; // 35 * 256

const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA_SLIDING: f32 = 1.0e4;
const EMBED_SCALE: f64 = @sqrt(1536.0);
const INV_SQRT_HID: f64 = 1.0 / @sqrt(1536.0);
const SQRT_PLE: f64 = 16.0;
const INV_SQRT_2: f64 = 0.7071067811865476;

fn isFull(i: usize) bool {
    return (i + 1) % 5 == 0;
}
fn isReader(i: usize) bool {
    return i >= FIRST_KV_SHARED;
}

inline fn c(t: zml.Tensor) zml.Tensor {
    return t.convert(.f32);
}
fn rmsScaleD(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const n = zml.nn.rmsNorm(x, .d, RMS_EPS);
    return n.mul(w.broad(n.shape()));
}
fn rmsScaleP(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
    const n = zml.nn.rmsNorm(x, .p, RMS_EPS);
    return n.mul(w.broad(n.shape()));
}
fn manualRope(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor, half: i64) zml.Tensor {
    const halves = x.split(.hd, &.{ half, half });
    const rh = zml.Tensor.concatenate(&.{ halves[1].negate(), halves[0] }, .hd);
    return x.mul(cos.broad(x.shape())).add(rh.mul(sin.broad(x.shape())));
}

// ---- Poids par couche : exactement gemma4_load_all.LayerW (17 tenseurs disque, bf16) ----
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

    pub fn init(v: zml.io.TensorStore.View) LayerW {
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

// ---- KV partagé YOCO (capturé par producers 13/14 ; pré-chargé en mode reader) ----
const SharedKV = struct {
    k_sliding: ?zml.Tensor = null,
    v_sliding: ?zml.Tensor = null,
    k_full: ?zml.Tensor = null,
    v_full: ?zml.Tensor = null,
};

// ---- Inputs runtime (depuis la fixture, arg de forward) ----
const Runtime = struct {
    embed_slice: zml.Tensor, // {.b,.s,.d} bf16
    embptl_slice: zml.Tensor, // {.b,.s,.lf} bf16
    cos_full: zml.Tensor, // {.b,.s,.hd=512}
    sin_full: zml.Tensor,
    attn_mask: zml.Tensor, // {.b,.h,.q,.k}
    // entrées du mode reader (fournies par l'oracle) :
    hidden_15: zml.Tensor, // {.b,.s,.d} résiduel entrant couche 15
    hidden_25: zml.Tensor, // {.b,.s,.d} résiduel entrant couche 25 (tranche reader 2)
    k_sliding: zml.Tensor, // {.b,.h,.k,.hd=256} KV partagé writer 13
    v_sliding: zml.Tensor,
    k_full: zml.Tensor, // {.b,.h,.k,.hd=512} KV partagé writer 14
    v_full: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) Runtime {
        return .{
            .embed_slice = v.createTensor("embed_slice", .{ .b, .s, .d }, null),
            .embptl_slice = v.createTensor("embptl_slice", .{ .b, .s, .lf }, null),
            .cos_full = v.createTensor("cos_full", .{ .b, .s, .hd }, null),
            .sin_full = v.createTensor("sin_full", .{ .b, .s, .hd }, null),
            .attn_mask = v.createTensor("attn_mask", .{ .b, .h, .q, .k }, null),
            .hidden_15 = v.createTensor("hidden_15", .{ .b, .s, .d }, null),
            .hidden_25 = v.createTensor("hidden_25", .{ .b, .s, .d }, null),
            .k_sliding = v.createTensor("kv_k_sliding", .{ .b, .h, .k, .hd }, null),
            .v_sliding = v.createTensor("kv_v_sliding", .{ .b, .h, .k, .hd }, null),
            .k_full = v.createTensor("kv_k_full", .{ .b, .h, .k, .hd }, null),
            .v_full = v.createTensor("kv_v_full", .{ .b, .h, .k, .hd }, null),
        };
    }

    pub fn load(self: *const Runtime, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Runtime) {
        return zml.io.load(Runtime, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

/// Forward d'UNE couche décodeur i. Producers (0-14) calculent K/V et publient à 13/14 (writes shared) ;
/// readers (15-34) réutilisent shared. Retourne le nouveau hidden (résiduel, ×layer_scalar inclus).
fn runLayer(layer: LayerW, i: usize, hidden: zml.Tensor, ple_i: zml.Tensor, rt: Runtime, shared: *SharedKV) zml.Tensor {
    const full = isFull(i);
    const reader = isReader(i);
    const hd: i64 = if (full) HD_FULL else HD_SLIDING;
    const half: i64 = @divExact(hd, 2);

    const input_ln = c(layer.input_layernorm);
    const q_proj = c(layer.q_proj);
    const q_norm = c(layer.q_norm);
    const o_proj = c(layer.o_proj);
    const post_attn_ln = c(layer.post_attention_layernorm);
    const pre_ff_ln = c(layer.pre_feedforward_layernorm);
    const gate_proj = c(layer.gate_proj);
    const up_proj = c(layer.up_proj);
    const down_proj = c(layer.down_proj);
    const post_ff_ln = c(layer.post_feedforward_layernorm);
    const ple_gate = c(layer.per_layer_input_gate);
    const ple_proj = c(layer.per_layer_projection);
    const ple_norm = c(layer.post_per_layer_input_norm);
    const layer_scalar = c(layer.layer_scalar);

    const h0 = rmsScaleD(hidden, input_ln);

    var q = h0.dot(q_proj, .d).reshape(.{ B, S, NH, hd }).withTags(.{ .b, .s, .nh, .hd });
    q = zml.nn.rmsNorm(q, .hd, RMS_EPS).mul(q_norm.broad(q.shape()));
    q = if (full)
        manualRope(q, rt.cos_full, rt.sin_full, half)
    else
        zml.nn.rope(q, null, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = ROPE_THETA_SLIDING } } });
    const q_final = q.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .q });

    var k_final: zml.Tensor = undefined;
    var v_final: zml.Tensor = undefined;
    if (reader) {
        if (full) {
            k_final = shared.k_full.?;
            v_final = shared.v_full.?;
        } else {
            k_final = shared.k_sliding.?;
            v_final = shared.v_sliding.?;
        }
    } else {
        const k_proj = c(layer.k_proj);
        const k_norm = c(layer.k_norm);
        const v_proj = c(layer.v_proj);
        var k = h0.dot(k_proj, .d).reshape(.{ B, S, KVH, hd }).withTags(.{ .b, .s, .nh, .hd });
        k = zml.nn.rmsNorm(k, .hd, RMS_EPS).mul(k_norm.broad(k.shape()));
        k = if (full)
            manualRope(k, rt.cos_full, rt.sin_full, half)
        else
            zml.nn.rope(k, null, .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = ROPE_THETA_SLIDING } } });
        k_final = k.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        var v = h0.dot(v_proj, .d).reshape(.{ B, S, KVH, hd }).withTags(.{ .b, .s, .nh, .hd });
        v = zml.nn.rmsNorm(v, .hd, RMS_EPS); // v_norm SANS scale
        v_final = v.transpose(.{ .b, .nh, .s, .hd }).rename(.{ .nh = .h, .s = .k });

        if (i == SLIDING_WRITER) {
            shared.k_sliding = k_final;
            shared.v_sliding = v_final;
        }
        if (i == FULL_WRITER) {
            shared.k_full = k_final;
            shared.v_full = v_final;
        }
    }

    const qs = q_final.splitAxis(.h, .{ .h = k_final.dim(.h), .hq = .auto });
    var scores = qs.dot(k_final, .hd).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .k });
    scores = scores.add(rt.attn_mask.broad(scores.shape()));
    const probs = scores.softmax(.k);

    const ps = probs.splitAxis(.h, .{ .h = v_final.dim(.h), .hq = .auto });
    const ctx = ps.dot(v_final, .k).merge(.{ .h = .{ .h, .hq } }).transpose(.{ .b, .h, .q, .hd });
    const attn_m = ctx.transpose(.{ .b, .q, .h, .hd }).merge(.{ .m = .{ .h, .hd } });
    const attn_out = attn_m.dot(o_proj, .m).rename(.{ .q = .s });

    const h1 = hidden.add(rmsScaleD(attn_out, post_attn_ln));
    const xff = rmsScaleD(h1, pre_ff_ln);
    const mlp_out = xff.dot(gate_proj, .d).gelu().mul(xff.dot(up_proj, .d)).dot(down_proj, .f);
    const h2 = h1.add(rmsScaleD(mlp_out, post_ff_ln));

    var g = h2.dot(ple_gate, .d).gelu();
    g = g.mul(ple_i);
    g = g.dot(ple_proj, .p);
    const h3 = h2.add(rmsScaleD(g, ple_norm));

    return h3.mul(layer_scalar.asScalar());
}

const Out8 = struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor };

// ---- Moteur : poids modèle (checkpoint slim) ----
const Engine = struct {
    per_layer_model_projection: zml.Tensor, // {.lf,.d}
    per_layer_projection_norm: zml.Tensor, // {.p}
    final_norm: zml.Tensor, // {.d}
    layers: []LayerW,

    pub fn init(allocator: std.mem.Allocator, base: zml.io.TensorStore.View) !Engine {
        const layers = try allocator.alloc(LayerW, NUM_LAYERS);
        const layers_base = base.withPrefix("layers");
        for (layers, 0..) |*layer, i| layer.* = LayerW.init(layers_base.withLayer(i));
        return .{
            .per_layer_model_projection = base.createTensor("per_layer_model_projection.weight", .{ .lf, .d }, null),
            .per_layer_projection_norm = base.createTensor("per_layer_projection_norm.weight", .{.p}, null),
            .final_norm = base.createTensor("norm.weight", .{.d}, null),
            .layers = layers,
        };
    }

    pub fn load(self: *const Engine, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Engine) {
        return zml.io.load(Engine, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }

    /// PLE frontend -> per_layer_inputs {.b,.s,.layer=35,.p=256}.
    fn perLayerInputs(self: Engine, rt: Runtime, embeds: zml.Tensor) zml.Tensor {
        const token_identity = rt.embptl_slice
            .scale(SQRT_PLE).convert(.f32)
            .reshape(.{ B, S, NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
        const context = embeds.dot(c(self.per_layer_model_projection), .d)
            .scale(INV_SQRT_HID)
            .reshape(.{ B, S, NUM_LAYERS, PLE_DIM }).withTags(.{ .b, .s, .layer, .p });
        const context_norm = rmsScaleP(context, c(self.per_layer_projection_norm));
        return context_norm.add(token_identity).scale(INV_SQRT_2);
    }

    /// Mode producer : embedding + PLE + couches 0-14. Retour : {embeds, L0, L4, L13, L14, _, _, _}.
    pub fn forwardProducer(self: Engine, rt: Runtime) Out8 {
        const embeds = rt.embed_slice.convert(.f32).scale(EMBED_SCALE);
        const ple = self.perLayerInputs(rt, embeds);
        var hidden = embeds;
        var shared = SharedKV{};
        var t0 = embeds;
        var t4 = embeds;
        var t13 = embeds;
        var t14 = embeds;
        for (self.layers[0..FIRST_KV_SHARED], 0..) |layer, i| {
            const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
            hidden = runLayer(layer, i, hidden, ple_i, rt, &shared);
            switch (i) {
                0 => t0 = hidden,
                4 => t4 = hidden,
                13 => t13 = hidden,
                14 => t14 = hidden,
                else => {},
            }
        }
        return .{ embeds, t0, t4, t13, t14, embeds, embeds, embeds };
    }

    /// Helper KV partagé depuis la fixture (mode reader).
    fn sharedFromRt(rt: Runtime) SharedKV {
        return .{ .k_sliding = rt.k_sliding, .v_sliding = rt.v_sliding, .k_full = rt.k_full, .v_full = rt.v_full };
    }

    /// Mode reader (tranche 1) : couches 15-24 depuis hidden_15 + KV partagé. Mémoire bornée (10 couches).
    /// Retour : {L15, L19, sortie L24 (=hidden_25), _, _, _, _, _}.
    pub fn forwardReader(self: Engine, rt: Runtime) Out8 {
        const embeds = rt.embed_slice.convert(.f32).scale(EMBED_SCALE);
        const ple = self.perLayerInputs(rt, embeds);
        var shared = sharedFromRt(rt);
        var hidden = rt.hidden_15;
        var t15 = hidden;
        var t19 = hidden;
        for (self.layers[15..25], 15..) |layer, i| {
            const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
            hidden = runLayer(layer, i, hidden, ple_i, rt, &shared);
            switch (i) {
                15 => t15 = hidden,
                19 => t19 = hidden,
                else => {},
            }
        }
        return .{ t15, t19, hidden, hidden, hidden, hidden, hidden, hidden };
    }

    /// Mode reader (tranche 2) : couches 25-34 depuis hidden_25 + KV partagé, + final norm.
    /// Retour : {L29, last_hidden, sortie L34 pré-norm, _, _, _, _, _}.
    pub fn forwardReader2(self: Engine, rt: Runtime) Out8 {
        const embeds = rt.embed_slice.convert(.f32).scale(EMBED_SCALE);
        const ple = self.perLayerInputs(rt, embeds);
        var shared = sharedFromRt(rt);
        var hidden = rt.hidden_25;
        var t29 = hidden;
        for (self.layers[25..NUM_LAYERS], 25..) |layer, i| {
            const ple_i = ple.choose1d(.layer, @as(i64, @intCast(i)));
            hidden = runLayer(layer, i, hidden, ple_i, rt, &shared);
            switch (i) {
                29 => t29 = hidden,
                else => {},
            }
        }
        const last_hidden = rmsScaleD(hidden, c(self.final_norm));
        return .{ t29, last_hidden, hidden, hidden, hidden, hidden, hidden, hidden };
    }
};

// ---- Oracle (depuis la fixture) ----
const Oracle = struct {
    h00: zml.Tensor,
    h01: zml.Tensor,
    h05: zml.Tensor,
    h14: zml.Tensor,
    h15: zml.Tensor,
    h16: zml.Tensor,
    h20: zml.Tensor,
    h25: zml.Tensor,
    h30: zml.Tensor,
    h34: zml.Tensor,
    last_hidden: zml.Tensor,

    pub fn init(v: zml.io.TensorStore.View) Oracle {
        return .{
            .h00 = v.createTensor("hidden_00", .{ .b, .s, .d }, null),
            .h01 = v.createTensor("hidden_01", .{ .b, .s, .d }, null),
            .h05 = v.createTensor("hidden_05", .{ .b, .s, .d }, null),
            .h14 = v.createTensor("hidden_14", .{ .b, .s, .d }, null),
            .h15 = v.createTensor("hidden_15", .{ .b, .s, .d }, null),
            .h16 = v.createTensor("hidden_16", .{ .b, .s, .d }, null),
            .h20 = v.createTensor("hidden_20", .{ .b, .s, .d }, null),
            .h25 = v.createTensor("hidden_25", .{ .b, .s, .d }, null),
            .h30 = v.createTensor("hidden_30", .{ .b, .s, .d }, null),
            .h34 = v.createTensor("hidden_34", .{ .b, .s, .d }, null),
            .last_hidden = v.createTensor("last_hidden", .{ .b, .s, .d }, null),
        };
    }

    pub fn load(self: *const Oracle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.sharding.Sharding) !zml.Bufferized(Oracle) {
        return zml.io.load(Oracle, self, allocator, io, platform, store, .{ .shardings = shardings, .parallelism = 1, .dma_chunks = 1, .dma_chunk_size = 16 * 1024 * 1024 });
    }
};

const FLAT: usize = @intCast(B * S * D);
const PASS_MAX_ABS: f32 = 1.0e-2;
const PASS_MEAN_ABS: f32 = 1.0e-4;

fn compareTap(allocator: std.mem.Allocator, io: std.Io, label: []const u8, layer_idx: usize, out_buf: *zml.Buffer, ref_buf: *zml.Buffer) !bool {
    var out_s = try out_buf.toSliceAlloc(allocator, io);
    defer out_s.free(allocator);
    var ref_s = try ref_buf.toSliceAlloc(allocator, io);
    defer ref_s.free(allocator);
    const out = out_s.items(f32);
    const ref = ref_s.items(f32);
    if (out.len != FLAT or ref.len != FLAT) {
        log.err("  {s} (L{d}): length mismatch out={d} ref={d}", .{ label, layer_idx, out.len, ref.len });
        return error.LengthMismatch;
    }
    var max_abs: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    var out_max: f32 = 0.0;
    var ref_max: f32 = 0.0;
    var nan_inf = false;
    for (out, ref) |a, b| {
        if (std.math.isNan(a) or std.math.isInf(a)) nan_inf = true;
        const diff = @abs(a - b);
        if (diff > max_abs) max_abs = diff;
        sum_abs += @as(f64, diff);
        if (@abs(a) > out_max) out_max = @abs(a);
        if (@abs(b) > ref_max) ref_max = @abs(b);
    }
    const mean_abs = @as(f32, @floatCast(sum_abs / @as(f64, @floatFromInt(FLAT))));
    const pass = !nan_inf and max_abs <= PASS_MAX_ABS and mean_abs <= PASS_MEAN_ABS;
    const verdict = if (nan_inf) "FAIL(NaN/Inf)" else if (pass) "PASS" else if (max_abs <= 1.0e-1) "WARN" else "FAIL";
    log.info("  {s:<24} (L{d:>2}): max_abs={e:.4} mean_abs={e:.4} | max|zml|={e:.3} max|ref|={e:.3} -> {s}", .{ label, layer_idx, max_abs, mean_abs, out_max, ref_max, verdict });
    return pass;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    const process_args = try init.minimal.args.toSlice(arena.allocator());
    if (process_args.len < 3) {
        log.err("Usage: gemma4_prefill <model.safetensors> <p5_7_5_hybrid.safetensors> [producer|reader]", .{});
        return error.MissingArgument;
    }
    const ckpt = process_args[1];
    const fixture = process_args[2];
    const mode = if (process_args.len >= 4) process_args[3] else "producer";
    const is_reader = std.mem.eql(u8, mode, "reader");
    const is_reader2 = std.mem.eql(u8, mode, "reader2");

    log.info("P5.7.5 — ZML moteur prefill — mode {s}", .{mode});

    var reg_ck: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, ckpt);
    var store_ck: zml.io.TensorStore = .fromRegistry(allocator, &reg_ck);

    var reg_fx: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, fixture);
    defer reg_fx.deinit();
    var store_fx: zml.io.TensorStore = .fromRegistry(allocator, &reg_fx);
    defer store_fx.deinit();

    const base = store_ck.view().withPrefix("model").withPrefix("language_model");
    const engine: Engine = try .init(arena.allocator(), base);
    const rt: Runtime = .init(store_fx.view());
    const oracle: Oracle = .init(store_fx.view());

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    log.info("Materializing weights...", .{});
    const eng_buf = try engine.load(arena.allocator(), io, platform, &store_ck, &.{sharding});
    const rt_buf = try rt.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    var orc_buf = try oracle.load(arena.allocator(), io, platform, &store_fx, &.{sharding});
    store_ck.deinit();
    reg_ck.deinit();
    log.info("Buffers loaded (checkpoint registry libéré).", .{});

    log.info("Compiling forward...", .{});
    var exe = if (is_reader)
        try platform.compile(allocator, io, engine, .forwardReader, .{rt}, .{ .shardings = &.{sharding} })
    else if (is_reader2)
        try platform.compile(allocator, io, engine, .forwardReader2, .{rt}, .{ .shardings = &.{sharding} })
    else
        try platform.compile(allocator, io, engine, .forwardProducer, .{rt}, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{ eng_buf, rt_buf });
    exe.call(args, &results);

    var r0, var r1, var r2, var r3, var r4, var r5, var r6, var r7 = results.get(struct {
        zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
        zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer,
    });
    defer {
        r0.deinit();
        r1.deinit();
        r2.deinit();
        r3.deinit();
        r4.deinit();
        r5.deinit();
        r6.deinit();
        r7.deinit();
    }

    log.info("Comparaison par couche vs oracle (PASS: max_abs<=1e-2 ET mean_abs<=1e-4) :", .{});
    var all_pass = true;
    if (is_reader) {
        all_pass = (try compareTap(allocator, io, "tap L15 (sliding rd)", 15, &r0, &orc_buf.h16)) and all_pass;
        all_pass = (try compareTap(allocator, io, "tap L19 (full rd)", 19, &r1, &orc_buf.h20)) and all_pass;
        all_pass = (try compareTap(allocator, io, "out L24 -> hidden_25", 24, &r2, &orc_buf.h25)) and all_pass;
    } else if (is_reader2) {
        all_pass = (try compareTap(allocator, io, "tap L29 -> hidden_30", 29, &r0, &orc_buf.h30)) and all_pass;
        all_pass = (try compareTap(allocator, io, "last_hidden (post-norm)", 35, &r1, &orc_buf.last_hidden)) and all_pass;
    } else {
        all_pass = (try compareTap(allocator, io, "STAGE0 embeds vs h00", 0, &r0, &orc_buf.h00)) and all_pass;
        all_pass = (try compareTap(allocator, io, "tap L0 (sliding prod)", 0, &r1, &orc_buf.h01)) and all_pass;
        all_pass = (try compareTap(allocator, io, "tap L4 (full prod)", 4, &r2, &orc_buf.h05)) and all_pass;
        all_pass = (try compareTap(allocator, io, "tap L13 (sliding wr)", 13, &r3, &orc_buf.h14)) and all_pass;
        all_pass = (try compareTap(allocator, io, "tap L14 (full wr)", 14, &r4, &orc_buf.h15)) and all_pass;
    }

    if (all_pass) {
        log.info("P5.7.5 mode={s} PASS", .{mode});
    } else {
        log.err("P5.7.5 mode={s}: divergence (1er tap non-PASS = régime fautif, contrat §6)", .{mode});
        return error.PrefillMismatch;
    }
}
