// W4 — brique poids 4-bit w4a16 compressed-tensors (jalon J1, spec docs/superpowers/specs/
// 2026-07-18-w4-poids-4bit-12b-design.md, plan docs/superpowers/plans/2026-07-24-w4-j1-brique-e2b.md).
//
// Format (D1, vérifié sur le checkpoint Google 12B) : par Linear [out, in] :
//   weight_packed i32 [out, in/8]  — nibble j du mot w = colonne 8w+j, stocke q+8 (non signé)
//   weight_scale bf16 [out, in/32] — une scale par groupe de 32 le long de la dim d'ENTRÉE
// Dequant = (nibble - 8) * scale, en bf16 (== référence compressed-tensors _dequantize).
//
// Extraction LOGIQUE (shiftRightLogical + AND 0xF) puis -8 : le nibble est q+8 NON SIGNÉ,
// un shift arithmétique propagerait le bit 31 (nibble 7 >= 8) et fausserait les nibbles hauts.
const std = @import("std");
const zml = @import("zml");

/// Poids linéaire packé : les 2 tenseurs du store (+ tags canoniques .o/.gp et .o/.g).
pub const W4Lin = struct {
    pk: zml.Tensor, // {.o, .gp} i32, gp = in/8
    sc: zml.Tensor, // {.o, .g}  bf16, g  = in/32

    pub fn init(v: zml.io.TensorStore.View, comptime name: []const u8) W4Lin {
        return .{
            .pk = v.createTensor(name ++ ".weight_packed", .{ .o, .gp }, null),
            .sc = v.createTensor(name ++ ".weight_scale", .{ .o, .g }, null),
        };
    }

    /// dequant + retag vers les tags attendus par le site moteur (ex .{ .d, .m } pour o_proj).
    pub fn toW(self: W4Lin, tagz: anytype) zml.Tensor {
        return dequantW4(self.pk, self.sc).withTags(tagz);
    }
};

/// Unpack seul : {.o, .gp} i32 -> {.o, .d} i32, valeurs q dans [-8, 7]. Exposé pour le gate W1.
pub fn unpackW4(pk: zml.Tensor) zml.Tensor {
    var nibs: [8]zml.Tensor = undefined;
    inline for (0..8) |j| {
        nibs[j] = pk
            .shiftRightLogical(zml.Tensor.scalar(4 * j, .i32))
            .logical(.AND, zml.Tensor.scalar(0xF, .i32));
    }
    // stack en axe mineur .nib : ordre little-endian j=0..7 == colonnes 8w+0..8w+7
    const stacked = zml.Tensor.stack(&nibs, .last, .nib); // {.o, .gp, .nib=8}
    const q = stacked.sub(zml.Tensor.scalar(8, .i32)); // q = nibble - 8
    return q.merge(.{ .d = .{ .gp, .nib } }); // {.o, .d = in}
}

/// Dequant complet : unpack -> bf16 -> * scale par groupe de 32 -> {.o, .d} bf16.
pub fn dequantW4(pk: zml.Tensor, sc: zml.Tensor) zml.Tensor {
    const q = unpackW4(pk).convert(.bf16); // {.o, .d}
    const grouped = q.splitAxis(.d, .{ .g = .auto, .gi = 32 }); // {.o, .g, .gi=32}
    // broadcast auto binaryOp : tags identiques même ordre, sc {.o,.g,.gi=1} -> {.o,.g,.gi=32}
    const deq = grouped.mul(sc.appendAxes(.{.gi}));
    return deq.merge(.{ .d = .{ .g, .gi } }); // {.o, .d}
}
