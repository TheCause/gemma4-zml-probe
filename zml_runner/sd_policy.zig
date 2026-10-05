// sd_policy.zig — Évaluateur de décision SD (spec docs/superpowers/specs/2026-09-24-decision-layer-design.md
// §4.4-4.5). PUR : aucune dépendance ZML — testable seul (`bazel test //examples/rqz:sd_policy_test`)
// et appelé tel quel par gemma4_decide (`--policy-selftest` rejoue les mêmes cas sur le binaire livré).
const std = @import("std");

pub const N = 4;
pub const Class = enum { direct, search, calculate, insufficient };
pub const ErrKind = enum { empty_set, duplicate_label, label_out_of_vocab, non_finite };

pub const Decision = union(enum) {
    decision: struct { option: Class, p: [N]f32, p_max: f32, margin: f32, mass_in: f32 },
    abstain: struct { option: Class, p: [N]f32, margin: f32, mass_in: f32, reason: enum { insufficient, low_margin } },
    err: ErrKind,
};

pub const Input = struct {
    zc: [N]f32, // logits après softcap, dans l'ordre des lettres A..D
    lse: f32, // logsumexp du vocabulaire complet
    label_ids: [N]u32,
    classes: [N]Class, // affectation lettre→classe du cas (propre au perm)
    vocab: u32,
    n_valid: usize = N, // < N simule un ensemble tronqué (empty_set si 0)
    theta: f32 = 0, // V1 : 0 (aucune abstention par marge)
};

pub fn evaluate(in: Input) Decision {
    // Validations de schéma AVANT toute lecture de p (spec §4.4) : une distribution concentrée
    // ne court-circuite jamais une erreur.
    if (in.n_valid == 0) return .{ .err = .empty_set };
    for (0..N) |i| {
        if (in.label_ids[i] >= in.vocab) return .{ .err = .label_out_of_vocab };
        for (0..i) |j| if (in.label_ids[i] == in.label_ids[j]) return .{ .err = .duplicate_label };
    }
    for (in.zc) |z| if (!std.math.isFinite(z)) return .{ .err = .non_finite };
    if (!std.math.isFinite(in.lse)) return .{ .err = .non_finite };

    // Softmax stable T=1 (§4.5), calcul en f64, rendu en f32.
    var m: f64 = in.zc[0];
    for (in.zc) |z| m = @max(m, @as(f64, z));
    var e: [N]f64 = undefined;
    var s: f64 = 0;
    for (in.zc, 0..) |z, i| {
        e[i] = @exp(@as(f64, z) - m);
        s += e[i];
    }
    var p: [N]f32 = undefined;
    var best: usize = 0;
    for (0..N) |i| {
        p[i] = @floatCast(e[i] / s);
        if (e[i] > e[best]) best = i; // égalité : la plus petite lettre (déterministe)
    }
    var second: f32 = 0;
    for (0..N) |i| if (i != best) {
        second = @max(second, p[i]);
    };
    const margin = p[best] - second;
    const mass_in: f32 = @floatCast(@exp(m + @log(s) - @as(f64, in.lse)));
    const option = in.classes[best];

    if (option == .insufficient)
        return .{ .abstain = .{ .option = option, .p = p, .margin = margin, .mass_in = mass_in, .reason = .insufficient } };
    if (margin < in.theta)
        return .{ .abstain = .{ .option = option, .p = p, .margin = margin, .mass_in = mass_in, .reason = .low_margin } };
    return .{ .decision = .{ .option = option, .p = p, .p_max = p[best], .margin = margin, .mass_in = mass_in } };
}

pub fn classFromStr(s: []const u8) ?Class {
    return std.meta.stringToEnum(Class, s);
}

// ---------------------------------------------------------------- tests (C-SD-F)
const ids_ok = [N]u32{ 236776, 236799, 236780, 236796 }; // valeurs quelconques < vocab
const orig = [N]Class{ .direct, .search, .calculate, .insufficient };
const rev = [N]Class{ .insufficient, .calculate, .search, .direct };

fn base(zc: [N]f32, classes: [N]Class) Input {
    return .{ .zc = zc, .lse = 20.0, .label_ids = ids_ok, .classes = classes, .vocab = 262144 };
}

pub const selftest_cases = [_]struct { name: []const u8, in: Input, want: []const u8 }{
    .{ .name = "empty_set", .in = blk: {
        var i = base(.{ 1, 2, 3, 4 }, orig);
        i.n_valid = 0;
        break :blk i;
    }, .want = "err:empty_set" },
    .{ .name = "duplicate_label", .in = blk: {
        var i = base(.{ 1, 2, 3, 4 }, orig);
        i.label_ids[3] = i.label_ids[0];
        break :blk i;
    }, .want = "err:duplicate_label" },
    .{ .name = "label_out_of_vocab", .in = blk: {
        var i = base(.{ 1, 2, 3, 4 }, orig);
        i.label_ids[2] = 262144;
        break :blk i;
    }, .want = "err:label_out_of_vocab" },
    .{ .name = "non_finite", .in = base(.{ 1, std.math.nan(f32), 3, 4 }, orig), .want = "err:non_finite" },
    .{ .name = "argmax_pos0", .in = base(.{ 9, 2, 3, 4 }, orig), .want = "decision:direct" },
    .{ .name = "argmax_pos3_insufficient", .in = base(.{ 1, 2, 3, 9 }, orig), .want = "abstain:insufficient" },
    .{ .name = "argmax_pos1_rev", .in = base(.{ 1, 9, 3, 4 }, rev), .want = "decision:calculate" },
    .{ .name = "argmax_pos2", .in = base(.{ 1, 2, 9, 4 }, orig), .want = "decision:calculate" },
    .{ .name = "argmax_pos3_rev", .in = base(.{ 1, 2, 3, 9 }, rev), .want = "decision:direct" }, // spec C-SD-F : argmax en position 3 → .decision
};

/// Σp = 1 ± 1e-6 pour toute variante qui porte p (vérifié AUSSI par --policy-selftest, binaire livré).
pub fn sumOk(d: Decision) bool {
    const p = switch (d) {
        .decision => |x| x.p,
        .abstain => |x| x.p,
        .err => return true,
    };
    var s: f64 = 0;
    for (p) |v| s += v;
    return @abs(s - 1.0) <= 1e-6;
}

pub fn describe(d: Decision, buf: []u8) []const u8 {
    return switch (d) {
        .decision => |x| std.fmt.bufPrint(buf, "decision:{s}", .{@tagName(x.option)}) catch "?",
        .abstain => |x| std.fmt.bufPrint(buf, "abstain:{s}", .{@tagName(x.reason)}) catch "?",
        .err => |e| std.fmt.bufPrint(buf, "err:{s}", .{@tagName(e)}) catch "?",
    };
}

test "C-SD-F : chaque cas fabriqué rend la variante attendue" {
    var buf: [64]u8 = undefined;
    for (selftest_cases) |c| {
        const got = describe(evaluate(c.in), &buf);
        std.testing.expectEqualStrings(c.want, got) catch |e| {
            std.debug.print("cas {s} : got {s} want {s}\n", .{ c.name, got, c.want });
            return e;
        };
    }
}

test "p somme à 1 et mass_in dans ]0,1]" {
    const d = evaluate(base(.{ 9, 2, 3, 4 }, orig));
    const x = d.decision;
    var s: f32 = 0;
    for (x.p) |v| s += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1), s, 1e-6);
    try std.testing.expect(x.mass_in > 0 and x.mass_in <= 1);
    try std.testing.expect(x.margin > 0 and x.p_max == x.p[0]);
}
