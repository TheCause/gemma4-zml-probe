// gemma4_g12a4k — variante L_MAX=4096 du décode 12B autonome (probe contexte long, 26 juil 2026).
//
// Ce fichier N'EST QUE le point d'entrée : tout le runner (CLI, tables RoPE/masques, boucle
// autonome, flags --oracle/--out-ids/--window-vacuity/--no-prealloc) vit dans
// `gemma4_g12auto.zig` et est réutilisé tel quel via `G12Auto(comptime L_MAX)`. La borne est
// COMPTIME (shapes du cache, masques {L_MAX,L_MAX}, tables) : elle change le graphe tracé.
// Pattern du couple bbs/bbatch (piège 18 : paramètre comptime explicite, PAS @import("root")).
//
// Chiffres du probe (== HF-fp32 STRICT 4000/4000 teacher-forcé, marge min 0.81) :
//   - pic VRAM réel 22 234 MiB (--no-prealloc) — 90 % d'une 3090 : les masques {L_MAX,L_MAX}
//     sont QUADRATIQUES, 8192 ne rentre pas avec ce design (piste : masques in-graph) ;
//   - 8,2 tok/s stable sur 4041 positions (−9 % vs le défaut 1280).
//
// Nom COURT (`gemma4_g12a4k`, 13c) : le quota comptime de `pjrt.zig structSize` scanne
// `@typeName` (piège 4).

const std = @import("std");
const g12auto = @import("gemma4_g12auto.zig");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    return g12auto.G12Auto(4096).run(init);
}
