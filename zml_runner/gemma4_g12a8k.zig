// gemma4_g12a8k — variante L_MAX=8192 du décode 12B autonome (sonde 8k du chantier masques
// in-graph, spec docs/superpowers/specs/2026-07-26-masques-ingraph-design.md, gate M3).
//
// Ce fichier N'EST QUE le point d'entrée : tout le runner (CLI, tables RoPE, boucle autonome,
// flags --oracle/--out-ids/--window-vacuity/--no-prealloc) vit dans `gemma4_g12auto.zig` et est
// réutilisé tel quel via `G12Auto(comptime L_MAX)`. La borne est COMPTIME (shapes du cache,
// masques in-graph {k=L_MAX}, tables) : elle change le graphe tracé.
// Pattern du couple bbs/bbatch (piège 18 : paramètre comptime explicite, PAS @import("root")).
//
// Rendue possible par les masques in-graph : les tables {L_MAX,L_MAX} quadratiques (512 MiB
// à 8192) sont supprimées, tout est linéaire en L_MAX. Restent linéaires : caches KV
// (~5,6 GiB f32 K+V), tables RoPE/positions. Chiffres mesurés : gate M3 (résultats du chantier).
//
// Nom COURT (`gemma4_g12a8k`, 13c) : le quota comptime de `pjrt.zig structSize` scanne
// `@typeName` (piège 4).

const std = @import("std");
const g12auto = @import("gemma4_g12auto.zig");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    return g12auto.G12Auto(8192).run(init);
}
