// gemma4_g12a8k — variante L_MAX=8192 du décode 12B autonome (sonde 8k du chantier masques
// in-graph, spec docs/superpowers/specs/2026-07-26-masques-ingraph-design.md, gate M3).
//
// Ce fichier N'EST QUE le point d'entrée : tout le runner (CLI, tables RoPE, boucle autonome,
// flags --oracle/--out-ids/--window-vacuity/--no-prealloc) vit dans `gemma4_g12auto.zig` et est
// réutilisé tel quel via `G12Auto(comptime L_MAX)`. La borne est COMPTIME (shapes du cache,
// masques in-graph {k=L_MAX}, tables) : elle change le graphe tracé.
// Pattern du couple bbs/bbatch (piège 18 : paramètre comptime explicite, PAS @import("root")).
//
// Masques in-graph : les tables {L_MAX,L_MAX} quadratiques (512 MiB à 8192) sont supprimées,
// tout est linéaire en L_MAX. ⚠ Verdict sonde M3 (26 juil 2026) : la COMPILE passe (38,9 s)
// mais l'EXÉCUTION OOM au premier step sur 24 Go — le mur suivant est le DOUBLE-BUFFERING des
// caches KV (2 × 5,6 GiB : pas de donation input→output des buffers de cache). Pistes :
// donation PJRT, ou ring sliding 1024. Cf docs/MASKS_INGRAPH_RESULTS.md §M3. La cible reste
// compilable comme sonde pour ces chantiers.
//
// Nom COURT (`gemma4_g12a8k`, 13c) : le quota comptime de `pjrt.zig structSize` scanne
// `@typeName` (piège 4).

const std = @import("std");
const g12auto = @import("gemma4_g12auto.zig");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    return g12auto.G12Auto(8192).run(init);
}
