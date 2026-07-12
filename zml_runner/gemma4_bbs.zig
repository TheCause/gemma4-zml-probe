// gemma4_bbs — variante `attn = .sdpa` du banc batché (Phase 2 du chantier batching, gates S2/S3).
//
// Ce fichier N'EST QUE le point d'entrée : tout le runner (CLI, tokenisation, boucle batchée,
// oracles par lane, mesure) vit dans `gemma4_bbatch.zig` et est réutilisé tel quel via
// `runWith(comptime attn, init)`. La seule différence est la variante d'attention du moteur, qui
// est COMPTIME (elle change le graphe tracé). Pattern des couples e1/e2 du repo (deux mains,
// sources partagées par `srcs` Bazel), sans dupliquer une ligne du runner.
//
// ⚠ NE PAS tenter la variante par `@import("root")` depuis gemma4_bbatch.zig : bbs importe
// bbatch, donc lire root depuis bbatch referme une boucle de dépendance (« dependency loop
// detected » au build). Le paramètre comptime explicite est le chemin qui marche.
//
// Nom COURT (`gemma4_bbs`, 11c) : le quota comptime de `pjrt.zig structSize` scanne `@typeName`.
//
// À savoir sur ce binaire (spec §3.4) :
//   - `zml.nn.sdpa` scale K par 1/√hd PAR DÉFAUT ; Gemma 4 a scaling = 1.0 (la norme passe par
//     q_norm) → `engine.zig` passe `.scale = 1.0`. Sans ça, les scores seraient divisés par 16.
//   - la variante N'EST PAS byte-identique au chemin manuel (scale émis sur K même à 1.0, ordre
//     transpose/merge différent) : c'est un A/B **mesuré et oraclé**, pas prouvé.
//   - les familles PrecRt qk_scores/softmax/pv_ctx sont NEUTRALISÉES en mode sdpa (sdpa fait ses
//     propres converts) → les gates de cette variante tournent en fp32 pur.

const std = @import("std");
const bbatch = @import("gemma4_bbatch.zig");

pub fn main(init: std.process.Init) !void {
    return bbatch.runWith(.sdpa, init);
}
