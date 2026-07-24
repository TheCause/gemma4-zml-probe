// G12 — géométrie Gemma 4 12B Unified (jalon J2, plan docs/superpowers/plans/
// 2026-07-24-w4-j2-12b-unified.md, contrat docs/U_12B_CONTRACT.md).
//
// Valeurs = décision D1 (cartographie 24 juil, vérifiée sur pièce — config.json + modeling
// 5.9.0) : 48 couches, pas de YOCO (first_kv_shared = num_layers → aucun reader, chaque
// couche écrit son slot), d 3840, 16 Q ; sliding = GQA 8 KV × 256 (groupe 2), full = MQA
// 1 KV × 512 avec K=V (attention_k_eq_v : V = v_norm(k_proj brut), branche D4 du moteur) ;
// pas de PLE (ple_dim = 0 → bloc comptime-mort) ; couches full aux (i+1) % 6 == 0
// (5, 11, 17, 23, 29, 35, 41, 47).
//
// sliding_writer/full_writer : sans YOCO il n'y a AUCUN reader — ces champs ne sont jamais
// consommés (les branches reader de runLayerGen sont comptime-mortes) ; 0 = valeur inerte.
//
// Ce fichier ne porte QUE la géométrie à ce stade (Task 2) ; les structs G12* (poids packés,
// assemblage EngineModel) arrivent en Task 8.

const engine = @import("engine.zig");

pub const g12: engine.Geom = .{
    .num_layers = 48,
    .first_kv_shared = 48,
    .sliding_writer = 0,
    .full_writer = 0,
    .d = 3840,
    .nh = 16,
    .kvh_sliding = 8,
    .kvh_full = 1,
    .hd_sliding = 256,
    .hd_full = 512,
    .ple_dim = 0,
    .full_period = 6,
    .k_eq_v_full = true,
    .rope_theta_sliding = 1.0e4,
    .softcap = 30.0,
};
