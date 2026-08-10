//! Référence INDÉPENDANTE des warpers top-p / température — gates G-D1 et G-D2 (dettes D1/D2).
//! Spec : `docs/superpowers/specs/2026-08-10-d1d2-gpu-coverage.md`.
//!
//! ⚠ CE FICHIER N'EST PAS UNE SECONDE IMPLÉMENTATION DE PRODUCTION. Il n'est appelé que par le
//! pont de gate, jamais par le chemin qui décide un token. Sa seule raison d'être : donner à
//! `sampling.applyTopP` un contradicteur écrit AUTREMENT.
//!
//! POURQUOI « AUTREMENT » EST LA CONDITION DU GATE. `sampling.applyTopP` trie **ascendant** et
//! retire ssi `cum <= 1 - p` (formulation littérale de HF). Une référence qui referait la même
//! chose comparerait le code à lui-même. Celle-ci trie **descendant** et garde ssi la cumsum
//! **exclusive** des probas est `< p` — le critère « nucleus » direct. Équivalence algébrique :
//! au rang descendant `d`, la cumsum ascendante du même token vaut `1 - cumdesc_excl(d)`, donc
//! `cumasc <= 1-p  ⟺  cumdesc_excl >= p`. Les deux chemins sont réellement distincts.
//!
//! ⚠ POURQUOI f64, ET CE QUE ÇA COÛTE EN HONNÊTETÉ. Sommer les probabilités des plus grandes
//! vers les plus petites (descendant) ne donne PAS le même f32 que l'inverse : l'addition
//! flottante n'est pas associative. Une référence descendante en f32 divergerait donc parfois
//! pour une raison purement numérique, et le gate n'aurait aucun moyen de distinguer ce bruit
//! d'un vrai désaccord structurel. La référence somme donc en **f64** (~1e-16 de bruit, contre
//! ~1e-7 en f32) et publie en plus `n_boundary_tight` / `n_boundary_ties` : si un désaccord
//! survient, on saura s'il est numérique ou structurel — au lieu de le supposer.

const std = @import("std");
const sampling = @import("sampling.zig");

/// Les BUFFERS et COMPTEURS du gate vivent dans `sampling.zig` (`sampling.GateD1D2`), pas ici :
/// `SamplingCfg` doit pouvoir les porter, et `sampling.zig` ne peut pas importer ce fichier sans
/// créer un cycle (ce fichier importe `sampling.zig` pour les warpers testés). Séparation
/// assumée : les DONNÉES du gate avec la config, les ALGORITHMES de référence ici.
pub const RefScratch = sampling.GateD1D2;

/// Verdict d'un step, publié tel quel (aucun agrégat qui masquerait un cas).
pub const TopPVerdict = struct {
    /// Nombre d'ids où référence et implémentation ne s'accordent pas sur « survivant ou non ».
    n_disagree: usize,
    /// Nombre d'ids retranchés par top-p APRÈS top-k, selon la référence.
    /// C'est l'ANTÉCÉDENT du gate : à 0, le gate est passé À VIDE (spec §4, G-D1).
    n_cut: usize,
    /// Candidats non filtrés soumis à top-p (après top-k).
    n_candidates: usize,
    /// Ids dont la cumsum exclusive tombe à moins de 1e-9 du seuil `p` : zone où f32 et f64
    /// peuvent légitimement trancher différemment.
    n_boundary_tight: usize,
    /// Ids à valeur de logit STRICTEMENT égale à celle de leur voisin de rang, à la frontière :
    /// l'ordre entre eux n'est pas déterminé par la valeur (tri instable des deux côtés).
    n_boundary_ties: usize,
};

fn descByLogit(logits: []const f32, a: u32, b: u32) bool {
    return logits[a] > logits[b];
}

/// Calcule l'ensemble des survivants de top-p sur `logits` (vecteur DÉJÀ passé par suppression,
/// température et top-k — c'est-à-dire l'entrée exacte de `sampling.applyTopP`).
/// Ne mute PAS `logits`. Remplit `s.keep`.
pub fn refTopPKeep(logits: []const f32, p: f32, min_keep: u32, s: *RefScratch) TopPVerdict {
    var v: TopPVerdict = .{ .n_disagree = 0, .n_cut = 0, .n_candidates = 0, .n_boundary_tight = 0, .n_boundary_ties = 0 };

    // Par défaut : tout id filtré en amont reste hors de l'ensemble ; tout candidat survit.
    var n: usize = 0;
    for (logits, 0..) |val, i| {
        const is_cand = val != sampling.FILTER;
        s.keep[i] = is_cand;
        if (is_cand) {
            s.idx[n] = @intCast(i);
            n += 1;
        }
    }
    v.n_candidates = n;

    // HF n'agit pas quand p >= 1 ; idem si 0 ou 1 candidat (rien à retrancher).
    if (p >= 1.0 or n <= 1) return v;

    // Tri DESCENDANT — l'autre sens que l'implémentation testée.
    std.mem.sortUnstable(u32, s.idx[0..n], logits, descByLogit);

    // Softmax en f64, max soustrait (le max est en TÊTE dans l'ordre descendant).
    const mx: f64 = @floatCast(logits[s.idx[0]]);
    var sum: f64 = 0;
    for (s.idx[0..n], 0..) |id, r| {
        s.prob[r] = @exp(@as(f64, @floatCast(logits[id])) - mx);
        sum += s.prob[r];
    }

    // Critère nucleus DIRECT : garder ssi la cumsum EXCLUSIVE (somme des probas strictement plus
    // grandes) est < p. `min_keep` protège la TÊTE du tri descendant (= les plus probables),
    // là où l'implémentation ascendante protège sa queue : même ensemble, écrit à l'envers.
    const keep_head = @min(@as(usize, min_keep), n);
    var cum_excl: f64 = 0;
    const thr: f64 = @floatCast(p);
    for (s.idx[0..n], 0..) |id, r| {
        const protected = r < keep_head;
        const survives = protected or (cum_excl < thr);
        if (!survives) {
            s.keep[id] = false;
            v.n_cut += 1;
        }
        // Zone d'incertitude numérique : à publier, jamais à absorber en silence.
        const gap = @abs(cum_excl - thr);
        if (gap < 1e-9) {
            v.n_boundary_tight += 1;
            if (r > 0 and logits[id] == logits[s.idx[r - 1]]) v.n_boundary_ties += 1;
        }
        cum_excl += s.prob[r] / sum;
    }
    return v;
}

/// Compare l'ensemble des survivants de la référence à celui de l'implémentation testée.
/// `after_impl` est le vecteur APRÈS `sampling.applyTopP` (muté en place).
/// `first_bad` reçoit l'id du premier désaccord, pour que le refus soit NOMMÉ et pas un compte nu.
pub fn compareKeep(after_impl: []const f32, s: *RefScratch, first_bad: *i64) usize {
    var n_bad: usize = 0;
    first_bad.* = -1;
    for (after_impl, 0..) |val, i| {
        s.keep_impl[i] = val != sampling.FILTER;
        if (s.keep_impl[i] != s.keep[i]) {
            if (first_bad.* < 0) first_bad.* = @intCast(i);
            n_bad += 1;
        }
    }
    return n_bad;
}

/// G-D2, mutant (a) : la température doit être une DIVISION, jamais `× (1/t)`.
/// Rend le nombre de logits pour lesquels les deux formulations diffèrent d'au moins 1 ULP.
/// À 0, le mutant ne mord pas et G-D2 ne prouve RIEN sur ce point — à publier tel quel.
pub fn tempDivVsMulDiffs(before: []const f32, after_impl: []const f32, t: f32) usize {
    const inv: f32 = 1.0 / t;
    var n: usize = 0;
    for (before, after_impl) |b, a| {
        if (b == sampling.FILTER) continue; // -inf / t == -inf * inv : jamais discriminant
        if (a != b * inv) n += 1;
    }
    return n;
}

/// G-D2, mutant (b) : l'ORDRE de la chaîne HF (décision Régis du 10 août).
///
/// ⚠ LA FORME NAÏVE DE CE MUTANT EST VACUE, ET C'EST DÉMONTRABLE. Déplacer la température
/// « après top-k » ne change RIEN : diviser par `t > 0` est monotone croissante, et le critère de
/// `applyTopK` (`x < kth`) est un pur ordre — même ensemble avant ou après division, et
/// `FILTER = -inf` divisé reste `-inf`. Un tel mutant passerait toujours, et on en conclurait à
/// tort que le gate garde l'ordre : un contrôle qui ne peut pas échouer n'est pas un contrôle.
///
/// Le mutant retenu est donc `TopK → TopP → Temp` (température APRÈS top-p) : `applyTopP` calcule
/// alors sa softmax sur des logits NON divisés — distribution moins piquée à `t < 1` — donc un
/// masque différent. `pre_temp` est l'état post-suppression, entrée commune aux deux chaînes.
/// Rend le nombre d'ids où l'ensemble des survivants du mutant diffère du nominal.
/// À 0 sur TOUS les steps, G-D2 ne prouve pas l'ordre : à publier tel quel.
pub fn orderMutantDiffs(
    pre_temp: []const f32,
    nominal_after: []const f32,
    t: f32,
    top_k: u32,
    top_p: f32,
    min_keep: u32,
    s: *RefScratch,
    scratch: *sampling.Scratch,
) usize {
    @memcpy(s.mut, pre_temp);
    sampling.applyTopK(s.mut, top_k, min_keep, scratch);
    sampling.applyTopP(s.mut, top_p, min_keep, scratch);
    if (t != 1.0) sampling.applyTemperature(s.mut, t);
    var n: usize = 0;
    for (s.mut, nominal_after) |m, nom| {
        if ((m != sampling.FILTER) != (nom != sampling.FILTER)) n += 1;
    }
    return n;
}
