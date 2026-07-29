//! Warpers de sampling — HOST-SIDE, hors du graphe.
//!
//! Fonctions **pures** sur des `f32` nus : aucune dépendance ZML, donc exerçables par
//! `--selftest-sampling` **sans GPU, sans PJRT, sans poids**. C'est ce qui rend praticable la
//! validation en régime déterministe (objectif O1 de la spec) : on itère en secondes.
//!
//! ⚠ CHAQUE FONCTION REPRODUIT UN WARPER HF PRÉCIS, LU À LA SOURCE (transformers 5.14.1).
//! Ne pas « simplifier » une formule : les écarts vivent aux BORDS, et deux d'entre eux
//! produisent des ensembles de survivants **DISJOINTS** de ceux de HF (spec §2, F9/F10). C'est
//! pourquoi les fixtures sont produites par les VRAIS warpers et jamais retranscrites.
//!
//! Ordre de la chaîne, mesuré (F8) — rangs réels sur 26 `append` :
//!   RepetitionPenalty(4) → SuppressTokens(15) → Temperature(17) → TopK(19) → TopP(20)

const std = @import("std");
const gencfg = @import("gencfg.zig");

/// HF : `filter_value = -inf` par défaut sur les deux warpers (F11).
pub const FILTER: f32 = -std.math.inf(f32);

/// Scratch alloué **UNE FOIS** par `run()`, dimensionné au vocab runtime.
/// ⚠ Interdit « aucune allocation par step » (spec §5) : sans ce buffer partagé, 200 tokens ×
/// 20 prompts brasseraient plusieurs Gio et le critère RSS sauterait de bonne foi.
pub const Scratch = struct {
    idx: []u32, // ordre de tri (top-p)
    prob: []f32, // softmax dans l'ordre trié (top-p)
    rm: []bool, // masque « à retirer », dans l'ordre trié (top-p)
    heap: []f32, // min-heap borné à k (top-k)

    pub fn init(allocator: std.mem.Allocator, vocab: usize) !Scratch {
        return .{
            .idx = try allocator.alloc(u32, vocab),
            .prob = try allocator.alloc(f32, vocab),
            .rm = try allocator.alloc(bool, vocab),
            .heap = try allocator.alloc(f32, vocab),
        };
    }

    pub fn deinit(self: *Scratch, allocator: std.mem.Allocator) void {
        allocator.free(self.idx);
        allocator.free(self.prob);
        allocator.free(self.rm);
        allocator.free(self.heap);
        self.* = undefined;
    }
};


/// Configuration de sampling, passée **PAR POINTEUR** aux 3 sites d'appel de `generateOnce`
/// (`gencfg.GenCfg` a le même traitement) : la fonction a déjà 20 paramètres positionnels, et un
/// oubli sur l'un des trois sites rendrait la fonctionnalité silencieusement inopérante en
/// `--repl` — c'est ce que GC10 avait vérifié pour la politique de décodage.
pub const SamplingCfg = struct {
    temperature: f32 = 1.0,
    top_k: u32 = 0,
    top_p: f32 = 1.0,
    min_keep: u32 = 1,
    seed: ?u64 = null,

    scratch: Scratch = undefined,
    work: []f32 = &.{},
    prng: std.Random.DefaultPrng = undefined,

    // Compteurs S2-PONT, publiés en fin de run.
    n_steps_compared: usize = 0,
    n_disagree: usize = 0,
    n_exact_top_ties: usize = 0,

    /// Le CHEMIN B est armé dès qu'un warper est demandé **ou** qu'un tirage l'est. Sans quoi
    /// `--top-k 1` seul — le régime neutre du gate-pont — n'activerait pas le chemin et le gate
    /// n'aurait rien à comparer.
    pub fn pathArmed(self: *const SamplingCfg) bool {
        return self.top_k != 0 or self.top_p < 1.0 or self.temperature != 1.0 or self.seed != null;
    }

    /// Le TIRAGE, lui, est armé **ssi `--seed` est fourni**. Choix explicite : une dérivation du
    /// type « armé ⇔ top_p<1 » armerait le hasard **par effet de bord** d'un réglage de filtrage,
    /// et rendrait `SeedRequired` intestable. Sans seed, la sélection reste un `argmax`.
    pub fn drawArmed(self: *const SamplingCfg) bool {
        return self.seed != null;
    }

    /// Réinitialise l'état PAR PROMPT (spec §5) — c'est la condition de S2-R en `--repl`.
    pub fn resetPerPrompt(self: *SamplingCfg) void {
        if (self.seed) |s| self.prng = std.Random.DefaultPrng.init(s);
    }
};

/// Température — **DIVISION**, jamais `× (1/T)` : l'équivalence est mathématique, pas binaire,
/// et casserait la bit-exactitude attendue des gates.
/// ⚠ HF n'instancie **PAS** ce warper quand `T == 1.0` (F8c) — l'appelant doit faire de même,
/// sinon on introduit une opération que la référence ne fait pas.
pub fn applyTemperature(logits: []f32, t: f32) void {
    for (logits) |*x| x.* = x.* / t;
}

/// top-k — F9 : l'élimination est **STRICTEMENT** sous le k-ième, donc **les ex æquo du k-ième
/// SURVIVENT**. Conséquence contre-intuitive et mesurée : `top_k` **ne borne pas** le nombre de
/// survivants (64 logits égaux avec `k=8` en laissent **64**). Une implémentation « garder les k
/// premiers indices du tri » diverge dès la première égalité.
/// `k_eff` reproduit `__init__` (`max(top_k, min_tokens_to_keep)`) puis le *safety check* de
/// `__call__` (`min(self.top_k, vocab)`).
pub fn applyTopK(logits: []f32, k: u32, min_keep: u32, s: *Scratch) void {
    if (k == 0) return; // convention HF : désactivé
    const k_eff = @min(@max(k, min_keep), logits.len);
    if (k_eff >= logits.len) return;
    const kth = kthLargest(logits, k_eff, s.heap);
    for (logits) |*x| {
        if (x.* < kth) x.* = FILTER; // STRICT — c'est tout le sujet
    }
}

/// Valeur du k-ième plus grand, **SANS MUTER `logits`**. Min-heap borné à k : O(n·log k), une
/// seule passe, zéro allocation.
///
/// ⚠ POURQUOI PAS UN QUICKSELECT IN-PLACE. `logits` est indexé par **ID DE TOKEN** : permuter le
/// tableau casserait la correspondance indice↔token. Le filtrage resterait juste en *multiset*
/// mais `argmax`/`sample` rendraient un **token faux** — et ce bug serait **invisible** à une
/// comparaison par classe d'équivalence. Le selftest compare donc AUSSI les indices.
///
/// ⚠ POURQUOI PAS LA STD. Vérifié par compilation sur le toolchain du projet : `std.sort` n'offre
/// **aucun** `select`/`nth_element` en 0.16-dev — seulement des tris **complets**, à 177,95 % d'un
/// step (spec F16).
fn kthLargest(logits: []const f32, k: usize, heap_buf: []f32) f32 {
    std.debug.assert(k > 0 and k <= logits.len and heap_buf.len >= k);
    const h = heap_buf[0..k];
    @memcpy(h, logits[0..k]);
    var i = k / 2;
    while (i > 0) {
        i -= 1;
        siftDown(h, i);
    }
    for (logits[k..]) |v| {
        if (v > h[0]) {
            h[0] = v;
            siftDown(h, 0);
        }
    }
    return h[0]; // racine du min-heap = le k-ième plus grand
}

fn siftDown(h: []f32, start: usize) void {
    var root = start;
    while (2 * root + 1 < h.len) {
        var child = 2 * root + 1;
        if (child + 1 < h.len and h[child + 1] < h[child]) child += 1;
        if (h[root] <= h[child]) return;
        std.mem.swap(f32, &h[root], &h[child]);
        root = child;
    }
}

/// top-p — F10, reproduit **à la lettre** : HF trie en **ASCENDANT**, cumsum des softmax, retire
/// avec `cum <= 1 - top_p`, et `min_tokens_to_keep` protège la **QUEUE** du tri ascendant (= les
/// plus probables).
///
/// ⚠ Ce n'est **PAS** la formulation habituelle (tri descendant, garder tant que `cum < p`). Les
/// deux diffèrent aux bords : sur 8 logits égaux avec `top_p = 0,25`, HF garde **{6,7}** et la
/// formulation naïve **{0,1}** — ensembles **DISJOINTS**, de même cardinal, de même masse.
///
/// ⚠ Le masque est appliqué **par indice d'origine** (le `scatter` de HF). L'appliquer
/// positionnellement produirait un masque juste dans l'espace TRIÉ et **faux** dans l'espace
/// vocabulaire.
///
/// OPTIMISATION, et sa justification : HF applique `top_k` **AVANT** `top_p` (F8), donc quand
/// `top_k` est armé tous les logits sauf ~k valent déjà `FILTER`. Or dans le tri **ascendant**
/// les `-inf` sont **en tête** et contribuent **0** à la cumsum : les inclure ou les ignorer donne
/// le **même** masque. On ne trie donc que les candidats **non filtrés** — 64 éléments au lieu de
/// 262 144. Quand `top_k` est désactivé, le sous-ensemble vaut tout le vocabulaire.
pub fn applyTopP(logits: []f32, p: f32, min_keep: u32, s: *Scratch) void {
    if (p >= 1.0) return;

    // 1. candidats non filtrés seulement (cf. justification ci-dessus)
    var n: usize = 0;
    for (logits, 0..) |v, i| {
        if (v != FILTER) {
            s.idx[n] = @intCast(i);
            n += 1;
        }
    }
    if (n <= 1) return;

    // 2. ordre ASCENDANT par valeur de logit
    std.mem.sortUnstable(u32, s.idx[0..n], logits, ascByLogit);

    // 3. softmax dans l'ordre ascendant (max soustrait — stabilité numérique)
    const mx = logits[s.idx[n - 1]];
    var sum: f32 = 0;
    for (s.idx[0..n], 0..) |id, r| {
        s.prob[r] = @exp(logits[id] - mx);
        sum += s.prob[r];
    }

    // 4. cumsum + masque `cum <= 1 - top_p` — le `<=` EST le critère (F10), pas un `<`
    const thr = 1.0 - p;
    var cum: f32 = 0;
    for (s.prob[0..n], 0..) |pr, r| {
        cum += pr / sum;
        s.rm[r] = (cum <= thr);
    }

    // 5. min_tokens_to_keep protège la QUEUE du tri ascendant (= les plus probables)
    const keep = @min(@as(usize, min_keep), n);
    for (s.rm[n - keep .. n]) |*b| b.* = false;

    // 6. « scatter » : appliquer PAR INDICE D'ORIGINE
    for (s.idx[0..n], 0..) |id, r| {
        if (s.rm[r]) logits[id] = FILTER;
    }
}

fn ascByLogit(logits: []const f32, a: u32, b: u32) bool {
    return logits[a] < logits[b];
}

/// Suppression `generation_config` — délègue à `gencfg.isSuppressed`. **Pas de seconde
/// implémentation** : la spec du chantier précédent exige une fonction unique, et le chemin
/// penalty appellera la même.
pub fn applySuppression(logits: []f32, policy: *const gencfg.GenCfg) void {
    for (policy.suppress) |id| {
        if (id < logits.len) logits[id] = FILTER;
    }
}

/// argmax host, tie-break explicite : **premier indice gagnant**.
/// ⚠ Ce tie-break n'est pas prouvé équivalent à celui du `topK` in-graph (`gencfg.zig:21-25`) :
/// le gate S2-PONT le **publie** au lieu de le supposer résolu.
pub fn argmax(logits: []const f32) u32 {
    var best: usize = 0;
    for (logits, 0..) |v, i| {
        if (v > logits[best]) best = i;
    }
    return @intCast(best);
}

/// Tirage multinomial sur les logits **filtrés** (les `FILTER` ont une probabilité nulle).
///
/// ⚠ Le RNG est fourni **construit** par l'appelant et **avance** d'un tirage à l'autre — il
/// n'est PAS re-seedé par token. Il est réinitialisé à `seed` **au début de chaque prompt**
/// (spec §5, « état par prompt »), ce qui est la condition de S2-R en `--repl`. Sans cette règle
/// écrite, « re-seeder par token » et « avancer » passent tous deux le gate en donnant des
/// sorties différentes.
///
/// Invariant que l'appelant doit asserter : `logits[résultat] > FILTER`.
pub fn sample(logits: []const f32, rng: std.Random) u32 {
    var mx: f32 = -std.math.inf(f32);
    for (logits) |v| {
        if (v > mx) mx = v;
    }
    var sum: f64 = 0;
    for (logits) |v| {
        if (v != FILTER) sum += @exp(@as(f64, v - mx));
    }
    var r = rng.float(f64) * sum;
    var last: u32 = 0;
    for (logits, 0..) |v, i| {
        if (v == FILTER) continue;
        last = @intCast(i);
        r -= @exp(@as(f64, v - mx));
        if (r <= 0) return @intCast(i);
    }
    return last; // garde anti-arrondi : rend le dernier NON filtré, jamais un token filtré
}
