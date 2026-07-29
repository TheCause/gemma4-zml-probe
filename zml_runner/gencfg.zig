// Politique de décodage `generation_config.json` — HOST-SIDE, hors du graphe.
// Spec : docs/superpowers/specs/2026-07-28-generation-config-design.md (§4.1, §4.2, §4.2bis, §4.3).
//
// POURQUOI CE MODULE EXISTE — finding du 27 juil (docs/FINDING_GENERATION_CONFIG.md) : le portage
// 12B faisait un argmax NU et ne s'arrêtait que sur `<turn|>` (106), alors que Google déclare
// `"suppress_tokens": [258883, 258882]` et TROIS `eos_token_id` `[1, 106, 50]`. Conséquence
// mesurée : le runner émettait `<image|>` (258882) en plein texte, ~1×/500-600 tokens, EN GREEDY.
// Le forward ZML était innocenté par l'A/B (ses logits reproduisent HF) : ce qui manquait était la
// politique de décodage. `69_u8_gen_oracle.py` partageait exactement le même angle mort, donc
// AUCUN gate existant ne pouvait détecter l'écart — d'où l'exigence que runner et oracle changent
// dans le même mouvement.
//
// POURQUOI CÔTÉ HOST, ET POURQUOI C'EST EXACT (§4.2) — le graphe sort déjà un top-5 TRIÉ
// décroissant, rapatrié à chaque step (`topK` = les rangs 1..5 de l'ordre décroissant complet,
// zml/tensor.zig:3096-3110). Avec |S| = 2 supprimés, l'argmax post-suppression est le premier
// élément de cet ordre hors S ; au plus 2 éléments sont retirés au-dessus de lui, donc son rang
// BRUT est ≤ 3 : il est déjà dans le top-5 rapatrié. La garde `suppress.len + 1 > TOP_K` rend cet
// argument vrai PAR CONSTRUCTION plutôt que par chance. Corollaire : `engine.zig` reste à 0 octet,
// `G12Step.forward` ne bouge pas, le StableHLO est byte-identique (gate GC0), zéro D2H de plus.
//
// ⚠ CE QUE CE DESIGN NE REPRODUIT PAS : HF écrit `-inf` puis fait `argmax` sur le vecteur COMPLET
// (logits_process.py:1906-1911, utils.py:2938). Nous sélectionnons dans un top-5 pré-trié. Les
// deux coïncident SAUF en cas d'égalité exacte, où le tie-break de `sort` peut différer de celui
// d'`argmax` (piège 15 ; le tie-break de `torch.argmax` n'est pas vérifié). Le selftest GC1
// COMPTE les égalités rencontrées — la réserve est instrumentée, pas seulement mentionnée.
//
// ⚠ CE MODULE N'APPLIQUE QUE 2 CLÉS SUR 8. `do_sample: true, top_k: 64, top_p: 0.95,
// temperature: 1.0` sont la configuration NOMINALE du modèle et ne sont PAS appliqués ici : dire
// « generation_config.json est appliqué » tout court serait faux. D'où le segment `ignored=[…]`
// du log, dérivé des clés effectivement présentes et jamais codé en dur.

const std = @import("std");
const log = std.log;

/// Taille du top-K rapatrié du device. DÉCLARATION UNIQUE (§4.2) : le type `Top5`, le `topK`
/// in-graph et la boucle de lecture du runner la référencent tous les trois, au lieu d'en garder
/// trois copies du littéral `5`. Le remplacement in-graph est un comptime de MÊME VALEUR, donc le
/// StableHLO émis est inchangé — et GC0 le prouve au lieu de le supposer.
pub const TOP_K: usize = 5;

/// Résultat de la sélection : le token retenu et son rang DANS LE TOP-5 BRUT.
/// `rank` n'est pas un détail de journal : c'est l'instrument de la claim C2 (histogramme des
/// rangs utilisés) et ce qui rend la marge de requalification des gates U8/W4g interprétable
/// quand la suppression mord (sans lui, la marge top1−top2 affichée ne parle plus du token choisi).
pub const Selected = struct { tok: usize, rank: usize };

/// Paramètres de validation que le module ne peut pas mesurer lui-même.
/// - `vocab_size` : borne des ids de suppression. HF ignorerait silencieusement un id hors bornes
///   (`isin` sur `arange(V)`) ; ici on REFUSE — un id hors vocab dans un fichier de politique est
///   un fichier faux, pas une nuance.
/// - `eot_id` : MESURÉ depuis le tokenizer par l'appelant (jamais hardcodé), pour le contrôle
///   croisé tokenizer ↔ config.
pub const Options = struct {
    vocab_size: u32,
    eot_id: u32,
};

pub const GenCfg = struct {
    /// Chemin résolu, pour le log. "" quand la politique est désactivée.
    path: []const u8,
    /// false ⇔ `--no-gen-config` : restaure EXACTEMENT le comportement d'avant le chantier.
    /// Ce n'est pas une commodité mais l'instrument du contre-test de non-vacuité GC4(a) :
    /// même binaire, une donnée de moins.
    enabled: bool,
    /// Trié, dédupliqué, borné [0, vocab_size).
    suppress: []const u32,
    /// Non vide, et contient `eot_id`.
    eos: []const u32,
    /// Clés PRÉSENTES dans le fichier mais NON appliquées par ce chantier (log).
    ignored: []const []const u8,

    pub fn deinit(self: *GenCfg, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.suppress);
        allocator.free(self.eos);
        for (self.ignored) |k| allocator.free(k);
        allocator.free(self.ignored);
        self.* = undefined;
    }

    /// Forme d'appel UNIQUE (§4.2bis, exigence §4.7) : le chemin repetition-penalty appellera la
    /// même. `id` est un `usize` parce que `Top5.idx` en est un — pas de conversion à l'appel.
    pub fn isSuppressed(self: *const GenCfg, id: usize) bool {
        for (self.suppress) |s| {
            if (s == id) return true;
        }
        return false;
    }

    /// Sémantique HF exacte (§4.3) : `EosTokenCriteria` = `isin(dernier_token, eos)` — un « any of »
    /// SANS priorité (stopping_criteria.py:534, 579-581), combiné en OU avec les autres critères.
    /// Le token est concaténé PUIS testé (utils.py:2945 puis :2949) : l'EOS fait partie de la
    /// sortie — le runner le conserve déjà et le strippe à la détok, seul l'élargissement 1 → 3
    /// était à faire.
    pub fn isEos(self: *const GenCfg, id: i64) bool {
        if (id < 0) return false;
        const u: u64 = @intCast(id);
        for (self.eos) |e| {
            if (e == u) return true;
        }
        return false;
    }

    /// LA sélection (§4.2). Le top-5 est trié décroissant : le premier élément non supprimé EST
    /// l'argmax post-suppression du vecteur complet, sous la garde `suppress.len + 1 <= TOP_K`.
    ///
    /// ⚠ Cette fonction vit ICI et non recopiée dans le runner, pour que GC1 exerce le code
    /// LIVRÉ et non un jumeau : un selftest qui teste sa propre copie est un instrument aveugle
    /// au même endroit que son sujet — c'est exactement le piège qui a laissé passer le finding
    /// du 27 juil (`69_u8_gen_oracle.py` faisait le même argmax nu que le runner).
    ///
    /// `error.AllTopKSuppressed` est INATTEIGNABLE sous la garde, et existe pour qu'un futur
    /// élargissement de la liste échoue BRUYAMMENT au lieu de choisir un token au hasard.
    pub fn select(self: *const GenCfg, idx: []const usize) error{AllTopKSuppressed}!Selected {
        for (idx, 0..) |id, j| {
            if (!self.isSuppressed(id)) return .{ .tok = id, .rank = j };
        }
        return error.AllTopKSuppressed;
    }
};

/// `--no-gen-config` : politique inerte, mais l'arrêt sur `eot_id` est conservé — c'est le
/// comportement d'AVANT le chantier, exactement (et pas « aucun arrêt du tout », qui serait un
/// troisième comportement et ruinerait l'A/B à un seul facteur de GC4(a)).
pub fn disabled(allocator: std.mem.Allocator, eot_id: u32) !GenCfg {
    const eos = try allocator.alloc(u32, 1);
    errdefer allocator.free(eos);
    eos[0] = eot_id;
    return .{
        .path = try allocator.dupe(u8, ""),
        .enabled = false,
        .suppress = try allocator.alloc(u32, 0),
        .eos = eos,
        .ignored = try allocator.alloc([]const u8, 0),
    };
}

/// Constructeur VALIDANT — le seul point d'entrée qui applique les six règles de la §4.1.
/// `parseFromSlice` s'y ramène après extraction JSON, et le selftest GC1 l'appelle directement
/// pour bâtir la politique de ses cas de sélection : une seule implémentation des validations.
pub fn fromLists(
    allocator: std.mem.Allocator,
    path: []const u8,
    suppress_raw: []const u32,
    eos_raw: []const u32,
    ignored_keys: []const []const u8,
    opts: Options,
) !GenCfg {
    // — eos : vide ⇒ refus. Un décodage sans aucun EOS ne s'arrêterait que sur `max_tokens`,
    //   ce qui n'est pas un arrêt mais une troncature.
    if (eos_raw.len == 0) {
        log.err("GENCFG: {s} — `eos_token_id` vide ou absent : un décodage sans EOS ne s'arrête jamais", .{path});
        return error.EosListEmpty;
    }

    // — contrôle croisé tokenizer ↔ config. Exercé DANS LES DEUX SENS par GC1 (un cas où l'eot
    //   est présent et doit être accepté, un cas où il est absent et doit être refusé).
    var eot_found = false;
    for (eos_raw) |e| {
        if (e == opts.eot_id) eot_found = true;
    }
    if (!eot_found) {
        log.err("GENCFG: {s} — eot_id={d} MESURÉ au tokenizer absent de eos_token_id={any} : tokenizer et config ne parlent pas du même modèle", .{ path, opts.eot_id, eos_raw });
        return error.EotNotInEosList;
    }

    // — suppress : bornes, puis tri, puis déduplication. Les doublons n'ont aucun effet sur HF
    //   mais fausseraient la garde `len + 1 > TOP_K` ci-dessous, donc ils sont retirés AVANT elle.
    for (suppress_raw) |s| {
        if (s >= opts.vocab_size) {
            log.err("GENCFG: {s} — id de suppression {d} hors vocab [0,{d}) : HF l'ignorerait en silence, nous le refusons", .{ path, s, opts.vocab_size });
            return error.SuppressIdOutOfRange;
        }
    }
    const sorted = try allocator.dupe(u32, suppress_raw);
    defer allocator.free(sorted);
    std.mem.sort(u32, sorted, {}, std.sort.asc(u32));

    var dedup = try allocator.alloc(u32, sorted.len);
    errdefer allocator.free(dedup);
    var n: usize = 0;
    for (sorted) |s| {
        if (n == 0 or dedup[n - 1] != s) {
            dedup[n] = s;
            n += 1;
        }
    }
    if (n != sorted.len) {
        log.warn("GENCFG: {s} — {d} doublon(s) dans suppress_tokens, dédupliqué(s) ({d} → {d})", .{ path, sorted.len - n, sorted.len, n });
    }
    dedup = try allocator.realloc(dedup, n);

    // — LA garde qui rend l'argument d'exactitude du §4.2 vrai par construction : avec |S|
    //   supprimés, l'argmax post-suppression est de rang brut ≤ |S| + 1 ; il faut qu'il tienne
    //   dans le top-K rapatrié. Au-delà, la sélection host serait un pari sur des logits qu'on
    //   n'a pas.
    if (n + 1 > TOP_K) {
        log.err("GENCFG: {s} — {d} ids à supprimer : l'argmax post-suppression peut être de rang {d} > TOP_K={d}, il ne serait pas dans le top-K rapatrié", .{ path, n, n + 1, TOP_K });
        return error.SuppressListTooLongForTopK;
    }
    if (n == 0) {
        log.info("GENCFG: {s} — suppress=[] (aucune suppression) : politique INERTE de ce côté, aucun gate de mordant ne peut être déclaré PASS avec ce fichier", .{path});
    }

    const eos = try allocator.dupe(u32, eos_raw);
    errdefer allocator.free(eos);
    const ignored = try allocator.alloc([]const u8, ignored_keys.len);
    errdefer allocator.free(ignored);
    for (ignored_keys, 0..) |kk, i| ignored[i] = try allocator.dupe(u8, kk);

    return .{
        .path = try allocator.dupe(u8, path),
        .enabled = true,
        .suppress = dedup,
        .eos = eos,
        .ignored = ignored,
    };
}

/// Clés que ce chantier applique. Toute AUTRE clé présente au fichier atterrit dans `ignored`,
/// pour que le log dise la vérité sur ce qui est réellement appliqué (2 sur 8).
const APPLIED_KEYS = [_][]const u8{ "suppress_tokens", "eos_token_id" };

/// Parse le CONTENU d'un `generation_config.json` (texte, pas chemin — le selftest GC1 exerce
/// ainsi le parser sur du texte réel, y compris malformé, sans passer par le disque).
/// Patron de parsing repris de `gemma4_bbatch.zig:406-426` : `std.json` est INCHANGÉ dans cette
/// toolchain (std/json/static.zig:73 ; seul `std.json.Stringify`, côté écriture, a bougé).
pub fn parseFromSlice(allocator: std.mem.Allocator, text: []const u8, path: []const u8, opts: Options) !GenCfg {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{ .allocate = .alloc_always }) catch |err| {
        log.err("GENCFG: {s} — JSON illisible : {s}", .{ path, @errorName(err) });
        return error.GenerationConfigInvalid;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        log.err("GENCFG: {s} — la racine JSON n'est pas un objet", .{path});
        return error.GenerationConfigInvalid;
    }
    const root = parsed.value.object;

    // — `begin_suppress_tokens` : sémantique TEMPORELLE différente (suppression au premier step
    //   seulement, logits_process.py:1863-1871). Non implémentée ⇒ refus explicite, pas une
    //   application approximative qui produirait des ids faux au premier token.
    if (root.get("begin_suppress_tokens") != null) {
        log.err("GENCFG: {s} — `begin_suppress_tokens` présente : sémantique temporelle non implémentée par ce chantier", .{path});
        return error.BeginSuppressUnsupported;
    }

    var suppress_raw: []u32 = &.{};
    defer if (suppress_raw.len != 0) allocator.free(suppress_raw);
    if (root.get("suppress_tokens")) |v| {
        if (v != .array) {
            log.err("GENCFG: {s} — `suppress_tokens` n'est pas une liste", .{path});
            return error.GenerationConfigInvalid;
        }
        suppress_raw = try allocator.alloc(u32, v.array.items.len);
        for (v.array.items, 0..) |item, i| {
            if (item != .integer) {
                log.err("GENCFG: {s} — `suppress_tokens[{d}]` n'est pas un entier", .{ path, i });
                return error.GenerationConfigInvalid;
            }
            // Négatif : refusé au même titre qu'un id ≥ vocab (§4.1). Le test vit ICI parce que
            // `fromLists` reçoit des u32 — un @intCast d'un négatif y arriverait déjà corrompu.
            if (item.integer < 0) {
                log.err("GENCFG: {s} — id de suppression négatif ({d}) : HF l'ignorerait en silence, nous le refusons", .{ path, item.integer });
                return error.SuppressIdOutOfRange;
            }
            if (item.integer >= opts.vocab_size) {
                log.err("GENCFG: {s} — id de suppression {d} hors vocab [0,{d})", .{ path, item.integer, opts.vocab_size });
                return error.SuppressIdOutOfRange;
            }
            suppress_raw[i] = @intCast(item.integer);
        }
    }

    // — `eos_token_id` : scalaire OU liste. La normalisation `int → [int]` reproduit exactement
    //   celle de HF (stopping_criteria.py:544-549), au lieu de la deviner.
    var eos_raw: []u32 = &.{};
    defer if (eos_raw.len != 0) allocator.free(eos_raw);
    if (root.get("eos_token_id")) |v| {
        switch (v) {
            .integer => |one| {
                if (one < 0 or one >= opts.vocab_size) {
                    log.err("GENCFG: {s} — eos_token_id {d} hors vocab [0,{d})", .{ path, one, opts.vocab_size });
                    return error.GenerationConfigInvalid;
                }
                eos_raw = try allocator.alloc(u32, 1);
                eos_raw[0] = @intCast(one);
            },
            .array => |arr| {
                eos_raw = try allocator.alloc(u32, arr.items.len);
                for (arr.items, 0..) |item, i| {
                    if (item != .integer or item.integer < 0 or item.integer >= opts.vocab_size) {
                        log.err("GENCFG: {s} — `eos_token_id[{d}]` invalide ou hors vocab [0,{d})", .{ path, i, opts.vocab_size });
                        return error.GenerationConfigInvalid;
                    }
                    eos_raw[i] = @intCast(item.integer);
                }
            },
            else => {
                log.err("GENCFG: {s} — `eos_token_id` n'est ni un entier ni une liste", .{path});
                return error.GenerationConfigInvalid;
            },
        }
    }

    // — `ignored` DÉRIVÉ des clés présentes, jamais codé en dur : si Google ajoute une clé
    //   demain, le log la nommera sans qu'on ait à y penser.
    var ignored: std.ArrayList([]const u8) = .empty;
    defer {
        for (ignored.items) |kk| allocator.free(kk);
        ignored.deinit(allocator);
    }
    var it = root.iterator();
    while (it.next()) |entry| {
        var applied = false;
        for (APPLIED_KEYS) |ak| {
            if (std.mem.eql(u8, ak, entry.key_ptr.*)) applied = true;
        }
        if (!applied) try ignored.append(allocator, try allocator.dupe(u8, entry.key_ptr.*));
    }
    // Ordre déterministe : le log est grepé par les gates, il ne doit pas dépendre de l'ordre
    // d'itération d'une hash map.
    std.mem.sort([]const u8, ignored.items, {}, lessThanStr);

    return fromLists(allocator, path, suppress_raw, eos_raw, ignored.items, opts);
}

fn lessThanStr(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

const CONFIG_NAME = "generation_config.json";

/// Découverte du fichier (§4.1) — miroir de `scripts/63_u_dequant_export.py:70-71`, qui traite
/// déjà ce cas en Python avec le commentaire « UN seul hop, `realpath` interdit ».
///
/// POURQUOI UN SEUL HOP : le runner reçoit `weights_12b/model.safetensors`, un répertoire qui ne
/// contient que deux symlinks ; le fichier vit dans le SNAPSHOT pointé. Un `realpath` COMPLET
/// atterrirait dans `blobs/<sha256>`, qui ne le contient pas non plus. Un hop, et un seul.
///
/// Rend un chemin ALLOUÉ (à libérer par l'appelant). Aucun repli silencieux : un runner qui ne
/// trouve pas sa politique refuse de tourner.
pub fn discoverAlloc(allocator: std.mem.Allocator, io: std.Io, explicit: ?[]const u8, ckpt: []const u8) ![]u8 {
    // 1. `--gen-config <FICHIER>` — un chemin de FICHIER, jamais un répertoire (message
    //    univoque : c'est l'erreur qu'on fera au premier usage pressé, cf D11).
    if (explicit) |p| {
        var f = std.Io.Dir.cwd().openFile(io, p, .{ .mode = .read_only }) catch |err| {
            log.err("GENCFG: --gen-config {s} illisible ({s}) — attendu un CHEMIN DE FICHIER (…/generation_config.json), pas un répertoire", .{ p, @errorName(err) });
            return error.GenerationConfigNotFound;
        };
        f.close(io);
        return allocator.dupe(u8, p);
    }

    const ckpt_dir = std.fs.path.dirname(ckpt) orelse ".";

    // 2. à côté du checkpoint.
    const direct = try std.fs.path.join(allocator, &.{ ckpt_dir, CONFIG_NAME });
    errdefer allocator.free(direct);
    if (std.Io.Dir.cwd().openFile(io, direct, .{ .mode = .read_only })) |f_| {
        var f = f_;
        f.close(io);
        return direct;
    } else |_| {}

    // 3. UN hop de symlink sur le checkpoint lui-même.
    var buf: [std.posix.PATH_MAX]u8 = undefined;
    const n = std.Io.Dir.cwd().readLink(io, ckpt, &buf) catch |err| switch (err) {
        // EINVAL (la cible n'est pas un lien) → NotLink ; ENOENT → FileNotFound. Les deux
        // signifient « pas de second chemin à essayer », pas « erreur système ».
        // ⚠ PAS de `allocator.free(direct)` ici : l'`errdefer` ci-dessus le libère déjà sur le
        // chemin d'erreur. Le double-free a été attrapé par le DebugAllocator au premier run de
        // GC1 — un cas d'erreur qu'aucun run nominal n'aurait exercé.
        error.NotLink, error.FileNotFound => {
            log.err("GENCFG: {s} introuvable — essayé : (2) {s} ; (3) 1 hop depuis {s} (pas un lien)", .{ CONFIG_NAME, direct, ckpt });
            return error.GenerationConfigNotFound;
        },
        else => return err,
    };
    const target = buf[0..n];

    // ⚠ La cible peut être RELATIVE : dans le cache HF les entrées de snapshot le sont
    // (`tokenizer.json -> ../../blobs/cc8d3a0c…`), alors que `weights_12b/model.safetensors` est
    // ABSOLUE (les deux cas mesurés). Résoudre le relatif contre `dirname(ckpt)`, sinon on
    // refuserait un jour un checkpoint parfaitement valide.
    // `resolve` et non `join` : `join` produirait `…/weights_12b/../snap/generation_config.json`,
    // qui S'OUVRE très bien mais n'est pas la même CHAÎNE que le chemin attendu — un gate qui
    // compare des chemins échouerait alors sur une différence d'écriture, pas de résolution.
    const target_abs = try std.fs.path.resolve(allocator, &.{ ckpt_dir, target });
    defer allocator.free(target_abs);

    const snap_dir = std.fs.path.dirname(target_abs) orelse ".";
    const hop = try std.fs.path.join(allocator, &.{ snap_dir, CONFIG_NAME });
    errdefer allocator.free(hop);
    if (std.Io.Dir.cwd().openFile(io, hop, .{ .mode = .read_only })) |f_| {
        var f = f_;
        f.close(io);
        allocator.free(direct);
        return hop;
    } else |_| {}

    // Idem : les deux `errdefer` libèrent `direct` et `hop` sur ce return d'erreur.
    log.err("GENCFG: {s} introuvable — essayé : (2) {s} ; (3) {s} (1 hop depuis {s})", .{ CONFIG_NAME, direct, hop, ckpt });
    return error.GenerationConfigNotFound;
}

/// Découverte + lecture + validations. C'est ce que le runner appelle (Task 3), APRÈS l'eot_id
/// mesuré au tokenizer et AVANT la garde VRAM : un fichier de politique invalide doit coûter une
/// seconde, pas une compile GPU.
pub fn load(allocator: std.mem.Allocator, io: std.Io, explicit: ?[]const u8, ckpt: []const u8, opts: Options) !GenCfg {
    const path = try discoverAlloc(allocator, io, explicit, ckpt);
    defer allocator.free(path);

    var f = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only }) catch |err| {
        log.err("GENCFG: {s} illisible : {s}", .{ path, @errorName(err) });
        return error.GenerationConfigNotFound;
    };
    defer f.close(io);
    const len: usize = @intCast(try f.length(io));
    const text = try allocator.alloc(u8, len);
    defer allocator.free(text);
    if (try f.readPositionalAll(io, text, 0) != len) return error.ShortRead;

    return parseFromSlice(allocator, text, path, opts);
}
