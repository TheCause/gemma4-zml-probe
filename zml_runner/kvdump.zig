// zml_runner/kvdump.zig — dump/restore de l'état de génération 12B.
// Spec : docs/superpowers/specs/2026-08-09-kv-cache-dump-restore-design.md §4.1.
// Std-only (pas de zml) : testable host, réutilisable par les 3 cibles.
//
// Format : UN safetensors auto-décrivant — `__metadata__` (manifest, strings) + 5 tenseurs
// (sl_k, sl_v, fl_k, fl_v en F32, ids_fed en I32). Le writer généralise
// `writeIdsSafetensors` (gemma4_g12auto.zig) : header JSON, longueur u64 LE, données brutes.
// Aucune API non prouvée dans ce repo : string-building par allocPrint/appendSlice (patron
// joinKeys), parsing par `std.json.parseFromSliceLeaky` sur une arena (cf readHeader),
// I/O par `writePositionalAll`/`readPositionalAll`/`length` (signatures confirmées Task 0.5
// contre lib/std/Io/File.zig:400,656,687 du SDK Zig 0.16.0-dev.2722).
const std = @import("std");
const log = std.log;

pub const FORMAT = "g12-kvdump-v1";

pub const TensorOut = struct {
    name: []const u8,
    dtype: []const u8, // "F32" | "I32"
    shape: []const i64,
    bytes: []const u8,
};

pub const MetaKV = struct { k: []const u8, v: []const u8 };

/// xxh64 — seule famille de checksum du chantier (décision Task 0.4 : `xxhash` disponible côté
/// Python, donc PAS de repli crc32). Signature confirmée : `hash(seed: u64, input: anytype) u64`
/// (lib/std/hash/xxhash.zig:581). Seed 0 = celle de `xxhash.xxh64(bytes)` en Python.
pub fn xxh64(bytes: []const u8) u64 {
    return std.hash.XxHash64.hash(0, bytes);
}

pub const Entry = struct { dtype: []const u8, shape: []i64, off0: usize, off1: usize };

pub const Header = struct {
    arena: std.heap.ArenaAllocator,
    meta: std.StringHashMapUnmanaged([]const u8),
    entries: std.StringHashMapUnmanaged(Entry),
    data_base: usize, // 8 + header_len : base absolue des data_offsets

    pub fn deinit(self: *Header) void {
        self.arena.deinit();
    }

    pub fn metaGet(self: *const Header, k: []const u8) ?[]const u8 {
        return self.meta.get(k);
    }
};

pub const ReadError = error{ KvDumpBadFormat, KvDumpTruncated };

// --- Refus BRUYANT de la troncature (dette K7, 10 août 2026) -------------------------------
// Avant : les 7 sites retournaient `KvDumpTruncated` NUE. Les appelants qui `catch` nomment le
// fichier (`loadCacheManifest`), mais ceux qui `try` (lecture des 4 caches, gemma4_g12auto.zig
// ~2650) laissaient l'erreur remonter sans un mot : l'utilisateur voyait `error:
// KvDumpTruncated` et rien d'autre — contraire au standard « refus bruyant » du repo (spec §4.5,
// 11 refus DC5 tous nommés). Le chemin est passé en paramètre parce que ni `readHeader` ni
// `readTensorInto` ne le connaissaient : c'est le prix d'un message autonome.
// Contrainte : AUCUNE allocation (pas d'`allocPrint`) — ces chemins sont hors boucle de step,
// donc sans risque D10, mais le repo ne s'autorise pas d'allouer pour un message d'erreur.

/// Lecture COURTE : on sait combien on attendait et combien on a lu.
/// `detail` précise `what` (nom de tenseur) ou vaut "" — concaténation par le formateur.
fn failTruncated(path: []const u8, what: []const u8, detail: []const u8, want: u64, got: u64) ReadError {
    log.err("kvdump: fichier tronqué '{s}' — {s}{s} : attendu {d} octets, lu {d}", .{ path, what, detail, want, got });
    return ReadError.KvDumpTruncated;
}

/// Lecture qui ÉCHOUE au lieu d'être courte : le nombre d'octets lus est inconnu, l'erreur
/// sous-jacente est nommée (ne jamais la faire passer pour un décompte).
fn failTruncatedIo(path: []const u8, what: []const u8, detail: []const u8, want: u64, cause: anyerror) ReadError {
    log.err("kvdump: lecture impossible '{s}' — {s}{s} : {d} octets attendus, échec I/O ({s})", .{ path, what, detail, want, @errorName(cause) });
    return ReadError.KvDumpTruncated;
}

/// Écrit un safetensors : __metadata__ d'abord, puis les tenseurs dans l'ordre donné,
/// data_offsets contigus. String-building par allocPrint/appendSlice — jamais {any}, jamais
/// d'API writer non prouvée dans ce repo.
pub fn write(allocator: std.mem.Allocator, io: std.Io, path: []const u8, tensors: []const TensorOut, meta: []const MetaKV) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.appendSlice(allocator, "{\"__metadata__\":{");
    for (meta, 0..) |kv, i| {
        if (i != 0) try header.append(allocator, ',');
        const frag = try std.fmt.allocPrint(allocator, "\"{s}\":\"{s}\"", .{ kv.k, kv.v });
        defer allocator.free(frag);
        try header.appendSlice(allocator, frag);
    }
    try header.append(allocator, '}');
    var off: usize = 0;
    for (tensors) |t| {
        const head = try std.fmt.allocPrint(allocator, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype });
        defer allocator.free(head);
        try header.appendSlice(allocator, head);
        for (t.shape, 0..) |d, i| {
            if (i != 0) try header.append(allocator, ','); // fmt est comptime : pas de `if` runtime dans allocPrint
            const dim = try std.fmt.allocPrint(allocator, "{d}", .{d});
            defer allocator.free(dim);
            try header.appendSlice(allocator, dim);
        }
        const tail = try std.fmt.allocPrint(allocator, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + t.bytes.len });
        defer allocator.free(tail);
        try header.appendSlice(allocator, tail);
        off += t.bytes.len;
    }
    try header.append(allocator, '}');
    var len_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_le, header.items.len, .little);
    const f = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer f.close(io);
    try f.writePositionalAll(io, &len_le, 0);
    try f.writePositionalAll(io, header.items, 8);
    var pos: usize = 8 + header.items.len;
    for (tensors) |t| {
        try f.writePositionalAll(io, t.bytes, pos);
        pos += t.bytes.len;
    }
}

/// Lit et parse le header. Ne lit AUCUN tenseur. `file` reste ouvert, possédé par l'appelant.
/// Le `Header` retourné possède l'arena : toutes les slices qu'il expose (clés, strings de
/// manifest, shapes) y vivent et restent valides jusqu'à `deinit`.
pub fn readHeader(gpa: std.mem.Allocator, io: std.Io, file: std.Io.File, path: []const u8) !Header {
    var len_le: [8]u8 = undefined;
    const n0 = try file.readPositionalAll(io, &len_le, 0);
    if (n0 != 8) return failTruncated(path, "préfixe de longueur du header", "", 8, n0);
    const hlen = std.mem.readInt(u64, &len_le, .little);
    if (hlen == 0 or hlen > 1 << 20) return ReadError.KvDumpBadFormat;
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const hbuf = try a.alloc(u8, hlen);
    const n1 = file.readPositionalAll(io, hbuf, 8) catch |cause| return failTruncatedIo(path, "header JSON", "", hlen, cause);
    if (n1 != hlen) return failTruncated(path, "header JSON", "", hlen, n1);
    // ⚠ `parseFromSliceLeaky` — PAS `parseFromSlice` (le patron gencfg.zig:239, qui garde un
    // `Parsed`). Un `Parsed` retenu dans ce struct est un PIÈGE MORTEL ici : `Parsed.deinit()`
    // lit `self.arena.child_allocator`, or ce child_allocator est `arena.allocator()` — il
    // capture l'adresse de l'ArenaAllocator LOCAL à cette fonction. Le struct étant retourné
    // PAR VALEUR, ce pointeur devient pendouillant et le deinit segfaulte (mordu à l'exécution
    // du premier restore réel, Task 5). En Leaky, tout vit dans `a` et une seule arena possède
    // tout — `deinit` opère sur la copie, dont le child_allocator est le gpa.
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, hbuf, .{ .allocate = .alloc_always }) catch
        return ReadError.KvDumpBadFormat;
    if (parsed != .object) return ReadError.KvDumpBadFormat;
    var meta: std.StringHashMapUnmanaged([]const u8) = .empty;
    var entries: std.StringHashMapUnmanaged(Entry) = .empty;
    var it = parsed.object.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        if (std.mem.eql(u8, key, "__metadata__")) {
            if (e.value_ptr.* != .object) return ReadError.KvDumpBadFormat;
            var mit = e.value_ptr.object.iterator();
            while (mit.next()) |m| {
                if (m.value_ptr.* != .string) return ReadError.KvDumpBadFormat;
                try meta.put(a, m.key_ptr.*, m.value_ptr.string);
            }
        } else {
            const obj = e.value_ptr.*;
            if (obj != .object) return ReadError.KvDumpBadFormat;
            const dt = obj.object.get("dtype") orelse return ReadError.KvDumpBadFormat;
            const sh = obj.object.get("shape") orelse return ReadError.KvDumpBadFormat;
            const offs = obj.object.get("data_offsets") orelse return ReadError.KvDumpBadFormat;
            if (dt != .string or sh != .array or offs != .array or offs.array.items.len != 2) return ReadError.KvDumpBadFormat;
            const shape = try a.alloc(i64, sh.array.items.len);
            for (sh.array.items, 0..) |v, i| {
                if (v != .integer) return ReadError.KvDumpBadFormat;
                shape[i] = v.integer;
            }
            if (offs.array.items[0] != .integer or offs.array.items[1] != .integer) return ReadError.KvDumpBadFormat;
            if (offs.array.items[0].integer < 0 or offs.array.items[1].integer < offs.array.items[0].integer) return ReadError.KvDumpBadFormat;
            try entries.put(a, key, .{
                .dtype = dt.string,
                .shape = shape,
                .off0 = @intCast(offs.array.items[0].integer),
                .off1 = @intCast(offs.array.items[1].integer),
            });
        }
    }
    return .{ .arena = arena, .meta = meta, .entries = entries, .data_base = 8 + hlen };
}

/// Taille en octets déclarée par le header pour ce tenseur (data_offsets), sans lire les données.
pub fn entryBytes(h: *const Header, name: []const u8) !usize {
    const e = h.entries.get(name) orelse return error.KvDumpBadFormat;
    return e.off1 - e.off0;
}

/// Vérifie dtype ET shape d'une entrée contre l'attendu compilé. Refus bruyant (spec §4.5).
pub fn expectShape(h: *const Header, name: []const u8, dtype: []const u8, shape: []const i64) !void {
    const e = h.entries.get(name) orelse return error.KvDumpBadFormat;
    if (!std.mem.eql(u8, e.dtype, dtype)) return error.KvDumpShapeMismatch;
    if (e.shape.len != shape.len) return error.KvDumpShapeMismatch;
    for (e.shape, shape) |got, want| {
        if (got != want) return error.KvDumpShapeMismatch;
    }
}

/// Lit un tenseur ENTIER dans `dest` (taille exacte exigée), puis vérifie son checksum
/// contre `expected_xxh64` (ordre : lecture → hash → comparaison, spec §4.3).
pub fn readTensorInto(io: std.Io, file: std.Io.File, path: []const u8, h: *const Header, name: []const u8, dest: []u8, expected_xxh64: u64) !void {
    const e = h.entries.get(name) orelse return error.KvDumpBadFormat;
    if (e.off1 - e.off0 != dest.len) return error.KvDumpShapeMismatch;
    const n = file.readPositionalAll(io, dest, h.data_base + e.off0) catch |cause| return failTruncatedIo(path, "tenseur ", name, dest.len, cause);
    if (n != dest.len) return failTruncated(path, "tenseur ", name, dest.len, n);
    if (xxh64(dest) != expected_xxh64) return error.KvDumpChecksumMismatch;
}

/// Parse une valeur de manifest en u64 (les checksums sont écrits en hexadécimal, le reste en
/// décimal — la base est donc un paramètre, jamais devinée).
pub fn metaInt(h: *const Header, key: []const u8, base: u8) !u64 {
    const s = h.metaGet(key) orelse return error.KvDumpBadFormat;
    return std.fmt.parseInt(u64, s, base) catch error.KvDumpBadFormat;
}

pub const CkptFingerprint = struct { bytes: u64, hdr_xxh64: u64 };

/// Fingerprint d'un checkpoint safetensors : taille du fichier + xxh64 de (8 octets de
/// longueur + header JSON). Jamais les 24 Go de données. Spec §4.1 — le fingerprint est par
/// CONTENU : un même checkpoint atteint par un autre chemin/symlink reste valide.
pub fn ckptFingerprint(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !CkptFingerprint {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer f.close(io);
    var len_le: [8]u8 = undefined;
    const n0 = try f.readPositionalAll(io, &len_le, 0);
    if (n0 != 8) return failTruncated(path, "préfixe de longueur du checkpoint", "", 8, n0);
    const hlen = std.mem.readInt(u64, &len_le, .little);
    if (hlen == 0 or hlen > 1 << 30) return ReadError.KvDumpBadFormat;
    const buf = try gpa.alloc(u8, 8 + hlen);
    defer gpa.free(buf);
    @memcpy(buf[0..8], &len_le);
    const n1 = try f.readPositionalAll(io, buf[8..], 8);
    if (n1 != hlen) return failTruncated(path, "header du checkpoint", "", hlen, n1);
    const size = try f.length(io);
    return .{ .bytes = size, .hdr_xxh64 = xxh64(buf) };
}
