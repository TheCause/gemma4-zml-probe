// Helper mémoire — lecture RSS / swap depuis /proc/self/status (Linux), no-op ailleurs.
//
// But : instrumenter le pic mémoire post-compile (go/no-go du chunking, cf
// docs/GENERATION_LONGUE_CHUNKING_DESIGN.md §6) et caractériser la fuite résiduelle (swap 2.8→4.1 Go
// observée, ENGINE_LOG 7 juin). Sur la 3090 (Linux) : lit VmRSS/VmSwap en Ko. Sur un build non-Linux
// (ex: Mac local pour syntaxe) : renvoie null (les log* sont comptime-morts → pas d'output parasite).
//
// API FICHIER — Zig 0.16-dev (post-Writergate) : l'IO de haut niveau est THREADÉE (std.Io). On copie le
// pattern PROUVÉ de lecture d'un PSEUDO-fichier (taille déclarée 0, comme /proc) tel qu'utilisé dans zml :
//   - bin/zml-smi/utils/sysfs.zig : std.Io.Dir.cwd().openFile(io,..) + file.reader(io,&buf) + reader.interface
//   - zml/io/vfs/*.zig            : reader.readSliceShort(..) (lecture COURTE = peut retourner < demandé)
// On NE peut PAS faire file.length()+readPositionalAll (pattern .safetensors de gemma4_policy_lookup) car
// /proc/self/status rapporte une taille de 0 → on lit en court (readSliceShort) dans un buffer fixe.
//
// NB validation : sur Mac la branche Linux est comptime-morte (lazy analysis) → un build local ne prouve
// PAS que readField compile. Chaque appel est tracé à un usage RÉEL du checkout zml ; le seul point à
// confirmer au smoke 3090 est `reader.interface.readSliceShort` (composition File.Reader→std.Io.Reader).
//
// Unité : kilo-octets (Ko) — cohérent avec /proc (kB = page-size-agnostic, base 1024).

const std = @import("std");
const builtin = @import("builtin");
const log = std.log;

/// RSS du process en Ko, ou null si /proc indisponible (non-Linux / lecture échouée).
pub fn rssKb(io: std.Io) ?u64 {
    if (builtin.os.tag != .linux) return null;
    return readField(io, "VmRSS:");
}

/// Swap du process en Ko, ou null.
pub fn swapKb(io: std.Io) ?u64 {
    if (builtin.os.tag != .linux) return null;
    return readField(io, "VmSwap:");
}

fn readField(io: std.Io, prefix: []const u8) ?u64 {
    var file = std.Io.Dir.cwd().openFile(io, "/proc/self/status", .{ .mode = .read_only }) catch return null;
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    // /proc/self/status fait ~1,3 Ko (< une page) ; VmRSS/VmSwap sont dans les premiers ~1 Ko → une
    // lecture courte suffit (readSliceShort = lecture partielle tolérée pour un pseudo-fichier).
    var content_buf: [8192]u8 = undefined;
    const n = reader.interface.readSliceShort(&content_buf) catch return null;
    const content = content_buf[0..n];
    var it = std.mem.splitScalar(u8, content, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, prefix)) {
            // "VmRSS:    12345 kB"
            const rest = line[prefix.len..];
            const trimmed = std.mem.trim(u8, rest, " \t");
            var num_end: usize = 0;
            for (trimmed) |ch| {
                if (ch < '0' or ch > '9') break;
                num_end += 1;
            }
            if (num_end == 0) return null;
            return std.fmt.parseInt(u64, trimmed[0..num_end], 10) catch null;
        }
    }
    return null;
}

/// Log formaté RSS+swap avec un tag. No-op total hors-Linux (branches comptime-mortes).
/// GiB arrondi à l'entier (u64, spec `{d}`) — évite tout spec de précision float non validable hors-3090.
pub fn logMem(io: std.Io, tag: []const u8) void {
    if (builtin.os.tag != .linux) return;
    const rss = rssKb(io) orelse return;
    const rss_gib: u64 = rss / 1048576;
    const swp = swapKb(io);
    if (swp) |s| {
        const s_gib: u64 = s / 1048576;
        log.info("[mem] {s}: RSS={d} KiB (~{d} GiB) swap={d} KiB (~{d} GiB)", .{ tag, rss, rss_gib, s, s_gib });
    } else {
        log.info("[mem] {s}: RSS={d} KiB (~{d} GiB) swap=n/a", .{ tag, rss, rss_gib });
    }
}
