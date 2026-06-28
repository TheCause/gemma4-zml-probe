// Helper mémoire — lecture RSS / swap depuis /proc/self/status (Linux), no-op ailleurs.
//
// But : instrumenter le pic mémoire post-compile (go/no-go du chunking, cf
// docs/GENERATION_LONGUE_CHUNKING_DESIGN.md §6) et caractériser la fuite résiduelle (swap 2.8→4.1 Go
// observée, ENGINE_LOG 7 juin). Sur la 3090 (Linux) : lit VmRSS/VmSwap en Ko. Sur un build non-Linux
// (ex: Mac local pour syntaxe) : renvoie null (les log* sont comptime-morts → pas d'output parasite).
//
// Unité : kilo-octets (Ko) — cohérent avec /proc (kB = page-size-agnostic, base 1024).

const std = @import("std");
const builtin = @import("builtin");
const log = std.log;

/// RSS du process en Ko, ou null si /proc indisponible (non-Linux / lecture échouée).
pub fn rssKb() ?u64 {
    if (builtin.os.tag != .linux) return null;
    return readField("VmRSS:");
}

/// Swap du process en Ko, ou null.
pub fn swapKb() ?u64 {
    if (builtin.os.tag != .linux) return null;
    return readField("VmSwap:");
}

fn readField(prefix: []const u8) ?u64 {
    var file = std.fs.openFileAbsolute("/proc/self/status", .{}) catch return null;
    defer file.close();
    var buf: [4096]u8 = undefined;
    const n = file.read(&buf) catch return null;
    const content = buf[0..n];
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
pub fn logMem(tag: []const u8) void {
    if (builtin.os.tag != .linux) return;
    const rss = rssKb() orelse return;
    const rss_gib: u64 = rss / 1048576;
    const swp = swapKb();
    if (swp) |s| {
        const s_gib: u64 = s / 1048576;
        log.info("[mem] {s}: RSS={d} KiB (~{d} GiB) swap={d} KiB (~{d} GiB)", .{ tag, rss, rss_gib, s, s_gib });
    } else {
        log.info("[mem] {s}: RSS={d} KiB (~{d} GiB) swap=n/a", .{ tag, rss, rss_gib });
    }
}
