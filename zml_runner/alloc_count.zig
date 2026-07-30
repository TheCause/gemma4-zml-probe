//! Compteur d'allocations — wrapper transparent de std.mem.Allocator.
//! Les 4 fonctions vtable délèguent au parent et comptent. Les corps suivent
//! std.testing.FailingAllocator de CE toolchain (0.16.0-dev.2722), signatures copiées.
//! ⚠ Compteurs u64 NON atomiques : corrects parce que seul le thread principal passe par ce
//! wrapper (std.Io.Threaded est construit sur le gpa non wrappé AVANT run() — dette DA-6).
const std = @import("std");

pub const CountingAllocator = struct {
    parent: std.mem.Allocator,
    n_alloc: u64 = 0,
    n_resize: u64 = 0,
    n_remap: u64 = 0,
    n_free: u64 = 0,
    bytes_alloc: u64 = 0,

    pub fn init(parent: std.mem.Allocator) CountingAllocator {
        return .{ .parent = parent };
    }

    pub fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    /// Somme des compteurs d'ACQUISITION (alloc + resize + remap) : c'est le delta
    /// qu'observent les gates. `free` est publié mais hors somme (libérer n'acquiert pas).
    pub fn calls(self: *const CountingAllocator) u64 {
        return self.n_alloc + self.n_resize + self.n_remap;
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.n_alloc += 1;
        self.bytes_alloc += len;
        return self.parent.rawAlloc(len, alignment, ret_addr);
    }
    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.n_resize += 1;
        return self.parent.rawResize(memory, alignment, new_len, ret_addr);
    }
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.n_remap += 1;
        return self.parent.rawRemap(memory, alignment, new_len, ret_addr);
    }
    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.n_free += 1;
        return self.parent.rawFree(memory, alignment, ret_addr);
    }
};
