// Brique TurboQuant V-only — greffée au point `post_v_norm` du socle `engine.zig`.
//
// Reproduit, comme BRIQUE injectée (pas comme copie du moteur), l'insert V-quant du POC gemma4_gen_vq.zig
// (lignes 265-270) : V (post-v_norm, [.b,.s,.nh,.hd], decode 1-step → b=s=nh=1) est quantifié via la chaîne
// MSE V-only `engine.quantizeV` (prouvée en Q3), avec le couple {codebook,Hadamard} sélectionné par
// `is_full` (256 pour sliding / 512 pour full). `is_full` est COMPTIME (les deux Hadamard ont des shapes
// différentes 256 vs 512 → la sélection doit être résolue à la compilation).
//
// Contrat de chargement (cf engine DESIGN §3.4) : les constantes sont des champs `Tensor` créés via
// `init(View)` → le loader les résout par `id` (bind dans la View de la fixture). Les poids du moteur
// vivant dans un store distinct (checkpoint), le main charge poids et brique en deux passes puis assemble.

const zml = @import("zml");
const engine = @import("engine.zig");

pub const TurboQuantVBrick = struct {
    codebook_256: zml.Tensor, // {c=16}
    hadamard_256: zml.Tensor, // {e=256, hd=256}
    codebook_512: zml.Tensor, // {c=16}
    hadamard_512: zml.Tensor, // {e=512, hd=512}

    /// Crée les constantes depuis la View de la fixture (mêmes clés/tags que le POC `Packed`, gen_vq:195-198).
    pub fn init(view: zml.io.TensorStore.View) TurboQuantVBrick {
        return .{
            .codebook_256 = view.createTensor("codebook_256", .{.c}, null),
            .hadamard_256 = view.createTensor("hadamard_256", .{ .e, .hd }, null),
            .codebook_512 = view.createTensor("codebook_512", .{.c}, null),
            .hadamard_512 = view.createTensor("hadamard_512", .{ .e, .hd }, null),
        };
    }

    /// Point d'extension : V post-v_norm → V quantifié (même forme). `is_full` comptime sélectionne la shape.
    pub fn post_v_norm(self: TurboQuantVBrick, v: zml.Tensor, comptime is_full: bool, ctx: engine.LayerCtx) zml.Tensor {
        _ = ctx; // non utilisé par TurboQuant (route sur is_full) ; dispo pour briques futures
        const cb = if (is_full) self.codebook_512 else self.codebook_256;
        const Pi = if (is_full) self.hadamard_512 else self.hadamard_256;
        // wrapper d'axes autour de quantizeV ([.k,.hd]) — repris verbatim du POC (gen_vq:268-270).
        const hd = v.dim(.hd);
        const v2 = v.reshape(.{ 1, hd }).withTags(.{ .k, .hd }); // decode : b=s=nh=1
        const o = engine.quantizeV(v2, cb, Pi); // [.k=1, .hd]
        return o.reshape(.{ 1, 1, 1, hd }).withTags(.{ .b, .s, .nh, .hd });
    }
};
