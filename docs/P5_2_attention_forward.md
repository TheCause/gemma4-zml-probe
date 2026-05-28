# P5.2 — Gemma 4 E2B attention forward (5 sous-gates)

**Status global** : P5.2.A PASS (28 mai 2026). B-E à venir.
**Scope** : implémentation ZML de l'attention forward consommant la policy table P5.1.

## Découpage (rappel)

| Sous-gate | Périmètre | État |
|---|---|---|
| **P5.2.A** | policy lookup ZML host-side (input `layer_idx` → routing) | **PASS** |
| P5.2.B | producer/read routing mock (dispatch sans calcul) | pending |
| P5.2.C | Q-only reader path (q_proj + q_norm + RoPE sur Q) | pending |
| P5.2.D | K/V producer path (qkv_proj + k_norm + v_norm + RoPE + write slot) | pending |
| P5.2.E | sliding mask au compute (pas troncation cache) | pending |

Hors scope P5.2 : fast prefill (option vLLM, P5.3+).

---

## P5.2.A — Policy lookup ZML host-side (PASS, 28 mai 2026)

### Objectif
Prouver que le runtime Zig consomme la policy table YOCO (`fixtures/yoco_policy_table.json` produite en P5.1) et **redérive la même table de façon indépendante**, sans toucher à l'attention.

### Contrat
- **Entrée CLI** : chemin vers `yoco_policy_table.json`.
- **Sorties** :
  1. Tableau pretty-print des 7 cas fixes (0, 4, 13, 14, 15, 19, 34).
  2. Validation Zig recompute vs JSON oracle sur les 35 entrées.
  3. Invariants sanity : 15 producers / 20 readers, writer full = 14, writer sliding = 13.
- **Interdits stricts** : QKV, RoPE, matmul attention, cache réel, dépendance `//zml`.

### Livrables
| Fichier | Rôle |
|---|---|
| `zml_runner/gemma4_policy_lookup.zig` | runner pur Zig (std uniquement), miroir verbatim de `compute_policy_table` (Python P5.1) |
| `zml_runner/BUILD.bazel` | nouveau `zig_binary` cible, sans deps `//zml` ni `//bazel` |
| `logs/P5_2_A_policy_lookup.log` | sortie `bazel run` complète, ~42 lignes |
| `docs/P5_2_attention_forward.md` | ce document |

### Implémentation
Logique miroir verbatim (cf. `compute_policy_table` Python et `Gemma4TextAttention.__init__` Transformers L777-L782 / `Gemma4Attention.__init__` vLLM L469-L489) :

```zig
const Policy = struct {
    fn build(allocator, n, k, layer_types) !Policy {
        const first = n - k;
        for (layer_types, 0..) |t, i_usize| {
            const i: u32 = @intCast(i_usize);
            const is_reader = (k > 0) and (i >= first);
            var target: u32 = i;
            if (is_reader) {
                var j: u32 = first;
                while (j > 0) { j -= 1; if (layer_types[j] == t) { target = j; break; } }
            }
            entries[i_usize] = .{ .layer_idx = i, .layer_type = t,
                                  .is_reader = is_reader, .target_kv_layer = target };
        }
    }

    fn lookup(self: Policy, layer_idx: u32) PolicyEntry {
        return self.entries[layer_idx];
    }
};
```

### Validation
- **Parse JSON** : `std.json.parseFromSlice(std.json.Value, ...)` puis walk dynamique.
- **Recompute Zig** : `Policy.build()` à partir de `(num_hidden_layers, num_kv_shared_layers, layer_types extraits du JSON)`.
- **Comparaison entry-by-entry** : 35/35 PASS (layer_idx, layer_type, is_reader, target_kv_layer).
- **Invariants** : 15 producers, 20 readers, writer full = 14, writer sliding = 13. Tous OK.

### Sortie observée
```
info:   layer_idx | layer_type        | is_reader | target_kv_layer | label
info:   ----------|-------------------|-----------|-----------------|------
info:           0 | sliding_attention | false     |               0 | sliding producer (own)
info:           4 | full_attention    | false     |               4 | full producer (own)
info:          13 | sliding_attention | false     |              13 | sliding writer
info:          14 | full_attention    | false     |              14 | full writer
info:          15 | sliding_attention | true      |              13 | sliding reader -> target 13
info:          19 | full_attention    | true      |              14 | full reader -> target 14
info:          34 | full_attention    | true      |              14 | last full reader -> target 14
info:
info:   35/35 entries match (Zig recompute == JSON oracle)
info: P5.2.A PASS: policy lookup ZML host-side validated end-to-end
```

### Notes Zig 0.16-dev capitalisées
Le runner PLE de P4.4.2 est compilé sous Zig 0.16-dev avec rules_zig. Trois APIs ont migré depuis 0.14 :
1. **Main** : `pub fn main(init: std.process.Init) !void` (pas `pub fn main() !void`). `init` fournit `arena`, `gpa`, `io`, `minimal.args`.
2. **Args** : `init.minimal.args.toSlice(arena.allocator())` au lieu de `std.process.argsAlloc(...)`. Args[0] = binary name.
3. **Filesystem** : `std.fs.cwd()` retiré. Utiliser `std.Io.Dir.cwd()` + `dir.openFile(io, path, .{})` + `file.length(io)` + `file.readPositionalAll(io, buf, 0)`. Fermeture explicite avec `file.close(io)`.

À réutiliser dans tous les runners pure-Zig du projet.

### Reproductibilité
```bash
# Depuis M1, deploy + build + run :
cd ~/dev/gemma4-zml-probe/zml_runner && ./deploy_to_3090.sh

ssh user@gpu-host 'export PATH="$HOME:$PATH" && cd /data/rqz_workspace/zml && \
  bazelisk run //examples/rqz:gemma4_policy_lookup -- \
    /data/gemma4-zml-probe/fixtures/yoco_policy_table.json'
```

### Critères de clôture P5.2.A
- [x] Build bazel PASS sur 3090
- [x] Run produit sortie attendue
- [x] 35/35 entries match Zig recompute vs JSON oracle
- [x] 4 invariants sanity PASS
- [x] Aucune dep `//zml`, aucun calcul attention
- [x] Log archivé `logs/P5_2_A_policy_lookup.log`

**Tag suggéré** : `p5.2-a-policy-lookup-pass`

---

## P5.2.B → P5.2.E (à venir)

Voir mémoire `project_gemma4_zml_probe.md` pour la roadmap détaillée. Une sous-gate à la fois, garde-fou strict.
