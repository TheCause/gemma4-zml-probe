# P5.2 — Gemma 4 E2B attention forward (5 sous-gates)

**Status global** : P5.2.A + P5.2.B PASS, P5.2.C.0 (oracle) PASS (28 mai 2026). C.1-C.3, D, E à venir.
**Scope** : implémentation ZML de l'attention forward consommant la policy table P5.1.

## Découpage (rappel)

| Sous-gate | Périmètre | État |
|---|---|---|
| **P5.2.A** | policy lookup ZML host-side (input `layer_idx` → routing) | **PASS** |
| **P5.2.B** | producer/read routing mock (slots factices, dispatch sans calcul) | **PASS** |
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

---

## P5.2.B — Producer/read routing mock (PASS, 28 mai 2026)

### Objectif
Valider que la policy table (oracle JSON, validée en P5.2.A) résout correctement vers des slots KV factices `producer_kv[15]`, **sans allouer le moindre cache réel ni faire aucun calcul Q/K/V**.

### Contrat
- **Entrée CLI** : chemin vers `yoco_policy_table.json` (oracle P5.1).
- **Mock interne** : `producer_kv[15]` = tableau pré-construit à l'init avec `.{ .producer_layer = i, .marker = 1000 + i }`. Aucune allocation cache réelle, juste un opaque traceur.
- **Sorties** :
  1. Affichage des 15 slots mock.
  2. Tableau des 9 cas fixes (0, 4, 13, 14, 15, 18, 19, 24, 34) avec leur slot résolu + mode (`producer_self` / `reader_shared`).
  3. Validation 35 entrées avec compteurs : `producer_self_routes`, `sliding_reader_routes_to_13`, `full_reader_routes_to_14`, writers stables.
- **Interdits stricts** : Q/K/V projection, RoPE, attention matmul/scores/softmax, sliding mask, cache réel.

### Livrables
| Fichier | Rôle |
|---|---|
| `zml_runner/gemma4_routing_mock.zig` | runner pur Zig, lit JSON, route via `producer_kv[entry.target_kv_layer]` |
| `zml_runner/BUILD.bazel` | nouveau `zig_binary` cible `gemma4_routing_mock` |
| `logs/P5_2_B_routing_mock.log` | sortie `bazel run` complète, ~57 lignes |
| `docs/P5_2_attention_forward.md` | ce document (section P5.2.B) |

### Sortie observée
```
info: Fixed-case routing:
info:   layer | type              | is_reader | target | slot.producer_layer | mode            | label
info:   ------|-------------------|-----------|--------|---------------------|-----------------|------
info:       0 | sliding_attention | false     |      0 |                   0 | producer_self   | producer sliding -> slot 0
info:       4 | full_attention    | false     |      4 |                   4 | producer_self   | producer full -> slot 4
info:      13 | sliding_attention | false     |     13 |                  13 | producer_self   | producer sliding writer -> slot 13
info:      14 | full_attention    | false     |     14 |                  14 | producer_self   | producer full writer -> slot 14
info:      15 | sliding_attention | true      |     13 |                  13 | reader_shared   | reader sliding -> slot 13
info:      18 | sliding_attention | true      |     13 |                  13 | reader_shared   | reader sliding -> slot 13
info:      19 | full_attention    | true      |     14 |                  14 | reader_shared   | reader full -> slot 14
info:      24 | full_attention    | true      |     14 |                  14 | reader_shared   | reader full -> slot 14
info:      34 | full_attention    | true      |     14 |                  14 | reader_shared   | reader full -> slot 14
info:
info: Full routing validation (35 entries):
info:   producer_self_routes        : 15/15
info:   sliding_reader_routes_to_13 : 16/16
info:   full_reader_routes_to_14    : 4/4
info:   writers stable               : sliding=13 (expect 13), full=14 (expect 14)
info:
info: P5.2.B PASS: producer/read routing mock validated end-to-end
```

### Décision design
Pas de partage de module Zig avec `gemma4_policy_lookup.zig` — sous-gates **indépendantes par design**. P5.2.B ré-implémente sa propre lecture JSON et son propre routage. Une consolidation éventuelle (factoriser une lib `policy.zig` partagée) sera arbitrée en P5.2.C+ s'il y a vraiment duplication coûteuse.

### Critères de clôture P5.2.B
- [x] Build bazel PASS sur 3090
- [x] Run produit sortie attendue
- [x] 9/9 cas fixes PASS
- [x] 15/15 producer self-routes, 16/16 sliding readers → 13, 4/4 full readers → 14
- [x] Writers stables : sliding=13, full=14
- [x] Aucune dep `//zml`, aucun calcul Q/K/V, aucun cache réel
- [x] Log archivé `logs/P5_2_B_routing_mock.log`

**Tag** : `p5.2-b-routing-mock-pass`

---

---

## P5.2.C — Q-only reader path (en cours, 4 sous-sous-gates)

### Découpage
| Sous-sous-gate | Périmètre | État |
|---|---|---|
| **P5.2.C.0** | PyTorch oracle Q-only reader, layer 15 sliding, no K/V | **PASS** |
| P5.2.C.1 | ZML q_proj | pending |
| P5.2.C.2 | ZML q_norm | pending |
| P5.2.C.3 | ZML RoPE Q-only | pending |

---

## P5.2.C.0 — PyTorch oracle Q-only reader layer 15 sliding (PASS, 28 mai 2026)

### Objectif
Premier calcul réel après 5 gates de cartographie/routing pure : produire un oracle PyTorch fp32 du chemin Q (q_proj → q_norm → RoPE → transpose) pour la layer 15 (premier reader sliding). Sérialiser comme fixture safetensors pour valider P5.2.C.1/2/3 ZML byte-par-byte ensuite.

### Périmètre strict
- **Cible** : layer 15 (sliding reader, `is_reader=True`, `first_kv_shared_layer_idx=15`).
- **Input** : synthétique déterministe `torch.manual_seed(1337)`, shape `[B=1, S=4, H=1536]`.
- **Poids chargés** : `q_proj.weight [2048, 1536]` et `q_norm.weight [256]` depuis `weights/model.safetensors` (pas de chargement du modèle complet — économie RAM).
- **Modules instanciés** :
  - `torch.nn.Linear(1536, 2048, bias=False)` pour q_proj
  - `Gemma4RMSNorm(256, eps=1e-6)` pour q_norm
  - `Gemma4TextRotaryEmbedding(text_config)` pour rotary (pas de poids, pure compute)
- **Interdits stricts** : k_proj, v_proj, k_norm, v_norm, attention scores, matmul QK, softmax, cache, sliding mask.

### Pipeline (miroir verbatim `Gemma4TextAttention.forward` L811-L824)
```
A) q_after_proj  = q_proj(hidden_input)                                         [B,S,n_heads*head_dim] = [1,4,2048]
B) q_view        = q_after_proj.view(B,S,n_heads,head_dim)
   q_after_norm  = q_norm(q_view)                                                [B,S,n_heads,head_dim] = [1,4,8,256]
C) (cos, sin)    = rotary(hidden_input, position_ids=arange(S), layer_type="sliding_attention")
   q_after_rope  = apply_rotary_pos_emb(q_after_norm, cos, sin, unsqueeze_dim=2) [1,4,8,256]
D) q_final       = q_after_rope.transpose(1,2).contiguous()                      [1,8,4,256]
```

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/14_q_only_reader_oracle.py` | générateur oracle (Python, depuis raw safetensors) |
| `fixtures/q_only_reader_layer15.safetensors` | 9 tenseurs (input + 2 poids + 2 rotary + 4 intermédiaires Q) ~12.7 MB |
| `fixtures/q_only_reader_layer15_manifest.json` | shapes/dtypes + spec refs + pipeline + interdits |
| `logs/14_q_only_reader_oracle.log` | sortie + sanity stats |

### Sanity stats (sortie observée)
```
q_after_proj   mean=-7.286e-03  std= 1.296e+00  min=-7.93  max= 9.62
q_after_norm   mean=-5.657e-03  std= 9.922e-01  min=-5.69  max= 6.65
q_after_rope   mean=-1.683e-02  std= 9.921e-01  min=-5.79  max= 6.65
q_final        mean=-1.683e-02  std= 9.921e-01  min=-5.79  max= 6.65
```

**Sanity RoPE** : à position 0, RoPE est identité (cos=1, sin=0) → `q_after_norm[0,0,...] == q_after_rope[0,0,...]`. À position 3 (last), RoPE rotate activement : `cos[0,3,0:4] = [-0.99, -0.94, -0.86, -0.75]`, `sin[0,3,0:4] = [0.14, 0.34, 0.52, 0.66]`, delta_max entre norm et rope = **6.97** → RoPE bien actif.

### Notes capitalisées
- **Prefixe checkpoint multi-modal** : Gemma 4 E2B est multi-modal (`vision_tower` + `audio_tower` + `language_model`). Les poids `language_model` sont préfixés `model.language_model.layers.X.self_attn...` — **pas** `model.layers.X.self_attn...` (qui n'existe pas). À retenir pour tous les futurs scripts pure-safetensors.
- **Poids K/V présents sur disque pour layer 15** (cf P5.0 § 4 nota bene). Le runtime Python Transformers les ignore via `_keys_to_ignore_on_load_unexpected` — mais ils sont accessibles via `safe_open` direct. Utile si oracle inclut K/V plus tard.
- **`safe_open`** s'importe depuis `safetensors` (pas `safetensors.torch`) — Pyright le warn correctement.
- **Pas de chargement du modèle complet** : économie RAM massive (8GB vs ~50MB), reproductible, isole le pipeline Q.

### Critères de clôture P5.2.C.0
- [x] Script lit `model.language_model.layers.15.self_attn.{q_proj,q_norm}.weight` depuis safetensors
- [x] Pipeline complet exécuté sans erreur (q_proj → q_norm → RoPE → transpose)
- [x] Sanity RoPE : identité à position 0, rotation active à position 3
- [x] Shapes attendues : `q_after_proj [1,4,2048]`, `q_after_norm [1,4,8,256]`, `q_after_rope [1,4,8,256]`, `q_final [1,8,4,256]`
- [x] Fixture safetensors écrit (~12.7 MB)
- [x] Manifest JSON écrit avec spec_refs + pipeline + interdits

**Tag** : `p5.2-c0-pytorch-oracle-pass`

---

## P5.2.C.1 → C.3, P5.2.D → P5.2.E (à venir)

Voir mémoire `project_gemma4_zml_probe.md` pour la roadmap détaillée. Une sous-sous-gate à la fois, garde-fou strict. P5.2.C.1 = portage ZML de `q_proj` avec comparaison byte-équivalente vs `q_after_proj` du fixture C.0.
