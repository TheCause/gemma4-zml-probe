# P5.2 — Gemma 4 E2B attention forward (5 sous-gates)

**Status global** : P5.2.A + P5.2.B PASS, P5.2.C COMPLET (C.0 oracle + C.1 q_proj + C.2 q_norm + C.3 RoPE) PASS (28 mai 2026), P5.2.D EN COURS (D.0 oracle + D.1 k_proj PASS, 28 mai 2026 soir). E à venir.
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
| **P5.2.C.1** | ZML q_proj (single dot, reduce .h) | **PASS** |
| **P5.2.C.2** | ZML q_norm (reshape + rmsNorm + mul) | **PASS** |
| **P5.2.C.3** | ZML RoPE Q-only (helper natif `zml.nn.rope`) | **PASS** |

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

---

## P5.2.C.1 — ZML q_proj reader layer 15 (PASS, 28 mai 2026)

### Objectif
Porter la projection linéaire `q_proj` en ZML pour layer 15 (sliding reader), exécuter le forward sur `hidden_input` du fixture C.0, comparer le résultat byte-équivalent contre l'oracle `q_after_proj`.

### Périmètre strict
- **Pipeline ZML** : un seul `dot` réduisant `.h` :
  ```zig
  q_after_proj_zml = hidden_input.dot(q_proj_weight, .h)
  // [.b=1, .s=4, .h=1536] dot [.o=2048, .h=1536] -> [.b=1, .s=4, .o=2048]
  ```
- **3 tenseurs chargés** depuis `fixtures/q_only_reader_layer15.safetensors` : `hidden_input`, `q_proj_weight`, `q_after_proj` (oracle). Les 6 autres tenseurs du fixture C.0 (q_norm_weight, rotary_cos/sin, q_after_norm, q_after_rope, q_final) sont **ignorés en C.1**.
- **Interdits stricts** : q_norm, reshape `[B,S,n_heads,head_dim]`, RoPE, transpose, K/V projection, attention scores, matmul QK, softmax, cache, sliding mask.

### Livrables
| Fichier | Rôle |
|---|---|
| `zml_runner/gemma4_q_proj.zig` | runner ZML (pattern P4.4.2 PLE runner) |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_q_proj` avec deps `//bazel` + `//zml` |
| `logs/P5_2_C1_q_proj.log` | sortie `bazel run` ~65 lignes |
| `docs/P5_2_attention_forward.md` | ce document (section P5.2.C.1) |

### Validation
**3 blocks fixed-point** (extraits oracle PyTorch, hardcoded dans le runner) :
- Block A `[0,0,:8]` flat_offset=0 — max_diff=4.53e-6
- Block B `[0,1,:8]` flat_offset=2048 — max_diff=5.13e-6
- Block C `[0,3,:8]` flat_offset=6144 — max_diff=2.38e-6
- **3 blocks max_diff = 5.13e-6**

**Scan global 8192 valeurs** :
- max_abs = **1.144e-5** à `flat_index=1926 (s=0, o=1926)`
- mean_abs = **6.34e-7**

**Tolerance** : 1e-4 → marge ~9× sous le seuil. **Résidu attendu** ~1.5e-5 (matmul PJRT-CPU Eigen-like vs PyTorch BLAS, cohérent P4.4.2 Gate E/J). Observé 1.14e-5 → conforme.

### Critères de clôture P5.2.C.1
- [x] Build bazel PASS (deps `//bazel` + `//zml`)
- [x] Run produit `q_after_proj_zml [1,4,2048]`
- [x] 3/3 fixed-point blocks PASS (max_diff < 1e-5)
- [x] Scan global 8192 valeurs : max_abs 1.14e-5, mean_abs 6.34e-7 (sous tolerance 1e-4)
- [x] Aucun q_norm, aucun RoPE, aucun transpose, aucun K/V, aucune attention
- [x] Log archivé `logs/P5_2_C1_q_proj.log`

**Tag** : `p5.2-c1-zml-q-proj-pass`

---

---

## P5.2.C.2 — ZML q_norm reader layer 15 (PASS, 28 mai 2026)

### Objectif
Étendre P5.2.C.1 (q_proj seul PASS) avec le pipeline q_norm pattern Llama (`normalized.mul(weight)`), comparer contre l'oracle `q_after_norm` du fixture C.0.

### Périmètre strict
- **Pipeline ZML** :
  ```zig
  q_after_proj  = hidden_input.dot(q_proj_weight, .h)            // reuse C.1
  q_4d          = q_after_proj.reshape({1,4,8,256})              // perd les tags
                    .withTags(.{.b, .s, .n, .d})                  // re-tag (piège #1)
  q_normalized  = zml.nn.rmsNorm(q_4d, .d, 1e-6)
  q_after_norm  = q_normalized.mul(q_norm_weight.broad(q_normalized.shape()))
  ```
- **4 tenseurs chargés** : `hidden_input`, `q_proj_weight`, `q_norm_weight`, `q_after_norm` (oracle).
- **Interdits stricts** : RoPE, transpose, K/V projection, attention scores, softmax, cache, sliding mask.

### Livrables
| Fichier | Rôle |
|---|---|
| `zml_runner/gemma4_q_norm.zig` | runner ZML (4 tenseurs du fixture C.0, étend C.1) |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_q_norm` |
| `logs/P5_2_C2_q_norm.log` | sortie `bazel run` ~76 lignes |
| `docs/P5_2_attention_forward.md` | ce document (section C.2) |

### Validation
**4 blocks fixed-point** (extraits oracle, hardcoded) :
- Block A `[0,0,0,:8]` flat_offset=0    — max_diff=3.81e-6
- Block B `[0,0,7,:8]` flat_offset=1792 — max_diff=1.19e-6
- Block C `[0,3,0,:8]` flat_offset=6144 — max_diff=1.43e-6
- Block D `[0,3,7,:8]` flat_offset=7936 — max_diff=4.77e-6
- **4 blocks max_diff = 4.77e-6**

**Scan global 8192 valeurs** :
- max_abs = **6.676e-6** à `flat_index=1926 (s=0, n=7, d=134)`
- mean_abs = **4.91e-7**

**Tolerance** : 1e-4 → marge ~15× sous le seuil. **Observation** : max_abs C.2 (6.68e-6) **plus petit** que max_abs C.1 (1.14e-5). La RMSNorm divise par `sqrt(mean(x²)+eps)` ≈ 1.7, ce qui réduit le résidu propagé d'environ le même facteur.

**Sanity layout** : position du max C.2 (s=0, n=7, d=134) = même position absolue que C.1 (flat_index 1926, qui était s=0, o=1926 = 7×256 + 134). Cohérence parfaite du reshape `[B,S,O] → [B,S,N,D]`.

### Notes capitalisées
- **Piège #1 (reshape sans tags)** : `reshape({...})` perd les tags ZML — **toujours** suivre par `.withTags(.{.b, .s, .n, .d})` avant op tag-based. Vérifié encore une fois en C.2.
- **Piège #2 (mul broadcast implicite)** : `q_norm_weight [d=256]` × `q_normalized [b,s,n,d]` nécessite `.broad(target.shape())` explicite. Pas de broadcast NumPy-like auto.
- **Piège #3 (pattern Llama vs Qwen)** : Gemma 4 utilise `normalized.mul(weight)` (Llama), pas `(1+weight)` (Qwen). Cf P4.4.2 Gate H.

### Critères de clôture P5.2.C.2
- [x] Build bazel PASS
- [x] Run produit `q_after_norm_zml [1,4,8,256]`
- [x] 4/4 fixed-point blocks PASS (max_diff < 5e-6)
- [x] Scan global 8192 valeurs : max_abs 6.68e-6 (sous tolerance 1e-4)
- [x] Aucun RoPE, aucun transpose, aucun K/V, aucune attention
- [x] Log archivé `logs/P5_2_C2_q_norm.log`

**Tag** : `p5.2-c2-zml-q-norm-pass`

---

---

## P5.2.C.3 — ZML RoPE Q-only reader layer 15 (PASS, 28 mai 2026)

### Objectif
Étendre P5.2.C.2 avec la rotation positionnelle RoPE sur Q seulement. Comparer contre l'oracle `q_after_rope` du fixture C.0. **Ferme P5.2.C complet** : chemin Q-only validé end-to-end pour une layer reader sliding.

### Décision design (inspection préalable du pattern ZML)
Avant codage, inspection de `zml/nn.zig` et `examples/llm/models/llama/model.zig` :
- **Helper natif** : `zml.nn.rope(x, pos_idx, opts)` (L270 de nn.zig). Pas besoin d'implémentation manuelle.
- **Conventions** :
  - x doit avoir tags `.s` et `.hd` (head_dim even).
  - `pos_idx` optionnel ; `null` → default `arange(0, x.dim(.s))` tag `.s`.
  - Layout : `.sequential` (HF) ou `.interleaved` (GGUF). Gemma 4 = HF style.
  - Scaling : union avec `.default { rope_theta }`, `.llama3`, `.yarn`, `.linear`. Gemma 4 sliding = `.default { 10_000 }`.
- **Math equivalence** : ZML `y_real = x_real*cos - x_imag*sin ; y_imag = x_real*sin + x_imag*cos` ≡ HF `q*cos + rotate_half(q)*sin` (avec cos/sin dupliqués). Preuve algébrique : pour split-half, les deux formules donnent exactement les mêmes coordonnées.
- **inv_freq** : ZML `theta^(-n/N_half)` ≡ PyTorch `1/base^(2n/head_dim)`. Identité mathématique.
- **Décision** : utiliser `zml.nn.rope` directement, **pas** consommer `rotary_cos`/`rotary_sin` du fixture. Si bytes-équivalents, on prouve aussi que les `inv_freq` ZML et PyTorch convergent en fp32.

### Périmètre strict
- **Pipeline ZML** :
  ```zig
  q_after_proj  = hidden_input.dot(q_proj_weight, .h)              // C.1
  q_4d          = q_after_proj.reshape({1,4,8,256})
                    .withTags(.{.b, .s, .nh, .hd})                  // piège #1
  q_normalized  = zml.nn.rmsNorm(q_4d, .hd, 1e-6)
  q_after_norm  = q_normalized.mul(q_norm_weight.broad(...))        // C.2 (pattern Llama)
  q_after_rope  = zml.nn.rope(q_after_norm, null, .{
                    .layout = .sequential,
                    .scaling = .{ .default = .{ .rope_theta = 10_000 } },
                  })
  ```
- **4 tenseurs chargés** : `hidden_input`, `q_proj_weight`, `q_norm_weight`, `q_after_rope` (oracle).
- **Tags** : passage de `.{.b,.s,.n,.d}` (C.2) à `.{.b,.s,.nh,.hd}` (C.3) — ZML rope helper requiert `.hd` strictement.
- **Interdits stricts** : transpose final `[.b,.nh,.s,.hd]`, K/V projection, attention scores, softmax, cache, sliding mask.

### Livrables
| Fichier | Rôle |
|---|---|
| `zml_runner/gemma4_q_rope.zig` | runner ZML (utilise `zml.nn.rope` natif) |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_q_rope` |
| `logs/P5_2_C3_q_rope.log` | sortie `bazel run` ~76 lignes |
| `docs/P5_2_attention_forward.md` | ce document (section C.3) |

### Validation
**4 blocks fixed-point** :
- Block A `[0,0,0,:8]` pos=0 — max_diff=**3.81e-6** (= C.2 bloc A, RoPE identité confirmée à pos 0)
- Block B `[0,0,7,:8]` pos=0 — max_diff=**1.19e-6** (= C.2 bloc B)
- Block C `[0,3,0,:8]` pos=3 — max_diff=**1.79e-6** (RoPE active, valeurs ≠ q_after_norm)
- Block D `[0,3,7,:8]` pos=3 — max_diff=**3.10e-6**
- **4 blocks max_diff = 3.81e-6**

**Scan global 8192 valeurs** :
- max_abs = **6.676e-6** à `flat_index=1926 (s=0, n=7, d=134)` — **identique à C.2**
- mean_abs = **4.96e-7**
- Tolerance 1e-4 → marge ~15× sous le seuil

### Observations majeures
- **RoPE est orthogonale par paires** : rotation `(cos θ, -sin θ ; sin θ, cos θ)` préserve la norme L2 de chaque paire `(x_real, x_imag)`. Donc max_abs C.3 = max_abs C.2 exactement. Aucun bruit propagé par la rotation.
- **Position du max conservée** depuis C.1 : `flat_index 1926` ↔ `(s=0, o=1926)` ↔ `(s=0, n=7, d=134)` ↔ `(s=0, nh=7, hd=134)` (1926 = 7×256 + 134). L'erreur arithmétique vient strictement du matmul de C.1, propagée sans amplification ni atténuation à travers q_norm + RoPE.
- **ZML rope natif ≡ PyTorch `apply_rotary_pos_emb` à 1e-5 près** sans utiliser les cos/sin pré-calculés du fixture. Donc les `inv_freq` ZML (fp32 `exp(-log(theta)*n/N)`) convergent avec PyTorch (`1/pow(theta, 2n/D)`) sous le seuil.

### Notes capitalisées
- **`zml.nn.rope` est l'helper canonique** — ne pas réimplementer manuellement.
- **Convention tag stricte** : `zml.nn.rope` requiert `.s` et `.hd`. Si layout source utilise d'autres noms (`.d`, `.head_dim`...), renommer via `.rename(...)` ou utiliser `.withTags(...)` après reshape.
- **`pos_idx = null`** est suffisant pour prefill ; pour decode incrémental, passer `arange + token_index.broad(...)` (cf llama L502-L505).
- **Layout `.sequential`** = HF (split-half), `.interleaved` = GGUF. Pour Gemma 4 / Llama / Mistral / Qwen ChatML, toujours `.sequential`.

### Critères de clôture P5.2.C.3 (et de C complet)
- [x] Build bazel PASS
- [x] Run produit `q_after_rope_zml [1,4,8,256]`
- [x] 4/4 fixed-point blocks PASS (max_diff bloc max = 3.81e-6)
- [x] Scan global 8192 valeurs : max_abs 6.68e-6 (= C.2, RoPE orthogonale)
- [x] Position 0 : RoPE identité confirmée (block A et B == C.2)
- [x] Position 3 : RoPE active (block C et D ≠ q_after_norm correspondants)
- [x] ZML inv_freq ≡ PyTorch inv_freq sous tolerance
- [x] Aucun transpose, aucun K/V, aucune attention
- [x] Log archivé `logs/P5_2_C3_q_rope.log`

**Tag** : `p5.2-c3-zml-rope-q-only-pass`

---

## P5.2.D — K/V producer path layer 13 sliding (EN COURS)

P5.2.C COMPLET = chemin Q-only validé pour une layer reader sliding (layer 15) end-to-end avec écart numérique stable à 6.68e-6 (sub-tolerance 1e-4).

P5.2.D porte le chemin **K/V producer/writer** sur la **layer 13** (sliding, writer pré-shared, `is_writer=True`, `first_kv_shared_layer_idx=15`). Choix layer 13 (pas 14) : réutilise la RoPE sliding theta=10000 de C.3 et évite la `rope_type=proportional` + `partial_rotary_factor=0.25` de la full attention (helper `zml.nn.rope` ne supporte pas encore ce mode — réservé pour layer 14 plus tard).

Sous-sous-gates :

| Sous-sous-gate | Périmètre | État |
|---|---|---|
| **P5.2.D.0** | PyTorch oracle K/V producer+writer layer 13 sliding | **PASS** |
| **P5.2.D.1** | ZML k_proj uniquement (single dot, reduce .h) | **PASS** |
| P5.2.D.2 | ZML v_proj uniquement (single dot, reduce .h) | pending |
| P5.2.D.3 | ZML k_norm (reshape + rmsNorm + mul) | pending |
| P5.2.D.4 | ZML RoPE K (helper natif `zml.nn.rope` sliding) | pending |
| P5.2.D.5 | ZML KV slot mock (cache write factice) | pending |

---

## P5.2.D.0 — PyTorch oracle K/V producer+writer layer 13 sliding (PASS, 28 mai 2026)

### Objectif
Premier calcul du chemin **producer K/V** : produire un oracle PyTorch fp32 du pipeline `k_proj / v_proj → view → k_norm → RoPE(K) → transpose` pour la layer 13 (sliding writer, première frontière où on touche aux poids producer). Sérialiser comme fixture `.pt` à 16 tenseurs pour valider les sous-sous-gates D.1 → D.5 byte-par-byte.

### Périmètre strict
- **Cible** : layer 13 (sliding writer, `is_writer=True`, `first_kv_shared = 15`).
- **Input** : synthétique déterministe `torch.manual_seed(1337)`, shape `[B=1, S=4, H=1536]` (mêmes invariants que C.0).
- **Poids chargés** : `k_proj.weight [256, 1536]`, `v_proj.weight [256, 1536]`, `k_norm.weight [256]` depuis `weights/model.safetensors` (raw safetensors, pas de chargement modèle complet).
- **V-norm absent du checkpoint** : assertion forte (`v_norm.weight` doit ne pas exister). Gemma 4 = K-norm seulement, NE PAS halluciner v_norm.
- **Modules instanciés** :
  - `torch.nn.Linear(1536, 256, bias=False)` pour k_proj
  - `torch.nn.Linear(1536, 256, bias=False)` pour v_proj
  - `Gemma4RMSNorm(256, eps=1e-6)` pour k_norm
  - `Gemma4TextRotaryEmbedding(text_config)` pour rotary (sliding theta=10000)
- **Interdits stricts** : q_proj, q_norm, attention scores, matmul QK, softmax, cache, sliding mask, layer 14 full attention, p-RoPE proportional.

### Pipeline (miroir verbatim `Gemma4TextAttention.forward` L811-L829)
```
A) k_after_proj    = k_proj(hidden_input)                                           [1,4,256]
   v_after_proj    = v_proj(hidden_input)                                           [1,4,256]
B) k_after_reshape = k_after_proj.view(1, 4, n_kv=1, head_dim=256)                  [1,4,1,256]
   v_after_reshape = v_after_proj.view(1, 4, n_kv=1, head_dim=256)                  [1,4,1,256]
C) k_after_norm    = k_norm(k_after_reshape)                                        [1,4,1,256]   (V non normé en Gemma 4)
D) (cos, sin)      = rotary(hidden_input, position_ids=arange(4), layer_type="sliding_attention")
   k_after_rope    = apply_rotary_pos_emb(k_after_norm, cos, sin, unsqueeze_dim=2)  [1,4,1,256]   (V non roté)
E) k_final         = k_after_rope.transpose(1, 2).contiguous()                      [1,1,4,256]
   v_final         = v_after_reshape.transpose(1, 2).contiguous()                   [1,1,4,256]
```

### Sanity checks
- `v_norm.weight` absent du safetensors (assertion hard).
- RoPE pos 0 = identité bit-exact (`|k_rope - k_norm|_max == 0.0`).
- RoPE pos 3 active (`|k_rope - k_norm|_max > 1e-3`, observé ~2.64e-1).

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/14_kv_oracle_layer13.py` | générateur oracle (Python, depuis raw safetensors) |
| `fixtures/p5_2_d0_kv_oracle_layer13.pt` | 16 tenseurs (input + 3 poids + 2 rotary + 10 intermédiaires K/V) ~3.2 MB |
| `fixtures/p5_2_d0_kv_oracle_layer13_manifest.json` | shapes/dtypes + spec refs + pipeline + interdits + sanity RoPE |

**Tag** : `p5.2-d0-pytorch-kv-oracle-pass`

---

## P5.2.D.1 — ZML k_proj producer layer 13 (PASS, 28 mai 2026)

### Objectif
Porter la projection linéaire `k_proj` en ZML pour layer 13 (sliding writer), exécuter le forward sur `hidden_input` du fixture D.0, comparer le résultat byte-équivalent contre l'oracle `k_after_proj`.

### Périmètre strict
- **Pipeline ZML** : un seul `dot` réduisant `.h` :
  ```zig
  k_after_proj_zml = hidden_input.dot(k_proj_weight, .h)
  // [.b=1, .s=4, .h=1536] dot [.kv=256, .h=1536] -> [.b=1, .s=4, .kv=256]
  ```
- **Tag axe sortie** : `.kv` (1 head × 256 head_dim — lisible, distingue du `.o` Q-path).
- **3 tenseurs chargés** depuis `fixtures/p5_2_d1_k_proj_layer13.safetensors` (fixture slim re-exportée depuis le `.pt` D.0 par `scripts/15_p5_2_d1_export_fixture.py`) : `hidden_input`, `k_proj_weight`, `k_after_proj` (oracle).
- **Interdits stricts** : v_proj, k_norm, RoPE, reshape `[B,S,n_kv,head_dim]`, transpose `[B,n_kv,S,head_dim]`, cache slot, attention scores, matmul QK, softmax, sliding mask.

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/15_p5_2_d1_export_fixture.py` | export safetensors slim depuis le `.pt` D.0 |
| `fixtures/p5_2_d1_k_proj_layer13.safetensors` | 3 tenseurs (input + 1 poids + oracle) ~1.6 MB |
| `fixtures/p5_2_d1_k_proj_layer13_manifest.json` | shapes/dtypes + pipeline + interdits |
| `zml_runner/gemma4_k_proj.zig` | runner ZML (pattern C.1 q_proj, sortie `.kv=256`) |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_k_proj` avec deps `//bazel` + `//zml` |
| `logs/P5_2_D1_k_proj.log` | sortie `bazel run` |

### Résultats numériques observés
- Forward result shape : `{b=1, s=4, kv=256, f32}` ✓
- Block A [0,0,:8] max_diff : **6.41e-7**
- Block B [0,1,:8] max_diff : **6.85e-7**
- Block C [0,3,:8] max_diff : **1.31e-6**
- Scan global 1024 fp32 : **max_abs 5.48e-6 à flat_index 894 (s=3, d=126)**, mean_abs 5.54e-7
- Tolérance 1e-4 → marge ~18 000×
- Cohérent avec C.1 q_proj (1.5e-5) : résidu matmul PJRT-CPU Eigen-like vs PyTorch BLAS. Magnitude inférieure ici car réduction sur `.h=1536` aussi mais output 4× plus petit (256 vs 2048) → moins d'accumulations divergentes.

### Critères de clôture P5.2.D.1
- [x] Build bazel PASS sur 3090
- [x] Run produit `k_after_proj_zml [1,4,256]`
- [x] 3/3 fixed-point blocks `[0,0,:8]`, `[0,1,:8]`, `[0,3,:8]` rapportés vs oracle
- [x] Scan global 1024 valeurs : max_abs 5.48e-6 < tolerance 1e-4
- [x] mean_abs 5.54e-7 (résidu matmul attendu)
- [x] Aucun v_proj, aucun k_norm, aucun RoPE, aucun reshape, aucun transpose, aucun cache, aucune attention
- [x] Log archivé `logs/P5_2_D1_k_proj.log`

**Tag** : `p5.2-d1-zml-k-proj-pass`

---

## P5.2.D.2 → P5.2.E (à venir)

D.1 PASS = projection K validée pour layer 13 (sliding writer) avec écart numérique 5.48e-6 (sub-tolerance 1e-4).

Prochaine sous-sous-gate : **P5.2.D.2** = ZML `v_proj` uniquement. Même forme que k_proj (output `[1,4,256]`), autre branche du dot. Stop discipliné : pas d'enchaînement automatique, attendre cadrage explicite avant D.2.

P5.2.E suivra avec le sliding mask au compute.
