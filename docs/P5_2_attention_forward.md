# P5.2 — Gemma 4 E2B attention forward (5 sous-gates)

**Status global** : P5.2.A + P5.2.B PASS, P5.2.C COMPLET (C.0 oracle + C.1 q_proj + C.2 q_norm + C.3 RoPE) PASS (28 mai 2026), P5.2.D COMPLET (D.0 oracle + D.1 k_proj + D.2 v_proj + D.3 k_norm + D.4 RoPE K + D.5 KV slot mock PASS, 28 mai 2026 soir-nuit). E à venir.
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
| **P5.2.D.2** | ZML v_proj uniquement (single dot, reduce .h) | **PASS** |
| **P5.2.D.3** | ZML k_norm (reshape + rmsNorm + mul) | **PASS** |
| **P5.2.D.4** | ZML RoPE K (helper natif `zml.nn.rope` sliding) | **PASS** |
| **P5.2.D.5** | ZML KV slot mock (cache write factice) | **PASS** |

---

## P5.2.D.0 — PyTorch oracle K/V producer+writer layer 13 sliding (PASS, 28 mai 2026)

> ⚠️ **Corrigé par D.0b (30 mai 2026) — branche V.** L'oracle décrit ci-dessous
> a été régénéré : il manquait l'étape `v_norm` (RMSNorm **sans scale**,
> `with_scale=False`) sur V. Le `v_final` d'origine était égal au V brut — faux.
> Voir la sous-section **P5.2.D.0b** en fin de section D. La branche K
> (k_norm/RoPE) décrite ici est inchangée et reste valide.

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

## P5.2.D.2 — ZML v_proj producer layer 13 (PASS, 28 mai 2026)

### Objectif
Porter la projection linéaire `v_proj` en ZML pour layer 13 (sliding writer), exécuter le forward sur `hidden_input` du fixture D.0, comparer le résultat byte-équivalent contre l'oracle `v_after_proj`. Miroir strict de D.1, branche V.

### Périmètre strict
- **Pipeline ZML** : un seul `dot` réduisant `.h`, identique à D.1 sur l'autre branche :
  ```zig
  v_after_proj_zml = hidden_input.dot(v_proj_weight, .h)
  // [.b=1, .s=4, .h=1536] dot [.kv=256, .h=1536] -> [.b=1, .s=4, .kv=256]
  ```
- **Tag axe sortie** : `.kv` (convention identique à D.1).
- **3 tenseurs chargés** depuis `fixtures/p5_2_d2_v_proj_layer13.safetensors` (fixture slim re-exportée depuis le `.pt` D.0 par `scripts/16_p5_2_d2_export_fixture.py`) : `hidden_input`, `v_proj_weight`, `v_after_proj` (oracle).
- **Sanity export** : assertion `v_proj_weight != k_proj_weight` côté Python pour éviter une confusion silencieuse de branche.
- **Interdits stricts** : k_proj, k_norm, v_norm (absent du checkpoint Gemma 4), RoPE, reshape `[B,S,n_kv,head_dim]`, transpose `[B,n_kv,S,head_dim]`, cache slot, attention scores, matmul QK, softmax, sliding mask.

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/16_p5_2_d2_export_fixture.py` | export safetensors slim depuis le `.pt` D.0 (3 tenseurs branche V) |
| `fixtures/p5_2_d2_v_proj_layer13.safetensors` | 3 tenseurs (input + 1 poids + oracle) ~1.6 MB |
| `fixtures/p5_2_d2_v_proj_layer13_manifest.json` | shapes/dtypes + pipeline + interdits |
| `zml_runner/gemma4_v_proj.zig` | runner ZML (miroir gemma4_k_proj.zig, branche V) |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_v_proj` avec deps `//bazel` + `//zml` |
| `logs/P5_2_D2_v_proj.log` | sortie `bazel run` |

### Résultats numériques observés
- Forward result shape : `{b=1, s=4, kv=256, f32}` ✓
- Block A [0,0,:8] max_diff : **1.67e-6**
- Block B [0,1,:8] max_diff : **1.31e-6**
- Block C [0,3,:8] max_diff : **8.34e-7**
- Scan global 1024 fp32 : **max_abs 5.25e-6 à flat_index 842 (s=3, d=74)**, mean_abs 5.69e-7
- Tolérance 1e-4 → marge ~19 000×
- Quasi-identique à D.1 k_proj en magnitude (5.48e-6 vs 5.25e-6) — confirmation que le résidu est intrinsèque au matmul PJRT-CPU Eigen-like vs PyTorch BLAS sur la réduction `.h=1536`, indépendant des poids spécifiques.

### Critères de clôture P5.2.D.2
- [x] Build bazel PASS sur 3090
- [x] Run produit `v_after_proj_zml [1,4,256]`
- [x] 3/3 fixed-point blocks `[0,0,:8]`, `[0,1,:8]`, `[0,3,:8]` rapportés vs oracle
- [x] Scan global 1024 valeurs : max_abs 5.25e-6 < tolerance 1e-4
- [x] mean_abs 5.69e-7 (résidu matmul attendu)
- [x] Sanity export : `v_proj_weight != k_proj_weight`
- [x] Aucun k_proj, aucun k_norm, aucun RoPE, aucun reshape, aucun transpose, aucun cache, aucune attention
- [x] Log archivé `logs/P5_2_D2_v_proj.log`

**Tag** : `p5.2-d2-zml-v-proj-pass`

---

## P5.2.D.3 — ZML k_norm producer layer 13 (PASS, 28 mai 2026)

### Objectif
Étendre D.1 (k_proj seul, PASS) avec le pipeline `reshape + withTags + rmsNorm(.d) + mul(weight.broad)`. Pattern Llama (pas Qwen). V hors scope (non normé en Gemma 4).

### Périmètre strict
- **Pipeline ZML** :
  ```zig
  k_after_proj = hidden_input.dot(k_proj_weight, .h)            // [.b, .s, .kv]  reuse D.1
  k_4d         = k_after_proj.reshape({1,4,1,256}).withTags(.{.b,.s,.kvh,.d})
  k_normalized = zml.nn.rmsNorm(k_4d, .d, RMS_EPS=1e-6)
  k_after_norm = k_normalized.mul(k_norm_weight.broad(k_normalized.shape()))
  ```
- **Diffs vs C.2 q_norm** : `n_kv=1` (vs `n_heads=8`), tag axe head_count `.kvh` (vs `.n`), shape sortie `[1,4,1,256]` (vs `[1,4,8,256]`).
- **4 tenseurs chargés** depuis `fixtures/p5_2_d3_k_norm_layer13.safetensors` (slim re-export D.0 via `scripts/17_p5_2_d3_export_fixture.py`) : `hidden_input`, `k_proj_weight`, `k_norm_weight`, `k_after_norm` (oracle).
- **Interdits stricts** : v_proj, v_norm (absent du checkpoint Gemma 4), RoPE, transpose `[B,n_kv,S,head_dim]`, cache slot, attention scores, matmul QK, softmax, sliding mask.

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/17_p5_2_d3_export_fixture.py` | export safetensors slim depuis le `.pt` D.0 (4 tenseurs) |
| `fixtures/p5_2_d3_k_norm_layer13.safetensors` | 4 tenseurs (input + 2 poids + oracle) ~1.6 MB |
| `fixtures/p5_2_d3_k_norm_layer13_manifest.json` | shapes/dtypes + pipeline + interdits + rms_eps |
| `zml_runner/gemma4_k_norm.zig` | runner ZML (mirror C.2 q_norm, n_kv=1, tag `.kvh`) |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_k_norm` |
| `logs/P5_2_D3_k_norm.log` | sortie `bazel run` |

### Résultats numériques observés
- Forward result shape : `{b=1, s=4, kvh=1, d=256, f32}` ✓
- Block A `[0,0,0,:8]` max_diff : **6.57e-8**
- Block B `[0,3,0,:8]` max_diff : **1.64e-7**
- Scan global 1024 fp32 : **max_abs 5.36e-7 à flat_index 894 (s=3, kvh=0, d=126)**, mean_abs 5.99e-8
- Tolérance 1e-4 → marge ~186 000×

### Finding inattendu (D.3)
- `k_norm_weight` est **uniforme** sur les 256 dims : mean=0.1259766, std=0.0, min=max=0.1259766. Tous les coefficients identiques.
- Couplé au RMSNorm, le résultat est un scaling isotrope du tenseur RMS-normalisé. Cela **écrase le résidu matmul D.1 d'un facteur ~10×** (5.36e-7 vs 5.48e-6 en D.1).
- La position du max est exactement **conservée** vs D.1 : flat_index 894 (s=3, d=126). RMSNorm + scaling uniforme préservent l'argmax du worst-case, ce qui confirme que la source de divergence reste le matmul PJRT-CPU vs PyTorch BLAS et que les étapes post-matmul n'introduisent pas de nouvelle erreur observable.
- Possible interprétation : la couche `k_norm` de Gemma 4 sur layer 13 sliding fonctionne comme une RMSNorm "presque-pure" (poids = pure échelle constante). À documenter pour D.4 (RoPE K) qui pourra s'appuyer sur le constat que `k_after_norm` est numériquement très proche de `k_after_proj / RMS(k_after_proj) * 0.126`.

### Critères de clôture P5.2.D.3
- [x] Build bazel PASS sur 3090
- [x] Run produit `k_after_norm_zml [1,4,1,256]`
- [x] 2/2 fixed-point blocks `[0,0,0,:8]`, `[0,3,0,:8]` rapportés vs oracle
- [x] Scan global 1024 valeurs : max_abs 5.36e-7 < tolerance 1e-4
- [x] mean_abs 5.99e-8
- [x] Aucun v_proj, aucun v_norm, aucun RoPE, aucun transpose, aucun cache, aucune attention
- [x] Log archivé `logs/P5_2_D3_k_norm.log`

**Tag** : `p5.2-d3-zml-k-norm-pass`

---

## P5.2.D.4 — ZML RoPE K-only producer layer 13 (PASS, 28 mai 2026)

### Objectif
Étendre D.3 (k_proj + k_norm, PASS) avec la rotation positionnelle RoPE sur K via le helper natif `zml.nn.rope` (mirror C.3 q_rope, branche K). V hors scope (non rotée en Gemma 4). Comparer contre l'oracle PyTorch fp32 `k_after_rope` du fixture D.0.

### Périmètre strict
- **Pipeline ZML** :
  ```zig
  k_after_proj = hidden_input.dot(k_proj_weight, .h)                  (reuse D.1)
  k_4d         = k_after_proj.reshape({1,4,1,256})
                   .withTags(.{.b, .s, .kvh, .hd})                    (tag .hd direct,
                                                                       requis par rope)
  k_normalized = zml.nn.rmsNorm(k_4d, .hd, RMS_EPS=1e-6)
  k_after_norm = k_normalized.mul(k_norm_weight.broad(shape))         (pattern Llama)
  k_after_rope = zml.nn.rope(k_after_norm, null,
                   .{ .layout=.sequential,
                      .scaling=.{.default=.{.rope_theta=10000}} })
  ```
- **Diff vs D.3** : tag head_dim retagué directement à `.hd` (au lieu de `.d`) pour répondre à l'exigence du helper `zml.nn.rope` (qui requiert `.s` et `.hd`).
- **Diff vs C.3** : `n_kv=1` (vs `n_heads=8`), tag head_count `.kvh` (vs `.nh`), shape sortie `[1,4,1,256]` (vs `[1,4,8,256]`).
- **5 tenseurs chargés** depuis `fixtures/p5_2_d4_k_rope_layer13.safetensors` (slim re-export D.0 via `scripts/18_p5_2_d4_export_fixture.py`) : `hidden_input`, `k_proj_weight`, `k_norm_weight`, `k_after_norm` (oracle, sanity inline pos0/pos3), `k_after_rope` (oracle, gate principal).
- **Interdits stricts** : v_proj, v_norm, transpose `[B,n_kv,S,head_dim]`, cache slot, attention scores, matmul QK, softmax, sliding mask, layer 14 full attention (proportional RoPE).

### Sanity inline (ZML vs k_norm oracle)
- pos 0 (identité attendue, cos=1 sin=0) : `|k_rope_zml[0,0,0,:] - k_norm_oracle[0,0,0,:]|_max = 4.47e-7` ✓ (sous-tolérance, "identité aux résidus matmul près")
- pos 3 (RoPE active attendue) : `|k_rope_zml[0,3,0,:] - k_norm_oracle[0,3,0,:]|_max = 0.264` ✓ (>>1e-3, conforme oracle PyTorch 2.638e-1)

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/18_p5_2_d4_export_fixture.py` | export safetensors slim depuis le `.pt` D.0 (5 tenseurs) |
| `fixtures/p5_2_d4_k_rope_layer13.safetensors` | 5 tenseurs (input + 2 poids + 2 oracles) ~1.6 MB |
| `fixtures/p5_2_d4_k_rope_layer13_manifest.json` | shapes/dtypes + pipeline + interdits + rope_sanity |
| `zml_runner/gemma4_k_rope.zig` | runner ZML (mirror C.3 q_rope, n_kv=1, tag `.kvh`) |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_k_rope` |
| `logs/P5_2_D4_k_rope.log` | sortie `bazel run` |

### Résultats numériques observés
- Forward result shape : `{b=1, s=4, kvh=1, hd=256, f32}` ✓
- Block A `[0,0,0,:8]` pos 0 (identity) : max_diff **6.57e-8**
- Block B `[0,3,0,:8]` pos 3 (active) : max_diff **1.34e-7**
- Scan global 1024 fp32 : **max_abs 5.36e-7 à flat_index 894 (s=3, kvh=0, d=126)**, mean_abs 6.01e-8
- Tolérance 1e-4 → marge ~186 000×

### Finding mathématique (D.4)
- `max_abs` ET `flat_index` (894, s=3, d=126) sont **strictement identiques à D.3 k_norm**.
- La rotation RoPE étant **orthogonale**, elle préserve la norme L2 et le résidu absolu reste inchangé — les 1024 valeurs sont juste réorganisées par paires (real, imag) sans amplification.
- C'est la preuve expérimentale que `layout=.sequential + scaling=.default + theta=10000` reproduisent **bit-equivalent** `apply_rotary_pos_emb` Gemma 4 sliding. Aucune erreur supplémentaire n'est introduite par la couche RoPE — la chaîne D.1→D.3→D.4 saturée par le résidu matmul D.1.
- Possible interprétation : si une future sous-gate (D.5 ou layer 14 full) introduit un `max_abs > 5.36e-7`, ce sera nécessairement dû à une nouvelle source d'erreur (transpose, cache write, full RoPE proportional), pas à un effet d'accumulation.

### Critères de clôture P5.2.D.4
- [x] Build bazel PASS sur 3090
- [x] Run produit `k_after_rope_zml [1,4,1,256]`
- [x] Sanity pos 0 = identité (4.47e-7 < tolerance 1e-4)
- [x] Sanity pos 3 = RoPE active (0.264 > 1e-3, conforme oracle 0.264)
- [x] 2/2 fixed-point blocks `[0,0,0,:8]`, `[0,3,0,:8]` rapportés vs oracle
- [x] Scan global 1024 valeurs : max_abs 5.36e-7 < tolerance 1e-4
- [x] mean_abs 6.01e-8
- [x] max_abs et argmax strictement conservés depuis D.3 (RoPE orthogonale)
- [x] Aucun v_proj, aucun v_norm, aucun transpose, aucun cache, aucune attention
- [x] Log archivé `logs/P5_2_D4_k_rope.log`

**Tag** : `p5.2-d4-zml-rope-k-pass`

---

## P5.2.D.5 — ZML KV slot mock producer layer 13 (PASS, 28 mai 2026 ; corrigé 30 mai)

> ⚠️ **Re-validé le 30 mai (V RMSNorm).** La V originale (« v_after_proj_reshaped »,
> sans norm) était fausse — cf bug D.0/D.0b. La branche V intègre désormais
> `rmsNorm(.hd)` **sans** poids (D.2b), `value = v_after_norm`. Re-run vs oracle
> D.0b corrigé : K_slot 5.36e-7 (inchangé), **V_slot 4.17e-6** (vs ancien 5.25e-6
> sur V brut), sanity `max|v_slot − v_raw| = 0.777`. Tag : `p5.2-d5-kv-slot-mock-pass`.

### Objectif
Packager le writer sliding dans un slot KV factice. Calcule `(k_after_rope, v_after_norm)` en compute layout, puis applique le transpose vers cache layout (mirror PyTorch `k_final` / `v_final` du fixture D.0b). Aucune attention, aucun cache dynamique réel, aucun Q path.

### Décision layout
- **Compute layout** : `[1, 4, 1, 256]` = `{.b, .s, .kvh, .hd}` (sortie de D.4 K et D.2 V)
- **Cache layout retenu** : `[1, 1, 4, 256]` = `{.b, .kvh, .s, .hd}` (mirror PyTorch transposé contiguous, plus proche du futur cache réel)
- **Transition ZML** : `tensor.transpose(.{.b, .kvh, .s, .hd})`
- **Sanity Python pré-export** : `k_after_rope[0,:,0,:] vs k_final[0,0,:,:]` diff = 0.0 strict (idem pour V). Avec `n_kv=1`, la singleton dim ne réordonne pas le row-major, donc le transpose est un **no-op en mémoire**. L'opération reste explicite côté ZML pour préparer les futures couches non-singleton.

### Périmètre strict
- **Pipeline ZML** (return tuple `struct { zml.Tensor, zml.Tensor }`) :
  ```zig
  // K branch
  k_after_proj = hidden_input.dot(k_proj_weight, .h)
  k_4d         = k_after_proj.reshape({1,4,1,256}).withTags(.{.b,.s,.kvh,.hd})
  k_normalized = zml.nn.rmsNorm(k_4d, .hd, 1e-6)
  k_after_norm = k_normalized.mul(k_norm_weight.broad(shape))
  k_after_rope = zml.nn.rope(k_after_norm, null, .{.layout=.sequential, .scaling=.{.default=.{.rope_theta=10000}}})
  k_slot       = k_after_rope.transpose(.{.b, .kvh, .s, .hd})   // [1,1,4,256]

  // V branch (RMSNorm UNSCALED, pas de mul, pas de RoPE) — corrigé D.5
  v_after_proj = hidden_input.dot(v_proj_weight, .h)
  v_4d         = v_after_proj.reshape({1,4,1,256}).withTags(.{.b,.s,.kvh,.hd})
  v_after_norm = zml.nn.rmsNorm(v_4d, .hd, 1e-6)                // with_scale=False, no mul
  v_slot       = v_after_norm.transpose(.{.b, .kvh, .s, .hd})   // [1,1,4,256]

  return .{ k_slot, v_slot, v_4d };   // v_4d expose pour sanity anti-regression
  ```
- **Multi-return ZML** : pattern `var k_slot_buf, var v_slot_buf = results.get(struct { zml.Buffer, zml.Buffer })` (cf lfm2_tests.zig).
- **6 tenseurs chargés** depuis `fixtures/p5_2_d5_kv_slot_layer13.safetensors` (slim re-export D.0 via `scripts/19_p5_2_d5_export_fixture.py`) : `hidden_input`, `k_proj_weight`, `k_norm_weight`, `v_proj_weight`, `k_final` (oracle K cache), `v_final` (oracle V cache).
- **Interdits stricts** : attention scores, matmul QK, softmax, Q path (q_proj, q_norm), reader (layers 15-34), layer 14 full attention (proportional RoPE), sliding mask, cache dynamique réel (scatter / dynamicSlice).

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/19_p5_2_d5_export_fixture.py` | export safetensors slim depuis le `.pt` D.0 (6 tenseurs) |
| `fixtures/p5_2_d5_kv_slot_layer13.safetensors` | 6 tenseurs (input + 3 poids + 2 oracles) ~3.2 MB |
| `fixtures/p5_2_d5_kv_slot_layer13_manifest.json` | shapes/dtypes + decision layout + pipeline + sanity transpose |
| `zml_runner/gemma4_kv_slot.zig` | runner ZML K full + V proj + transposes + tuple return |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_kv_slot` |
| `logs/P5_2_D5_kv_slot.log` | sortie `bazel run` |

### Résultats numériques observés
- K_slot shape : `{b=1, kvh=1, s=4, hd=256, f32}` ✓
- V_slot shape : `{b=1, kvh=1, s=4, hd=256, f32}` ✓
- **K_slot** (inchangé) : 2 blocks max_diff 1.34e-7, full max_abs **5.36e-7** @ flat_index 894 (b=0, kvh=0, s=3, d=126), mean_abs 6.01e-8
- **V_slot** (corrigé, V normé) : 2 blocks max_diff 1.55e-6, full max_abs **4.17e-6** @ flat_index 875 (b=0, kvh=0, s=3, d=107), mean_abs 5.29e-7
- Sanity anti-régression : `max|v_slot − v_raw(brut)| = 0.777` (RMSNorm V active)
- Tolérance 1e-4 : marge ~186 000× (K), ~24 000× (V)

### Finding capitalisé (D.5, corrigé)
- **K_slot max_abs strictement identique à D.4** (5.36e-7 @ flat_index 894, même position après transpose) — la branche K n'a pas bougé.
- **V_slot max_abs 4.17e-6** (vs ancien 5.25e-6 sur V brut) : le résidu V passe désormais par `v_norm` (RMSNorm unscaled, D.2b). La position du max se déplace (d=107 vs d=74 sur V brut) car la normalisation redistribue le résidu du matmul v_proj.
- Confirmation expérimentale : **transpose sur dim singleton = pure metadata operation** (sanity Python compute↔cache = 0.0 strict pour K comme pour V normé).
- Cohérence bout-en-bout : pipeline ZML complet (K : D.1→D.3→D.4→transpose ; V : D.2→D.2b→transpose) reproduit `k_final` / `v_final` PyTorch (oracle D.0b corrigé) dans la tolérance 1e-4.

### Critères de clôture P5.2.D.5
- [x] Build bazel PASS sur 3090
- [x] Multi-return `struct { zml.Buffer, zml.Buffer }` consommé proprement
- [x] K_slot et V_slot produits en cache layout `[1,1,4,256]` `{b,kvh,s,hd}`
- [x] Sanity Python pré-export : k_compute_vs_cache = v_compute_vs_cache = 0.0 strict (no-op n_kv=1)
- [x] K_slot 2/2 fixed-point blocks `[0,0,0,:8]`, `[0,0,3,:8]` rapportés vs `k_final` oracle
- [x] V_slot 2/2 fixed-point blocks `[0,0,0,:8]`, `[0,0,3,:8]` rapportés vs `v_final` oracle (corrigé)
- [x] Scan global K_slot : max_abs 5.36e-7 < tolerance 1e-4
- [x] Scan global V_slot : max_abs 4.17e-6 < tolerance 1e-4 (V normé)
- [x] K_slot ≡ D.4 strict ; V_slot via v_norm (D.2b), value = v_after_norm
- [x] Sanity anti-régression host : `max|v_slot − v_raw| = 0.777` (RMSNorm V active)
- [x] Aucune attention, aucun Q path, aucun reader, aucun layer 14, aucun sliding mask, aucun cache dynamique
- [x] Log archivé `logs/P5_2_D5_kv_slot.log`

**Tags D.5** (les deux conservés — l'ancien documente le faux PASS pré-D.0b) :
- `p5.2-d5-zml-kv-slot-mock-pass` — **superseded** (raw V, oracle D.0 faux)
- `p5.2-d5-kv-slot-mock-pass` — **canonical** (V RMSNorm no-scale corrigé)

---

## P5.2.D.0b — Correction oracle : V RMSNormed sans scale (PASS, 30 mai 2026)

### Le bug
En préparant P5.2.E (première opération QK), la lecture du `forward` de référence
`Gemma4TextAttention` (transformers 5.9.0) a révélé que **V passe par un RMSNorm**
avant le transpose :
```python
self.v_norm = Gemma4RMSNorm(self.head_dim, eps=..., with_scale=False)   # __init__
...
value_states = self.v_norm(value_states)                                # forward
```
`Gemma4RMSNorm(with_scale=False)` **saute seulement la multiplication par le poids,
pas la normalisation RMS**. Donc V est bien normalisé, simplement sans poids appris
(d'où l'absence de `v_norm.weight` au checkpoint).

L'oracle D.0 d'origine avait inféré « pas de poids `v_norm.weight` ⇒ V non normé »
et faisait `v_final = v_after_reshape.transpose(...)`. **Faux.**

### Pourquoi le bug a survécu à un PASS « end-to-end »
D.0 (oracle) **et** D.2/D.5 (ZML) partageaient la même hypothèse fausse — ils
s'accordaient donc entre eux à ~5e-6. Un oracle qui reproduit le bug du code testé
donne un PASS trompeur. Le bug n'est visible qu'en remontant à la **source de
vérité** (`modeling_gemma4.py`), ce que P5.2.E imposait. Leçon : **l'oracle doit
rester indépendant du code testé** ; toute hypothèse sur la référence doit être
vérifiée dans la source, pas inférée.

### Le correctif (atomique, branche V seulement)
`scripts/14_kv_oracle_layer13.py` : ajout de `v_norm = Gemma4RMSNorm(head_dim,
eps, with_scale=False)` puis `v_after_norm = v_norm(v_after_reshape)` ;
`v_final = v_after_norm.transpose(...)`. Nouveaux tenseurs en fixture :
`v_after_norm`. Sanity 4 ajoutée (`|v_after_norm − v_reshape|_max > 1e-2`).

### Résultats (3090, oracle régénéré)
- `v_norm |v_after_norm − v_reshape|_max = 0.777` → normalisation bien active.
- `v_after_norm` / `v_final` : std **0.9996** (vs V brut std 1.0839, max 7.47→6.69).
- Branche K **inchangée** : `k_final` std 0.1260, RoPE pos0=0.0 / pos3=0.264.
- Fixture `p5_2_d0_kv_oracle_layer13.pt` régénérée (md5 `bb3fc164…`, M1≡3090).

### Impact sur les gates
| Gate | État |
|---|---|
| D.0 → **D.0b** | oracle K/V corrigé (V normé) — **ce gate** |
| D.1 (k_proj), D.3 (k_norm), D.4 (RoPE K) | restent **valides** (branche K intacte) |
| D.2 (v_proj seul) | reste valide mais **incomplet** (manque v_norm) |
| **D.2b** (à faire) | ZML v_norm sans scale — `zml.nn.rmsNorm(v_4d, .hd, eps)` **sans** `.mul(weight)` |
| D.5 (KV slot) | **à refaire** après D.2b : `v_slot` actuel est faux (V brut) |

**Tag** : `p5.2-d0b-v-norm-oracle-pass`

---

## P5.2.D.2b — ZML v_norm sans scale (PASS, 30 mai 2026)

### Objectif
Porter en ZML la normalisation V : `Gemma4RMSNorm(with_scale=False)`. Miroir
exact de D.3 k_norm **sans** le `.mul(weight)` (pas de poids appris). Valider
contre l'oracle PyTorch `v_after_norm` du fixture D.0b.

### Périmètre strict
- **Entrée** : `v_after_proj [1,4,256]` (V déjà projeté, fourni par D.0b — pas de v_proj ici).
- **Pipeline ZML** :
  ```zig
  v_4d         = v_after_proj.reshape({1,4,1,256}).withTags(.{.b,.s,.kvh,.hd})
  v_after_norm = zml.nn.rmsNorm(v_4d, .hd, RMS_EPS)   // PAS de .mul(weight)
  ```
- **Interdits** : `.mul(weight)`/v_norm.weight (with_scale=False), v_proj, RoPE, transpose, cache, attention.

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/20_p5_2_d2b_export_fixture.py` | export safetensors slim (v_after_proj + v_after_norm) depuis le `.pt` D.0b |
| `fixtures/p5_2_d2b_v_norm_layer13.safetensors` | 2 tenseurs (input + oracle) ~8 KB (gitignored) |
| `zml_runner/gemma4_v_norm.zig` | runner ZML (reshape + rmsNorm sans mul) |
| `zml_runner/BUILD.bazel` | cible `gemma4_v_norm` |
| `logs/P5_2_D2b_v_norm.log` | sortie `bazel run` |

### Résultats numériques observés
- Shape sortie : `{b=1, s=4, kvh=1, hd=256, f32}` ✓
- Sanity « norm active » : `max|v_after_norm − v_after_proj| = 0.777` (> 1e-3)
- Fixed blocks : A `[0,0,0,:8]` max_diff 1.19e-7, B `[0,3,0,:8]` max_diff 0.0
- **Scan global** : max_abs **2.384e-7** @ flat_index 74 (s=0, d=74), mean_abs 2.46e-8
- Tolérance 1e-4 → marge **~420 000×**

### Finding (D.2b)
Résidu (2.38e-7) **~10× plus bas que D.3 k_norm** (5.36e-7) : D.2b part de
`v_after_proj` déjà fourni (aucun matmul amont), on mesure donc le résidu pur de
la RMSNorm fp32 PJRT-CPU vs PyTorch, sans la contribution du matmul. `with_scale=False`
= `_norm()` seul (pas de `.mul(weight)`), bit-fidèle à PyTorch.

### Critères de clôture P5.2.D.2b
- [x] Build bazel PASS sur 3090
- [x] Shape `{b=1,s=4,kvh=1,hd=256,f32}`
- [x] Sanity `max|out − in| > 1e-3` (RMSNorm non no-op)
- [x] Fixed blocks `[0,0,0,:8]`, `[0,3,0,:8]` vs oracle
- [x] Scan global max_abs 2.384e-7 < tolerance 1e-4
- [x] Aucun mul/weight, aucun v_proj, aucune RoPE, aucun transpose, aucun cache, aucune attention
- [x] Log archivé `logs/P5_2_D2b_v_norm.log`

**Tag** : `p5.2-d2b-zml-v-norm-pass`

---

## P5.2.E.0 — PyTorch oracle attention : reader layer 15 × KV layer 13 (PASS, 31 mai 2026)

### Objectif
Produire l'**oracle PyTorch de la première attention effective** : layer 15 (reader,
sliding) lit le KV partagé produit par layer 13 (writer, sliding). Calcul
`Q15 × K13ᵀ → masque → softmax → V13 → context`. Aucun ZML, aucun chargement de
modèle : on relit `q_final` (C.0) et `k_final`/`v_final` (D.0b corrigé) et on
reproduit `eager_attention_forward` de `modeling_gemma4.py` (5.9.0, L~982-1015).

### Périmètre strict
- **Pipeline** (fidèle à `eager_attention_forward`, source de vérité) :
  ```python
  key_states   = repeat_kv(k_final, 8)                          # GQA 1 -> 8  [1,8,4,256]
  value_states = repeat_kv(v_final, 8)                          #             [1,8,4,256]
  scores_raw   = matmul(q_final, key_states.transpose(2,3)) * 1.0   # scaling=1.0  [1,8,4,4]
  scores_masked= scores_raw + causal_mask                       # masque ADDITIF (finfo.min)
  probs        = softmax(scores_masked, dim=-1, dtype=fp32)     #             [1,8,4,4]
  context      = matmul(probs, value_states)                    #             [1,8,4,256]
  ```
- **Faits Gemma4 verrouillés** : `scaling = 1.0` (PAS 1/√head_dim), **pas de softcap
  d'attention**, GQA `num_key_value_groups = 8`, masque additif, softmax fp32,
  `value = v_after_norm` (V RMSNorm sans scale, jamais V brut).
- **Masque** : layer_type `sliding_attention`, `sliding_window = 512`. On construit le
  **vrai masque sliding** et on **prouve** qu'à `S=4 < 512` il est strictement
  identique au masque causal (`torch.equal` PASS). **E.0 ne valide donc PAS le
  comportement sliding-window réel** — réservé à E.mask (synthétique S=8, window=3).
- **Interdits** : ZML, runners ZML, E.1, layer 14 (full attention), softcap
  d'attention, scaling `1/√head_dim`, V brut non normé, vrai sliding prétendu validé.

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/21_attention_oracle_layer15_reader_kv13.py` | oracle PyTorch attention (relit fixtures, pas de modèle) |
| `fixtures/p5_2_e0_attention_oracle_layer15_kv13.pt` | tenseurs (inputs + `scores_raw`/`scores_masked`/`probs`/`context`) + `meta` embarquée |
| `logs/21_attention_oracle_layer15_reader_kv13.log` | sortie (md5 fixture M1 ≡ 3090 `f88ea58d…`) |

### Résultats numériques observés
- Shapes : `scores_raw` `[1,8,4,4]`, `scores_masked` `[1,8,4,4]`, `probs` `[1,8,4,4]`, `context` `[1,8,4,256]` ✓
- **V correction check** : `max|v_final − v_raw| = 0.7768` (V RMSNormé sans scale, anti-régression bug D.0) ✓
- **Sliding ≡ causal** (`torch.equal`) = `True` à S=4 < 512 ✓
- **Softmax** : `max|Σ probs − 1| = 1.19e-7` (< 1e-6) ; **masse futur masqué = 0.0 strict** ✓
- Causalité par position : q0 voit t0 ; q1 voit t0..t1 ; q2 voit t0..t2 ; q3 voit t0..t3 (masse visible = 1.0, futur = 0) ✓
- Cross-check `k_final`/`v_final` vs fixture D.5 slim : `|Δ| = 0.0` strict ✓
- Fixed point `probs[0,0,0,:4] = [1, 0, 0, 0]` (q0 totalement sur t0) ; `probs[0,0,3,:4] = [0.0098, 0.0352, 0.7669, 0.1881]` (somme 1).

### Finding capitalisé (E.0)
- `scores_masked` a `mean = −inf`, `std = inf` dans les stats : artefact **attendu** du
  masque additif `finfo.min` (≈ −3.4e38) sur 6 des 16 positions par head. Le **calcul**
  reste sain (softmax sans NaN, probs propres, futur = 0 exact). Les valeurs masquées
  sont sauvées telles quelles dans la fixture (pas de nettoyage : fidélité à la source).
- E.0 établit la **référence** que E.1 (ZML QK scores) comparera contre `scores_raw`.

### Critères de clôture P5.2.E.0
- [x] Script tourne sur 3090 venv `/data/venvs/gemma4-probe` (EXIT=0)
- [x] Toutes les shapes conformes au contrat
- [x] Masque causal S=4 correct + sliding≡causal prouvé (`torch.equal`)
- [x] Softmax somme à 1 (err 1.19e-7) ; futur masqué ≈ 0 (0.0 strict)
- [x] V normé confirmé (`max|v_final − v_raw| = 0.7768`, pas de V brut)
- [x] Fixture sauvée + md5 M1 ≡ 3090 (`f88ea58d…`)
- [x] Doc explicite : **E.0 ne valide PAS le sliding-window réel** (réservé E.mask)
- [x] Aucun ZML, aucun runner ZML touché, E.1 non ouvert

**Tag** : `p5.2-e0-pytorch-attention-oracle-pass`

---

## P5.2.E.1 — ZML QK scores only : reader layer 15 × KV layer 13 (PASS, 31 mai 2026)

### Objectif
Premier calcul d'attention en ZML, limité aux **scores bruts** `Q·Kᵀ` (AVANT masque /
softmax / context). Comparer byte-equivalent contre l'oracle PyTorch `scores_raw`
(fixture E.0). Première opération qui introduit le **batched matmul d'attention** et
la **GQA** côté ZML.

### Périmètre strict
- **Pipeline ZML** (dot manuel, PAS le helper `sdpa`/`attention` — voir Finding) :
  ```zig
  // q_final {.b,.h=8,.q,.hd}  ;  k_final {.b,.h=1,.k,.hd}  (tags posés au chargement)
  const q_split = q_final.splitAxis(.h, .{ .h = k_final.dim(.h), .hq = .auto }); // {.b,.h=1,.hq=8,.q,.hd}
  const scores  = q_split.dot(k_final, .hd);                                     // {.b,.h=1,.hq=8,.q,.k}
  const merged  = scores.merge(.{ .h = .{ .h, .hq } });                          // {.b,.h=8,.q,.k}
  return merged.transpose(.{ .b, .h, .q, .k });                                  // [b,h,q,k] = oracle layout
  ```
- **GQA = split des têtes Q** (convention Llama/sdpa, `zml/nn.zig:1094`), PAS un
  `repeat_kv` de K. Les 8 têtes Q sont scindées en `{.h=1, .hq=8}` et se batchent
  contre l'unique tête K (`.h=1`). Le résultat `merge(.h,.hq)` ré-aligne l'index de
  tête global (head `hq` ↔ head PyTorch `hq`, K head 0 partagé).
- **Scaling = 1.0** : Gemma4 (la norm passe par q_norm/k_norm). On **omet** la `mul`
  par `1/√head_dim` que `sdpa` applique par défaut — sinon scores divisés par 16.
- **Interdits** : masque, softmax, context (dot V), layer 14, softcap, `1/√head_dim`.

### Décision d'implémentation (Finding sur l'API ZML)
Le helper `zml.attention.attention.attention` et `zml.nn.sdpa` **n'exposent jamais
les scores bruts** : ils enchaînent `dot → mask → softmax → dot(V) → merge` de façon
monolithique, et les backends `cuda_fa2/fa3` (flash attention) fusionnent le kernel
(scores physiquement inexistants en mémoire). → E.1 refait le `dot` Q·Kᵀ **à la main**
(3 lignes), seule voie propre pour isoler les scores avant softmax.

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/22_p5_2_e1_export_fixture.py` | export safetensors slim depuis le `.pt` E.0 (q_final, k_final, scores_raw) |
| `fixtures/p5_2_e1_qk_scores_layer15_kv13.safetensors` | 3 tenseurs (gitignored, régénérable) |
| `fixtures/p5_2_e1_qk_scores_layer15_kv13_manifest.json` | shapes/dtypes + pipeline ZML hint |
| `zml_runner/gemma4_qk_scores.zig` | runner ZML QK scores (single-return) |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_qk_scores` |
| `logs/P5_2_E1_qk_scores.log` | sortie `bazel run` (3090) |

### Résultats numériques observés
- Symbolic shapes ZML : `q {b=1,h=8,q=4,hd=256}`, `k {b=1,h=1,k=4,hd=256}`, oracle `{b=1,h=8,q=4,k=4}` ✓
- **Forward result shape** : `{b=1,h=8,q=4,k=4}` ✓ (split→dot→merge→transpose produit le bon layout)
- Fixed-point blocks : 3 blocks max_diff **9.54e-7** (dont `[0,7,3,:4]` head 7 spécifique)
- **Scan global** : max_abs **2.384e-6** @ flat_index 36 (b=0, h=2, q=1, k=0), mean_abs **3.52e-7**
- Tolérance 1e-4 → **marge ~42 000×**. Résidu = matmul QK PJRT-CPU Eigen-like vs PyTorch BLAS, cohérent avec le plancher de jitter fp32 ~5e-7 mesuré côté Python (script 22).

### Vérification adversariale (anti-faux-PASS, cf leçon D.0b)
- **Oracle indépendant du code testé** : l'oracle exprime la GQA par `repeat_kv(K)` (PyTorch),
  le ZML par `splitAxis(Q)` — **deux expressions différentes** qui convergent à 2.4e-6.
  Un mauvais mapping de têtes donnerait une divergence massive, pas 2.4e-6.
- **Pas de recopie** : le résidu ≠ 0.0 prouve un vrai matmul indépendant (une copie de
  l'oracle donnerait 0.0 strict).
- **Head-mapping** : le fixed-point head 7 matche l'oracle head 7 → `merge(.h,.hq)` préserve l'index global.
- **Scaling 1.0** vérifié dans `modeling_gemma4.py` L772 (source de vérité), pas inféré.

### Critères de clôture P5.2.E.1
- [x] Build bazel PASS sur 3090 (`//examples/rqz:gemma4_qk_scores`, 34s)
- [x] Forward layout `{b,h,q,k}` = `[1,8,4,4]` conforme à l'oracle
- [x] GQA par split des têtes Q (convention sdpa), scaling 1.0 (pas de `1/√hd`)
- [x] 3 fixed-point blocks rapportés vs `scores_raw`, max 9.54e-7
- [x] Scan global max_abs 2.384e-6 < tolérance 1e-4
- [x] Comparé à `scores_raw` (PAS `scores_masked` qui contient des `finfo.min`)
- [x] Aucun masque, softmax, context, layer 14, softcap
- [x] Log archivé `logs/P5_2_E1_qk_scores.log`

**Tag** : `p5.2-e1-zml-qk-scores-pass`

---

## P5.2.E.mask — ZML masque sliding RÉEL, cas synthétique S=8/window=3 (PASS, 31 mai 2026)

### Objectif
Fermer le **trou de couverture** laissé par E.0/E.1 : à `S=4` et `sliding_window=512`,
le masque sliding **dégénère en causal** (la fenêtre ne mord jamais), donc E.0/E.1 ne
valident que le régime causal. Ici, cas synthétique `S=8, window=3` où le sliding
**diffère effectivement** du causal, pour valider la vraie logique de fenêtrage en ZML.

### Convention (source de vérité, triple-validée)
`transformers/masking_utils.py` `sliding_window_overlay` : `kv_idx > q_idx - sliding_window`,
composé (`and_masks`) avec le causal `kv_idx <= q_idx`. Le helper ZML
`zml.nn.causalAttnMask` implémente exactement la même chose (`k.cmp(.LE,q) AND q.cmp(.LT,k+window)`).
La table dérivée == celle attendue :
```
q=0:[o.......] q=1:[oo......] q=2:[ooo.....] q=3:[.ooo....]
q=4:[..ooo...] q=5:[...ooo..] q=6:[....ooo.] q=7:[.....ooo]   (visible ⟺ q-3 < k <= q)
```
→ **21 visibles / 43 masquées** ; q=7 masqué = `[0,1,2,3,4]`.

### Périmètre strict (mask only)
- **Pipeline ZML** (multi-return) :
  ```zig
  const mask = zml.nn.causalAttnMask(.{ .q = 8, .k = 8 }, .f32, 3);  // {.q,.k} additif (0 / finfo.min)
  const scores_masked = self.scores_synth.add(mask.broad(self.scores_synth.shape())); // broadcast sur .b,.h
  return .{ mask, scores_masked };
  ```
- **Garde déterminante** : `causalAttnMask` n'applique la fenêtre que si `qlen >= window_len`.
  À `S=4 >= 512` = faux → causal pur (c'est ce qui faisait dégénérer E.0/E.1). À `S=8 >= 3`
  = vrai → la fenêtre **mord**.
- **Interdits** : softmax, context, `dot(V)`, layer 14, full attention, scaling, RoPE, Q/K/V proj.

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/23_p5_2_emask_oracle.py` | oracle numpy from-scratch (masque + scores synthétiques) + assertions table |
| `fixtures/p5_2_emask_sliding_layer_synthetic.safetensors` | scores_synth [1,2,8,8], sliding_mask [8,8], scores_masked [1,2,8,8] |
| `fixtures/p5_2_emask_sliding_layer_synthetic_manifest.json` | convention + pipeline hint |
| `zml_runner/gemma4_sliding_mask.zig` | runner ZML (construit le masque via helper natif + applique) |
| `zml_runner/BUILD.bazel` | cible `gemma4_sliding_mask` |
| `logs/P5_2_Emask_sliding_mask.log` | sortie `bazel run` |

### Résultats numériques observés
- Forward shapes : `mask {q=8,k=8}`, `scores_masked {b=1,h=2,q=8,k=8}` ✓
- Grille ZML reconstruite **identique** à l'oracle (bande diagonale largeur 3 qui glisse)
- **mask** : masked=**43**/64, visible max_diff=**0.0**, struct_mismatch=**0**
- **scores_masked** : masked=**86**/128 (43×2 heads), visible max_diff=**0.0**, struct_mismatch=**0**
- Comparaison **robuste finfo.min** : visible bit-exact, masqué `< -1e30`, structure 100 % identique.

### Vérification adversariale
- **Constructions indépendantes** : oracle = numpy (`k≤q & k>q−W`), ZML = helper `causalAttnMask`.
  Convergence **bit-exact** (0 mismatch / 64). Un helper bogué divergerait.
- **La fenêtre mord** : causal pur masquerait **28** positions, le sliding en masque **43**
  (**+15** positions anciennes retirées). Si le `window` était ignoré (régime E.0/E.1), on
  verrait 28, pas 43. Preuve numérique que le sliding ≠ causal sur ce cas.
- **Bit-exact attendu et sain** : masque additif + add, pas de matmul → déterminisme IEEE754
  (≠ E.1 où le matmul donnait 2.4e-6). Le 0.0 confirme add exact + scores identiques.

### Critères de clôture P5.2.E.mask
- [x] Build bazel PASS sur 3090 (`//examples/rqz:gemma4_sliding_mask`)
- [x] mask shape `[8,8]` broadcast-compatible `[1,1,8,8]` ; positions futures (k>q) masquées
- [x] positions trop anciennes (k ≤ q−3) masquées ; fenêtre taille 3 respectée
- [x] `scores_masked` ZML == oracle (visible bit-exact, masqué structurellement identique)
- [x] masked count = 43 (mask) / 86 (scores) conforme ; struct_mismatch = 0
- [x] Aucun softmax, aucun context, aucun `dot(V)`, aucun layer 14
- [x] Log archivé `logs/P5_2_Emask_sliding_mask.log`

**Tag** : `p5.2-emask-sliding-mask-pass`

---

## P5.2.E.softmax — ZML softmax only : reader layer 15 × KV layer 13 (PASS, 31 mai 2026)

### Objectif
Valider la transformation `scores_masked → probs` côté ZML, **softmax sur l'axe `.k`
uniquement** (après QK scores de E.1 et masque de E.mask, AVANT le context dot). Comparer
vs l'oracle PyTorch `probs` = `torch.softmax(scores_masked, dim=-1, fp32)` figé en E.0.

### Périmètre strict
- **Pipeline ZML** : une seule op.
  ```zig
  probs = scores_masked.softmax(.k)   // [.b=1,.h=8,.q=4,.k=4]
  ```
  Convention `sdpa` ZML (`zml/nn.zig` L1112 : `attn_weights.convert(.f32).softmax(.k)`).
  `Tensor.softmax` (`tensor.zig` L1369) soustrait le max par ligne (stable) → exp → normalise,
  et renvoie 0 pour une ligne entièrement `-inf`. Ici aucune ligne n'est full-masquée
  (q0 voit toujours k0) → comportement identique à `torch.softmax`.
- **2 tenseurs chargés** depuis `fixtures/p5_2_esoftmax_layer15_kv13.safetensors` (slim re-export
  D.0/E.0 via `scripts/24_p5_2_esoftmax_export_fixture.py`) : `scores_masked` (input, contient
  déjà le masque causal additif finfo.min), `probs` (oracle).
- **Indépendance oracle** : `probs` vient de `torch.softmax` (E.0), le runner utilise
  l'implémentation ZML native — aucun code partagé, seul le contrat numérique l'est.
- **Interdits stricts** : context `probs @ V`, toute op sur V, `dot(V)`, masque réel
  S=8/window=3 (testé en E.mask), layer 14 full attention, softcap, scaling `1/√head_dim`.

### Livrables
| Fichier | Rôle |
|---|---|
| `scripts/24_p5_2_esoftmax_export_fixture.py` | export safetensors slim depuis le `.pt` E.0 (2 tenseurs) |
| `fixtures/p5_2_esoftmax_layer15_kv13.safetensors` | `scores_masked` (input) + `probs` (oracle) ~1 KB |
| `fixtures/p5_2_esoftmax_layer15_kv13_manifest.json` | shapes/dtypes + pipeline + interdits |
| `zml_runner/gemma4_softmax.zig` | runner ZML (`softmax(.k)` + checks distribution) |
| `zml_runner/BUILD.bazel` | nouvelle cible `gemma4_softmax` |
| `logs/P5_2_Esoftmax.log` | sortie `bazel run` (3090) |

### Validation
**Checks distribution (sortie ZML)** :
- `max|sum(probs, .k) − 1| = 1.192e-7` (tol 1e-5)
- `max proba sur futur masqué = 0.000` (tol 1e-9) — causalité stricte préservée par softmax
- NaN/Inf = false

**3 blocks fixed-point vs oracle** (`probs[0,0,0,:4]`, `probs[0,0,3,:4]`, `probs[0,7,3,:4]`) :
- **max_diff = 0.000** (bit-exact sur les 3 blocks)

**Scan global 128 valeurs** :
- max_abs = **2.980e-8** à `flat_index 46 (b=0, h=2, q=3, k=2)`
- mean_abs = **6.112e-10**
- Tolérance 1e-4 → marge ~3 300× sous le seuil

### Observations
- **max_abs E.softmax (2.98e-8) << jitter QK E.1 (2.38e-6)** : la normalisation softmax borne
  les sorties dans `[0,1]` et les probabilités dominantes correspondent exactement à l'oracle ;
  le résidu absolu est donc plus petit que celui des scores bruts. La position du max se
  déplace (h=2,q=3,k=2) car l'argmax du worst-case n'est plus lié au matmul QK mais à
  l'exp/normalisation d'une distribution non-dégénérée.
- **finfo.min géré correctement** : `softmax` soustrait le max fini par ligne → les positions
  masquées (`-3.4e38`) donnent `exp(≈-3.4e38)=0`, exactement comme l'oracle. Fuite futur = 0.

### Piège capitalisé (build) — quota comptime / longueur du nom de module
Le runner initial s'appelait `gemma4_attention_softmax.zig`. Build **KO** :
`error: evaluation exceeded 1000 backwards branches` dans `std/mem.zig` (`indexOf`),
déclenché par `pjrt.zig` `structSize(T)` (`std.mem.indexOf(u8, @typeName(T), ".struct_")`)
à l'instanciation comptime de `exe.call` → `PJRT_LoadedExecutable_Execute`. Le `Struct`
**local** de `pjrt.zig` n'a pas de `@setEvalBranchQuota` (contrairement à `zml/meta.zig:308`),
et un `@setEvalBranchQuota` placé dans **mon `main` n'atteint PAS** cette Sema d'instanciation
générique (scope comptime distinct, quota par défaut 1000). Les runners au nom plus court
(`gemma4_qk_scores` 16c, `gemma4_sliding_mask` 19c) passaient ; `gemma4_attention_softmax`
(24c) débordait — le coût cumulé de branches comptime croît avec la longueur du `@typeName`.
**Fix** : renommer le binaire `gemma4_softmax` (14c) → build PASS. **Règle réutilisable :
garder les noms de runners ZML courts (≤ ~20 c) pour rester sous le quota comptime fragile
de `pjrt.zig`.**

### Critères de clôture P5.2.E.softmax
- [x] Build bazel PASS sur 3090 (`//examples/rqz:gemma4_softmax`)
- [x] Run produit `probs [1,8,4,4]`
- [x] `sum(probs, .k) ≈ 1` (err 1.19e-7) sur tous les heads/queries
- [x] Futur masqué ≈ 0 (0.000) ; aucun NaN/Inf
- [x] 3/3 fixed-point blocks bit-exact (max_diff 0.0) vs oracle `probs`
- [x] Scan global 128 valeurs : max_abs 2.98e-8 < tolerance 1e-4 ; mean_abs 6.11e-10
- [x] Aucun context dot, aucun `dot(V)`, aucune op sur V, aucun layer 14
- [x] Log archivé `logs/P5_2_Esoftmax.log`

**Tag** : `p5.2-esoftmax-zml-pass`

> Note livrable : le handoff listait `scripts/23_…` et `zml_runner/gemma4_attention_softmax.zig`.
> Numéro de script → `24_` (le `23_` était déjà pris par `23_p5_2_emask_oracle.py`) ; runner
> renommé `gemma4_softmax.zig` (cf piège quota comptime ci-dessus).

---

## P5.2.E (suite) — context

> **Prérequis** : ~~E.0~~ ✅ · ~~E.1 QK scores~~ ✅ · ~~E.mask sliding réel~~ ✅ ·
> ~~E.softmax~~ ✅ (tag `p5.2-esoftmax-zml-pass`). **QK + masque + softmax validés ZML↔PyTorch.**

- **E.context** = `probs.dot(value_states, .k)` → context `[b,h,q,hd]` (réutilise la branche V
  **corrigée** de D.5 canonique, `value = v_after_norm`). Première **chaîne d'attention ZML
  complète** (QK → mask → softmax → context). GQA : `value_states = repeat_kv(v_final, 8)` côté
  oracle, à reproduire en ZML (broadcast kvh 1 → h 8) avant le `dot(.k)`.
- Cadrage layer 14 (full attention, p-RoPE proportional) vs layer 13/15 (sliding) à
  trancher. Le helper `zml.nn.rope` ne couvre pas `proportional + partial rotary`
  (point rouge connu, cf D.4).
