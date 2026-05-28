# P5.1 — Gemma 4 E2B YOCO policy table

**Status** : PASS (28 mai 2026). Table statique 35 entrées générée + validée + sérialisée.
**Scope** : table d'accès cache K/V uniquement. Pas d'attention, pas de RoPE, pas de decode loop.

## 1. Contrat

Pour chaque layer `i ∈ [0, num_hidden_layers)`, produire un quadruplet :

```
(layer_idx, layer_type, is_reader, target_kv_layer)
```

Sémantique :
- `is_reader = (num_kv_shared_layers > 0) and (i >= first_kv_shared_layer_idx)`, avec
  `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers`.
- Si non-reader : `target_kv_layer = i` (own slot).
- Si reader : `target_kv_layer = last_index j < first_kv_shared_layer_idx such that layer_types[j] == layer_types[i]`.

C'est la formule littérale du code Transformers `Gemma4TextAttention.__init__` (`modeling_gemma4.py:777-782`) et du code vLLM `Gemma4Attention.__init__` (`gemma4.py:469-489`). Strictement identique dans les deux runtimes, validé en P5.0 § 6.2.

Pour gemma-4-E2B-it (`num_hidden_layers=35`, `num_kv_shared_layers=20`), `first_kv_shared_layer_idx = 15`. Donc :
- Layers 0..14 sont producers (`is_reader = False`, `target = self`).
- Layers 15..34 sont readers (`is_reader = True`, `target ∈ {13, 14}`).

## 2. Livrables

| Fichier | Rôle |
|---|---|
| `scripts/13_yoco_policy_table.py` | Génération + validation à partir d'AutoConfig (pas de model load, pas de poids) |
| `fixtures/yoco_policy_table.json` | Table sérialisée : config + summary + 35 entrées |
| `logs/13_yoco_policy_table.log` | Stdout du run (table pretty-printed + summary + `validation: PASS`) |
| `docs/P5_1_yoco_policy_table.md` | Ce document |

## 3. Validation (assertions baked-in `validate_e2b`)

Toutes ces assertions PASS sur `google/gemma-4-E2B-it` :

- `num_producers == 15`
- `num_readers == 20`
- `first_kv_shared_layer_idx == 15`
- `writers_per_layer_type == {"full_attention": 14, "sliding_attention": 13}`
- `readers target ⊂ {13, 14}` (set égal à `{13, 14}`, pas un sur-ensemble)
- Pour chaque producer `i` : `target == i` (identity)
- Pour chaque reader full_attention : `target == 14`
- Pour chaque reader sliding_attention : `target == 13`

## 4. Distribution observée

```
Producers (15) :
  full_attention   = [4, 9, 14]                                    (3)
  sliding_attention = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]     (12)

Readers (20) :
  full_attention   = [19, 24, 29, 34]                              (4)
  sliding_attention = [15-18, 20-23, 25-28, 30-33]                 (16)

Totaux globaux (cf. P5.0 § 3) :
  full_attention   = 3 + 4 = 7
  sliding_attention = 12 + 16 = 28
  Sum                = 35 ✓

Writers désignés (les 2 derniers producers de chaque type, dans `prev_layers[:15]`) :
  layer 14 (full)    → cache "full_attention"
  layer 13 (sliding) → cache "sliding_attention"
```

## 5. Format `fixtures/yoco_policy_table.json`

```json
{
  "repo": "google/gemma-4-E2B-it",
  "source": "P5.1 policy table from AutoConfig (no model load)",
  "spec_refs": [
    "transformers/models/gemma4/modeling_gemma4.py L777-L782 (Gemma4TextAttention.__init__)",
    "vllm/model_executor/models/gemma4.py L469-L489 (Gemma4Attention.__init__)"
  ],
  "config": {
    "num_hidden_layers": 35,
    "num_kv_shared_layers": 20,
    "sliding_window": 512,
    "first_kv_shared_layer_idx": 15
  },
  "summary": { ... },
  "table": [
    {"layer_idx": 0, "layer_type": "sliding_attention", "is_reader": false, "target_kv_layer": 0},
    ...
    {"layer_idx": 34, "layer_type": "full_attention", "is_reader": true, "target_kv_layer": 14}
  ]
}
```

## 6. Pattern de consommation (futur ZML, non implémenté)

```zig
// À l'init de l'attention runtime, charger la table une fois :
const policy = try loadPolicyTable("fixtures/yoco_policy_table.json", allocator);
// policy.table[i].target_kv_layer
// policy.table[i].is_reader
// policy.table[i].layer_type

// Per-layer attention forward :
fn attentionForward(layer_idx: usize, ...) !Tensor {
    const entry = policy.table[layer_idx];
    if (entry.is_reader) {
        // skip qkv_proj.K, qkv_proj.V entirely
        // read K, V from kv_cache.producer_kv[entry.target_kv_layer]
        // apply RoPE on Q only
    } else {
        // standard q_proj + k_proj + v_proj
        // q_norm, k_norm, v_norm
        // RoPE on Q and K
        // write K, V into kv_cache.producer_kv[layer_idx]
    }
    // attention compute uses per_layer_sliding_window when layer_type == sliding_attention
    // sliding window applied via mask, not by truncating the cache
}
```

Aucune allocation runtime supplémentaire : la table est immuable, calculée à l'init depuis `layer_types`. Le cache K/V réel = 15 slots seulement (cf. § 6.6 du P5.0).

## 7. Hors scope (rappel)

- Pas d'attention compute.
- Pas de RoPE.
- Pas de RMSNorm.
- Pas de `qkv_proj`.
- Pas de decode loop.
- Pas de fast prefill (option vLLM, hors scope).

P5.2+ implémentera l'attention forward effective en ZML en consommant cette table.

## 8. Reproductibilité

```bash
ssh user@gpu-host \
  'source /data/venvs/gemma4-probe/bin/activate && \
   cd /data/gemma4-zml-probe && \
   python scripts/13_yoco_policy_table.py'
```

Sortie : `fixtures/yoco_policy_table.json` + `logs/13_yoco_policy_table.log`. Idempotent (la config est lue depuis cache HF).

## 9. Critères de clôture P5.1

- [x] Script `13_yoco_policy_table.py` lit la config sans charger les poids.
- [x] La logique reproduit verbatim Transformers + vLLM.
- [x] Validation `validate_e2b` PASS.
- [x] Fixture JSON sérialisée.
- [x] Doc design présent.

**P5.1 prêt à fermer.**
