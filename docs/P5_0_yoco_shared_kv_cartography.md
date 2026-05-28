# P5.0 — Gemma 4 E2B YOCO / Shared KV cartography

**Status** : DRAFT (étapes 1-2 faites le 28 mai 2026 ; étapes 3+ pour prochaine session)
**Scope** : architecture only, no ZML code.

## 1. Inputs

- Model : `google/gemma-4-E2B-it`
- Prior state : P4.4.2 CLOSED / PASS (dernier commit `c40648c`, tag `gate/P4.4.2-gate-J-pass`)
- Objectif : comprendre `num_kv_shared_layers=20` avant tout code ZML couche-attention.

## 2. Config facts (source `logs/09_yoco_config_map.log`)

```
transformers config class : Gemma4Config
text_config class         : Gemma4TextConfig
num_hidden_layers         : 35
num_kv_shared_layers      : 20
sliding_window            : 512
num_attention_heads       : 8
num_key_value_heads       : 1
head_dim                  : 256
hidden_size               : 1536
hidden_size_per_layer_input: 256
final_logit_softcapping   : 30.0
```

Observations :
- GQA ratio 8:1 (8 query heads, 1 KV head par couche owner).
- `head_dim = 256` cohérent avec PLE dim D (réutilisation de la même valeur).
- `sliding_window = 512` tokens.

## 3. Attention layer pattern (source 09)

```
layer_types length        : 35
|full|                    : 7
|sliding|                 : 28

full_attention layers     : [4, 9, 14, 19, 24, 29, 34]
sliding_attention layers  : [0,1,2,3, 5,6,7,8, 10,11,12,13,
                             15,16,17,18, 20,21,22,23,
                             25,26,27,28, 30,31,32,33]
```

Motif : **4 sliding + 1 full**, répété 7 fois. Les couches full sont aux positions `5n + 4` (n = 0..6) → 4, 9, 14, 19, 24, 29, 34.

Cohérent avec ce qu'on avait noté dans `memory/project_gemma4_zml_probe.md`.

## 4. Weight ownership map (source `logs/10_yoco_weight_map.log`)

**Finding inattendu** — Les 35 couches possèdent toutes leurs propres poids attention complets :

```
Layers WITH own k_proj+v_proj+k_norm  : [0..34]   count = 35
Layers WITHOUT own KV (YOCO consumers): []        count = 0

Cross-check with config:
  Expected num_kv_shared_layers = 20
  Observed YOCO consumers      = 0
  Match expected = NO (investigate)
```

Total 2011 weight keys. Chaque couche `model.layers.<i>.self_attn.{q,k,v,o}_proj.weight` et `{q,k}_norm.weight` est présent pour `i ∈ [0, 34]`.

**Conséquence majeure** : pour gemma-4-E2B-it, le partage KV n'est **pas** matérialisé par une absence de poids (contrairement à ce que suggérait l'issue vLLM-metal référencée — probablement spécifique à E4B ou autre régime). Donc :

> Le sharing KV chez Gemma 4 E2B-it est **runtime-only** : les 20 couches "consumers" possèdent leurs poids k/v mais les **ignorent à l'exécution** et lisent les K/V d'une couche source.

C'est un point d'inflexion architectural. Il faut maintenant lire le code Transformers pour comprendre :
- quelle est la couche source pour chaque couche consumer ?
- les poids k/v "inutilisés" sont-ils strictement morts ou ont-ils un usage subtil (debug, fallback, training-only) ?
- est-ce que `num_kv_shared_layers = 20` désigne les 20 *dernières* couches comme consumers (4 sliding + 1 full × 3 ≈ 15-ish, donc pas exactement 20) ou une autre coupe ?

## 5. Transformers source trace

**À faire prochaine session.** Procédure attendue :

```python
import inspect
import transformers.models.gemma4.modeling_gemma4 as mg
src = inspect.getsource(mg.Gemma4TextModel)
src += inspect.getsource(mg.Gemma4DecoderLayer)
src += inspect.getsource(mg.Gemma4Attention)
# grep : shared_kv | num_kv | layer_types | full_attention | sliding_attention
#        past_key | cache | k_proj | v_proj | k_norm | shared_kv_states
```

Tracer :
- où `shared_kv_states` est initialisé
- où il est écrit (probablement à la sortie d'une couche "source")
- où il est lu (par les couches "consumers")
- comment `layer_idx` détermine qui est source / consumer
- interaction avec `past_key_values` (cache decode)
- interaction avec `layer_types` (full vs sliding)

## 6. vLLM source trace

**À faire prochaine session.** Cibles :
- `Gemma4SelfDecoderLayers` (wrapper première moitié YOCO).
- `Gemma4Model` (entrée).
- `_run_decoder_layers` (split YOCO en pratique).
- `kv_sharing_fast_prefill` (drapeau).
- Où se positionne le PLE par rapport au split YOCO (probablement avant).

Si vLLM n'est pas installé, lecture publique seulement — ne pas installer pour cette phase.

## 7. Prefill lifecycle hypothesis

**À renseigner après étape 5.** Esquisse provisoire :

1. Tokenisation → ids
2. Embeddings principal + PLE (P4.4.2, validé) → `inputs_embeds` + `per_layer_inputs`
3. Décodage couche par couche :
   - Couches "source" (à identifier) calculent et écrivent leurs K/V dans un buffer partagé indexé par couche cible.
   - Couches "consumer" sautent leur propre calcul K/V et lisent depuis le buffer.
4. Comportement full vs sliding dans chaque couche (truncation cache, masque).

## 8. Decode lifecycle hypothesis

**À renseigner après étape 5.** À tracer :
- Mise à jour cache token par token côté source.
- Diffusion vers consumers (pointeur unique vs copie ?).
- Sliding window truncation côté sliding layers.
- Position token dans le cache.

## 9. ZML implications (provisoire)

Sans code encore. Contraintes anticipées :

- Le cache K/V ne peut pas être un simple `[L, B, S, H_kv, D]` uniforme : les positions "source" doivent être écrites, les positions "consumer" doivent pointer vers une source.
- Deux régimes de cache par couche : `full_attention` (cache croissant jusqu'à fin) vs `sliding_attention` (cache borné à 512).
- `shared_kv_states` est probablement une struct ZML dédiée (pas une simple paire `k, v`).
- Standard de comparaison (réutilisable, validé en Gate J) :
  ```zig
  var ref_slice = try buffers.<reference_tensor>.toSliceAlloc(allocator, io);
  defer ref_slice.free(allocator);
  ```
- Les 3 pièges ZML capitalisés en P4.4.2 (reshape perd tags, mul/add ne broadcast pas, pattern Llama vs Qwen) restent valables.

## 10. Open questions / BLOCKERS

1. **(BLOCKER P5.0)** Si tous les poids K/V sont présents, comment Transformers décide-t-il quelles couches sont source vs consumer ? `num_kv_shared_layers=20` désigne-t-il les 20 dernières, les 20 sliding, ou un mapping plus complexe ?
2. Est-ce que le partage est intra-bloc (une couche source par bloc 5 couches) ou inter-bloc (les 15 premières couches partagent vers les 20 dernières) ?
3. Le PLE injection (validé P4.4.2) se fait à chaque couche : est-elle indépendante du sharing K/V ou y a-t-il un couplage ?
4. Quel est le comportement attendu si on charge le modèle "naïvement" et qu'on exécute les 35 couches avec leurs propres K/V (i.e. en ignorant le sharing) ? Y a-t-il un test diff ?

## Critères de clôture P5.0 (rappel)

P5.0 est fermé quand on peut répondre à ces 9 questions :

1. Quelles couches full_attention ? **OK : [4, 9, 14, 19, 24, 29, 34]**
2. Quelles couches sliding_attention ? **OK : les 28 restantes**
3. Quelles couches possèdent leurs propres K/V ? **OK : les 35 (toutes)**
4. Quelles couches partagent K/V ? **À tracer dans Transformers source (runtime-only).**
5. Quelle couche écrit dans `shared_kv_states` ? **À tracer.**
6. Quelle couche lit `shared_kv_states` ? **À tracer.**
7. Comment `past_key_values` interagit avec `shared_kv_states` ? **À tracer.**
8. Que change prefill vs decode ? **À tracer.**
9. Quel layout de cache faudra-t-il prévoir en ZML ? **À écrire après 4-8.**

**3/9 répondues. Prochaine session : étapes 3-4 (source Transformers + vLLM) → fermer 4-8.**
