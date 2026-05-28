# P5.0 — Gemma 4 E2B YOCO / Shared KV cartography

**Status** : DRAFT — étapes 1-4 faites le 28 mai 2026 ; **P5.0 = 9/9 questions répondues**, prêt à fermer.
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

**Nota bene (28 mai PM, après étape 3)** — la cartographie ci-dessus a lu le **safetensors brut** (35 clés `k_proj/v_proj/k_norm/v_norm` présentes). Côté Python *runtime*, c'est différent : `Gemma4TextAttention.__init__` (cf § 5.3) n'instancie PAS ces modules pour les shared layers (`if not self.is_kv_shared_layer:`), et `Gemma4TextModel.__init__` les drop du checkpoint via `_keys_to_ignore_on_load_unexpected`. Donc :
- **Disque** : 35 jeux de poids K/V (safetensors contient tout — Google a probablement entraîné les K/V pour les 35 puis a verrouillé le sharing au design).
- **RAM Python** : 15 jeux de poids K/V (layers 0..14), les 20 layers 15..34 sont nues côté K/V/k_norm/v_norm.
- **GPU forward** : 15 couches produisent K/V, 2 d'entre elles (les dernières non-shared de chaque `layer_type`) stockent leurs K/V finaux dans `shared_kv_states[layer_type]`, les 20 couches partagées lisent ce buffer.

Le sharing est **runtime-only au sens où aucune fusion de poids n'est faite** (les poids "extra" du safetensors sont silencieusement ignorés). L'hypothèse "tous les 35 ont leurs poids K/V" est vraie **sur disque** et fausse **dans le module Python instancié**.

## 5. Transformers source trace

### 5.1 Source inspected
- Transformers version : **5.9.0**
- File : `/data/venvs/gemma4-probe/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py`
- Logs : `logs/11_modeling_gemma4_source_trace.txt` (1157 lignes, 7 classes + `eager_attention_forward` + `Gemma4TextConfig`), `logs/11_modeling_gemma4_grep.txt` (96 hits), `logs/11_modeling_gemma4_all_relevant_blocks.txt` (1400 lignes, fenêtres ±10/+25 autour de chaque terme), `logs/11_modeling_gemma4_relevant_grep.txt` (117 hits).
- Classes réelles : `Gemma4TextDecoderLayer` / `Gemma4TextAttention` (préfixe `Text`, pas `Gemma4DecoderLayer` / `Gemma4Attention` qui n'existent pas).

### 5.2 Where `shared_kv_states` appears

```
134:        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())          # init   (Gemma4TextModel.forward)
143:                shared_kv_states=shared_kv_states,                             # plumb  (decoder_layer call)
156:            shared_kv_states=shared_kv_states if kwargs.get("return_shared_kv_states", False) else None,  # ret
503:            shared_kv_states=outputs.shared_kv_states,                         # plumb  (Gemma4Model)
652:            shared_kv_states=outputs.shared_kv_states,                         # plumb  (Gemma4ForCausalLM)
696:        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,  # sig (DecoderLayer)
710:            shared_kv_states=shared_kv_states,                                 # plumb  (self_attn call)
812:        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],        # sig (Attention)
830:            key_states, value_states = shared_kv_states[self.layer_type]       # READ
848:            shared_kv_states[self.layer_type] = key_states, value_states       # WRITE
```

Gating connexe sur `is_kv_shared_layer` et `store_full_length_kv` :

```
57:            if layer.self_attn.is_kv_shared_layer:                              # drop ckpt keys (TextModel.__init__)
58:                self._keys_to_ignore_on_load_unexpected.extend(
59:                    [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")]
60:                )
777:        first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
778:        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
779:        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
780-782:    self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
790:        if not self.is_kv_shared_layer:                                         # k/v/k_norm/v_norm NOT instantiated
829:        if self.is_kv_shared_layer:                                             # read branch
845:        if past_key_values is not None and not self.is_kv_shared_layer:        # cache update only for producers
847:        if self.store_full_length_kv:                                          # designated writer
```

### 5.3 Initialization
- **Initialized in** : `Gemma4TextModel.forward`, ligne 134.
- **Initial value** : `UserDict()` (vide ; commentaire explicite — `dict` natif casse FSDP2 lors du `_apply_to_tensors`, d'où `UserDict`).
- **Réutilisation** : si `kwargs["shared_kv_states"]` est fourni (chemin externe, ex. génération multi-step), il est récupéré ; sinon `UserDict()` neuf à chaque forward.
- **Shape / structure visible** : signature DecoderLayer ligne 696 et Attention ligne 812 → `dict[str, tuple[torch.Tensor, torch.Tensor]]`. Clés possibles = valeurs de `config.layer_types`. Pour E2B = `{"full_attention", "sliding_attention"}` → max 2 entrées par forward. Chaque entrée stocke `(K, V)` avec shapes `[batch, n_kv_heads, seq_total, head_dim]` (post-RoPE, post-RMSNorm, post-cache-update).

### 5.4 Write path
- **Written by** : `Gemma4TextAttention.forward`, ligne 848.
- **Condition** : `self.store_full_length_kv == True`. Calculée en `__init__` (ligne 780-782) :
  ```
  store_full_length_kv = (not is_kv_shared_layer) and (layer_idx == "dernière occurrence de ce layer_type dans prev_layers")
  ```
  où `prev_layers = config.layer_types[:first_kv_shared_layer_idx]`.
- **Layer relation** (E2B : `first_kv_shared_layer_idx = 35 - 20 = 15`, `prev_layers = layer_types[0..14]`) :
  - Pour layer 14 (`full_attention`) : `prev_layers[::-1].index("full_attention") = 0` → store at `15 - 1 - 0 = 14` ✓
  - Pour layer 13 (`sliding_attention`) : `prev_layers[::-1].index("sliding_attention") = 1` → store at `15 - 1 - 1 = 13` ✓
  - **Donc 2 writers exactement** : **layer 14 (full)** et **layer 13 (sliding)**.
- **Quand l'écriture a lieu** : *après* `past_key_values.update(...)` (ligne 845-846). Les K/V écrits dans `shared_kv_states` sont donc le **cache complet K/V après concat** (pas seulement les K/V du token courant).
- **Prefill/decode distinction** : aucune. Même chemin dans les deux régimes. À chaque forward (prefill T tokens OU decode 1 token), les writers recalculent leur projection K/V, update le cache standard `past_key_values`, puis stockent le résultat dans `shared_kv_states[layer_type]`.

### 5.5 Read path
- **Read by** : `Gemma4TextAttention.forward`, ligne 830.
- **Condition** : `self.is_kv_shared_layer == True`. Le K/V est lu **avant** RoPE/RMSNorm/transpose — donc le tenseur stocké par le writer a déjà subi ces transformations (cf. lignes 838-843 dans la branche producer).
- **Layer relation** :
  - `is_kv_shared_layer = (layer_idx >= 15)` pour E2B → **layers 15..34** lisent (20 layers, cohérent avec `num_kv_shared_layers=20`).
  - Chaque reader lit `shared_kv_states[self.layer_type]`. Donc tous les readers `full_attention` (layers 19, 24, 29, 34) lisent **le même tenseur** écrit par layer 14. Tous les readers `sliding_attention` (15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33) lisent **le même tenseur** écrit par layer 13.
- **Device handling** : `key_states.to(query_states.device)` (ligne 832-833) — supporte writer et reader sur devices différents.
- **Skip cache update** : les readers **n'appellent pas** `past_key_values.update()` (gate ligne 845 `not self.is_kv_shared_layer`). Le cache standard ne contient donc que les K/V des 15 producer layers.
- **Prefill/decode distinction** : aucune côté reader. La forme du tenseur lu reflète l'état du cache writer **après l'étape courante** (full pour prefill, +1 token pour decode).

### 5.6 Relation to `num_kv_shared_layers=20`
- Définit le split : `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers` = `35 - 20 = 15`.
- **Indique le nombre de layers consommatrices** (par le bas du stack vers le haut) : layers `[first_kv_shared_layer_idx, num_hidden_layers)` = **layers 15..34**.
- Donc `num_kv_shared_layers=20` ≡ "les 20 *dernières* couches sont consommatrices". Ce n'est PAS un mapping intra-bloc, c'est un préfixe inversé : **bas de stack = producers, haut de stack = consumers**.
- Layers `0..14` (15 producers) : possèdent leur K/V module, **2 d'entre elles** sont les writers de `shared_kv_states` (les dernières par `layer_type`).
- Layers `15..34` (20 consumers) : aucun K/V module instancié, lecture exclusive via `shared_kv_states[layer_type]`.

### 5.7 Relation to `layer_types`
- `shared_kv_states` est un **dict keyé par `layer_type`** (pas par `layer_idx`). Donc une seule paire `(K,V)` par type, partagée entre tous les consumers du même type.
- Sur E2B, `layer_types` a deux valeurs uniques (`"full_attention"` / `"sliding_attention"`). Le dict a donc au plus 2 entrées.
- Les rotary embeddings sont aussi indexés par `layer_type` (cf. `Gemma4TextRotaryEmbedding.forward` ligne 964-978) → les `cos/sin` que voit le writer sont **ceux de son propre layer_type**, et le reader applique RoPE sur les Q seuls (les K/V lus de `shared_kv_states` ont déjà la RoPE du producer baked-in).
- **Subtilité importante** : layer 14 (writer full) produit ses K avec `rope_theta=1e6, partial_rotary_factor=0.25`. Layer 19 (reader full) verra ces K/V — pas de re-RoPE côté reader. Idem layer 13 (writer sliding) avec `rope_theta=1e4` pour ses readers sliding.

### 5.8 Relation to `past_key_values`
- Les deux mécanismes coexistent mais **ne croisent pas** :
  - `past_key_values` (le `DynamicCache` standard) : update uniquement par les **15 producer layers** (gate ligne 845). 15 slots dans le cache, indexés par `self.layer_idx`.
  - `shared_kv_states` : update par les **2 writers** seulement, indexé par `layer_type`.
- Commentaire-clé lignes 826-828 (verbatim) : *"We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache once we are past the sliding window — so we always use `shared_kv_states` instead, even when past_key_values is not None."*
- D'où **le nom `store_full_length_kv`** : pour les sliding writers, le cache `past_key_values` peut être tronqué à 512 tokens, mais `shared_kv_states` garde la **longueur full** (le tenseur stocké ligne 848 est `key_states, value_states` post-update, qui pour sliding n'est PAS encore appliqué de fenêtre — la fenêtre est appliquée via le masque `sliding_attention` dans `eager_attention_forward`).
- Conséquence : `shared_kv_states["sliding_attention"]` peut être **plus long** que les K/V réellement utilisés par chaque reader sliding individuel (qui restreindra par masque).
- Côté consumer décode (token T+1) :
  1. Producer layers 0..14 reçoivent le nouveau token, calculent K/V, update `past_key_values`, et **writers 13/14** écrasent `shared_kv_states[layer_type]` avec K/V de taille T+1.
  2. Reader layers 15..34 prennent `shared_kv_states[self.layer_type]` (taille T+1), pas de cache local pour eux.

### 5.9 Finding
- **Hypothèse handoff** : *"toutes les 35 couches possèdent leurs poids K/V complets ; le sharing est runtime-only"*.
- **Confirme partiellement** :
  - ✅ **Sur disque (safetensors)** : oui, 35 jeux de poids K/V (cohérent avec § 4).
  - ❌ **Côté Python instancié** : non, seulement 15 jeux. Les 20 shared layers ne créent ni `k_proj` ni `v_proj` ni `k_norm` ni `v_norm` (ligne 790 `if not self.is_kv_shared_layer:`).
  - ✅ **Runtime-only au sens où aucune fusion de poids n'est faite** : les K/V "extra" du safetensors sont silencieusement ignorés par `_keys_to_ignore_on_load_unexpected` (lignes 54-60). Aucun pointeur de poids partagé entre layers.
- **Réfute** le sous-entendu "les 20 couches consumers exécutent leur k_proj/v_proj puis ignorent le résultat" : non, elles ne l'exécutent pas du tout. Le code dispatch via `if self.is_kv_shared_layer:`.
- **Précise** la sémantique de "runtime sharing" : c'est un **buffer `dict[layer_type, (K,V)]` rempli à chaque forward par 2 writers explicites** (couches 13 et 14 pour E2B), lu par 20 readers explicites (couches 15-34). Pas de pointeur global, pas de mémoire partagée GPU-level. Recalcul intégral à chaque forward.
- **Implication ZML** : on ne peut PAS modéliser le cache K/V uniformément `[L=35, ...]`. Il faut :
  - 15 slots K/V producer-cache (indexés par layer_idx, sliding ou full selon le type) ;
  - 2 slots `shared_kv_states` (1 par layer_type), réécrits à chaque forward step ;
  - 20 layers consumers qui ne maintiennent rien et lisent le buffer.

  ZML peut soit (a) suivre la structure Transformers (cache producer + buffer shared), soit (b) fusionner les deux en cache logique `[L, ...]` avec aliasing des slots 15..34 vers slots 13/14 selon `layer_type`. **Décision déférée à P5.1.**

### 5.10 Sliding window — vérification
- `sliding_window=512` est défini sur l'objet Attention (`self.sliding_window = config.sliding_window if self.is_sliding else None`, ligne 764), et passé à `attention_interface` (ligne 862) — donc appliqué via le **masque**, pas via une troncation du cache.
- Pour les **readers sliding**, ils reçoivent `shared_kv_states["sliding_attention"]` qui est de longueur "writer's cache length" (cf. § 5.8) et appliquent le masque sliding 512 par-dessus → **chaque reader sliding "voit" indépendamment sa propre fenêtre de 512 tokens** sur le même tenseur partagé.
- Détail important : la **layer_scalar** (buffer ligne 675) multiplie hidden_states à la fin du DecoderLayer — pas lié à shared_kv_states mais cohérent avec "chaque layer reste personnalisée même si elle partage K/V".

## 6. vLLM source trace

### 6.1 Source inspected
- vLLM version : **main @ `75f6945c`** (28 mai 2026, fetched via `gh api repos/vllm-project/vllm/contents/...`).
- File : `vllm/model_executor/models/gemma4.py` (1721 lignes).
- vLLM NON installé localement (ni dans aucun venv 3090) → lecture publique uniquement, conforme à la consigne « ne pas installer pour cette phase ».
- Logs : `logs/12_vllm_gemma4_source_trace.txt` (1721 lignes annotées d'un header de provenance), `logs/12_vllm_gemma4_grep.txt` (103 hits).
- Modules connexes mentionnés mais non extraits : `vllm/model_executor/layers/attention.py` (classe `Attention`), `vllm/v1/attention/backends/utils.py` (`KVSharingFastPrefillMetadata`). Non nécessaires pour fermer Q9.

### 6.2 Self/Cross split
**Question** : vLLM conserve-t-il 15 producers / 20 readers ?

**Finding** : **OUI, exactement** la même formule qu'en Transformers. La cartographie producers/readers est strictement identique.

**Evidence** (gemma4.py lignes 469-477) :
```python
first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared_layers  # 35 - 20 = 15
if layer_idx >= first_kv_shared_layer_idx:
    self.is_kv_shared_layer = True
    prev_layers = config.layer_types[:first_kv_shared_layer_idx]
    current_layer_type = config.layer_types[layer_idx]
    kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)
```
- Layers 0..14 = producers (instancient `qkv_proj`, calculent K/V, écrivent dans le cache pool).
- Layers 15..34 = readers (`is_kv_shared_layer = True`).
- Pour E2B : 2 sources de lecture seulement — layer 14 pour `full_attention`, layer 13 pour `sliding_attention` (identique aux 2 writers Transformers).

### 6.3 Shared KV representation
**Question** : vLLM représente-t-il seulement 2 shared KV states par type, comme Transformers ?

**Finding** : **NON, mécanisme fondamentalement différent**. Pas de buffer `shared_kv_states` éphémère par-forward. vLLM utilise un **string-pointer** statique vers la layer cible, résolu par le **backend d'attention** au niveau du **kv_cache pool**.

**Evidence** (lignes 486-509) :
```python
kv_sharing_target_layer_name = (
    f"{param_name_before_layers}.layers."
    f"{kv_shared_layer_index}.self_attn.attn"
)
...
self.attn = Attention(
    ...,
    kv_sharing_target_layer_name=kv_sharing_target_layer_name,
    prefix=f"{prefix}.attn",
)
```
Et dans `forward` (lignes 528-542) :
```python
if not self.is_kv_shared_layer:
    # apply K norm + RoPE on K, V norm
    ...
else:
    # Shared: only apply RoPE to Q (drop K, V calculation entirely)
    q = self.rotary_emb(positions, q, k)[0]

attn_output = self.attn(q, k, v)  # backend reads target layer's K/V cache
```

**Comparaison** :
| Mécanisme | Transformers 5.9 | vLLM main |
|---|---|---|
| Stockage shared K/V | `UserDict()` éphémère, créé par-forward (ligne 134 de `modeling_gemma4.py`) | Pas de buffer Python ; sharing au niveau du `kv_cache` pool backend |
| Indirection | Lecture/écriture explicite Python (`shared_kv_states[layer_type] = ...`) | String pointer statique (`kv_sharing_target_layer_name`) résolu par backend |
| Granularité | 2 entrées max par forward (`full_attention`, `sliding_attention`) | N pointers vers M slots cache (N=20 readers, M=2 sources E2B) — résolution implicite |
| `qkv_proj` shared layer | Pas instanciée (`if not self.is_kv_shared_layer:` ligne 790) | Instanciée (`QKVParallelLinear` ligne 416) mais K/V calculés puis ignorés (le backend lit les K/V de la target) |
| RoPE sur K shared | N/A (K pas calculé) | N/A (K calculé puis ignoré, RoPE Q-only ligne 540) |

**Subtilité** : vLLM **calcule** quand même `qkv_proj(hidden_states)` pour les shared layers (ligne 520), puis split en `q, k, v` mais **n'utilise pas** k/v. Coût compute marginal mais pas zéro — un `qkv_proj` complet vs un `q_proj` seul aurait été plus efficient. C'est un choix de simplicité d'API (unifier le forward producer/consumer).

### 6.4 Fast prefill
**Question** : le `kv_sharing_fast_prefill` change-t-il le layout logique ?

**Finding** : **NON, c'est une optimisation orthogonale du compute path**, pas un changement de structure de cache. Le sharing K/V (string pointer) marche identiquement avec ou sans fast prefill. Le fast prefill change seulement **où** les couches s'exécutent (deux subgraphs compilés) et **sur quels tokens** la cross-decoder tourne.

**Evidence** :
- `cache_config.kv_sharing_fast_prefill` est un toggle (ligne 1106), pas une condition de structure cache.
- Le `kv_sharing_target_layer_name` est setté à `__init__` (ligne 486) **indépendamment** du flag fast_prefill — donc le sharing K/V est actif dans les deux chemins.
- Quand `fast_prefill_enabled == True` :
  - `Gemma4SelfDecoderLayers` (lignes 793-923, décoré `@support_torch_compile(enable_if=...)`) compile embedding + PLE + layers 0..14 dans **un graphe**.
  - `Gemma4CrossDecoderLayers` (lignes 929-958) compile layers 15..34 dans **un autre graphe**.
  - `fast_prefill_forward` (lignes 1196-1279) :
    1. Self-decoder forward sur **tous** les `num_tokens` tokens.
    2. Récupère `logits_indices_padded` (positions où on a besoin de logits — typiquement la dernière position en prefill).
    3. Copie `self_decoder_hidden_states[logits_indices_padded]` dans `self.hidden_states[:num_padded]`.
    4. Cross-decoder forward sur **seulement** `num_padded` positions (avec le sharing K/V actif via le pool).
  - **Gain attendu** : en prefill long (ex. 4k tokens), self-decoder fait 15 layers × 4k tokens, cross-decoder fait 20 layers × **1 token** au lieu de 20 × 4k. ≈ 80% de FLOPs économisés sur la 2ème moitié.
- Quand `fast_prefill_enabled == False` (chemin normal, lignes 1301-1349) : boucle séquentielle sur les 35 layers, comme Transformers. Le sharing K/V reste actif (via pointer).

**Implication ZML** : pas besoin de répliquer le fast_prefill pour atteindre la parité Gemma 4. Le sharing K/V seul suffit pour la correction. Fast prefill = optimisation perf optionnelle, à reporter à P5.2+ si jamais.

### 6.5 Sliding/full separation
**Question** : comment vLLM sépare sliding/full dans le cache ?

**Finding** : **pas de séparation au niveau du cache lui-même**. Le cache pool stocke les K/V de chaque producer layer indépendamment de son `layer_type`. Le sliding window est **appliqué côté backend** via `per_layer_sliding_window` au moment du compute attention (masque), pas via troncation du cache.

**Evidence** (lignes 442-443, 506) :
```python
self.is_sliding = layer_type == "sliding_attention"
sliding_window = config.sliding_window if self.is_sliding else None
...
self.attn = Attention(
    ...,
    per_layer_sliding_window=sliding_window,  # applied at attention compute, not cache
    kv_sharing_target_layer_name=kv_sharing_target_layer_name,
)
```
- Le sharing K/V est **intra-type** : layer 19 (full) → target layer 14 (full) ; layer 15 (sliding) → target layer 13 (sliding). Garanti par `prev_layers[::-1].index(current_layer_type)` (ligne 476).
- Le cache du producer 13 (sliding) **stocke tous les tokens** ; le masque sliding 512 est appliqué au compute par chaque reader sliding individuellement. **Cohérent avec Transformers `store_full_length_kv`** (cf § 5.8).

### 6.6 Implication for ZML cache layout (Q9 close)
**Question Q9** : quel layout statique devient raisonnable ?

**Finding** : **layout A** (suivre Transformers : 15 slots producer-cache + 2 slots shared) ET **layout B** (cache uniforme `[L=35,...]` avec aliasing) sont tous deux compatibles avec la sémantique. **vLLM réalise effectivement le layout B mais sous une forme dégénérée** : son cache pool a 35 logical layer slots mais **seuls 15 sont alloués/écrits** ; les 20 slots des shared layers sont **redirections par string pointer** vers les slots des producers. C'est un "layout B avec pointeurs", pas un "layout B avec aliasing par layer_type".

**Décision pour ZML statique** :

**Adopter le layout A explicite** — plus simple à compiler en Zig, pas de gestion de pointeurs dynamiques :

```
KV cache:
  producer_kv[15]            // layers 0..14
                             // chaque slot shape: [B, S_max, n_kv_heads, head_dim]
                             // full-length (pas de troncation sliding au niveau cache)
                             // 7 slots sliding (layers 0,1,2,3,5,6,7,...) + 7 slots full + bias
                             // Note: les 15 producers contiennent 11 sliding + 4 full pour E2B
                             //   (cf § 3 layer_types : full = [4, 9, 14], sliding = autres)

Per-layer access table (15+20=35 entries, calculée à l'init depuis layer_types) :
  access[i] for i in 0..14    = ("write", i)
  access[i] for i in 15..34   = ("read", target_idx)
                                  où target_idx = 14 si layer_types[i] == "full_attention"
                                                  13 si layer_types[i] == "sliding_attention"

Sliding window mask :
  appliqué dans le compute attention par chaque layer
  shared layers : appliquent leur propre fenêtre 512 sur le cache full-length du target
                  comme Transformers et vLLM le font

État éphémère par-forward : zéro.
  - Pas de UserDict (style Transformers) : redondant si on a la table d'accès.
  - Pas de string pointers dynamiques (style vLLM) : statiques, calculés à l'init.
```

**Coût statique** :
- Mémoire : 15 slots de cache K/V (vs 35 naïf), ratio **15/35 ≈ 43% du cache d'une variante non-YOCO de même architecture**.
- Compute producer : identique à un transformer standard.
- Compute consumer : `q_proj` + RoPE Q-only + attention sur target cache. **Pas de qkv_proj sur les 20 shared layers** (gain par rapport à vLLM qui calcule puis jette ; en ZML on peut sauter complètement). Économie ≈ 2/3 du compute QKV global.

**Compatibilité multi-régimes** :
- ✅ Prefill normal : tous les producers calculent et écrivent leur slot ; tous les readers lisent leur slot cible.
- ✅ Decode token T+1 : producers font 1-token append à leur slot ; readers lisent le slot cible à jour.
- ❌ Fast prefill (vLLM `kv_sharing_fast_prefill`) : non implémenté. Pas requis pour parité Gemma 4 correcte. À considérer P5.2+ si bench montre un goulot.

**Validation prochaine** : implémenter ce layout en P5.1 avec un seul oracle de référence (sortie PyTorch fp32 = `logits`) sur prompt fixe court → comparer ZML vs PyTorch via le pattern Gate de P4.4.2.

## 7. Prefill lifecycle (fermé via § 5 + § 6)

1. Tokenisation → `input_ids` shape `[B, T]`.
2. Embeddings principal + PLE (P4.4.2, validé) → `inputs_embeds` + `per_layer_inputs`.
3. Décodage couche par couche, layers 0..34 séquentiellement (chemin "normal" sans fast prefill) :
   - **Layers 0..14 (producers)** : `qkv_proj` complet → split q/k/v → q_norm + k_norm + v_norm + RoPE(q,k) → écrivent K/V dans `producer_kv[layer_idx]` (full-length, pas de troncation) → attention compute avec sliding_window appliqué via masque si layer_type=sliding → o_proj.
   - **Layers 15..34 (readers)** : `q_proj` seul (ou `qkv_proj` puis drop k,v comme vLLM) → q_norm + RoPE(q) → attention compute en lisant K/V depuis `producer_kv[target_idx]` (target_idx = 14 pour full, 13 pour sliding) → masque sliding 512 appliqué si layer_type=sliding → o_proj.
4. Post-layer : pre/post layernorms, MLP (ou MoE si layer.enable_moe_block), PLE residual (`per_layer_input_gate` + `per_layer_projection`), layer_scalar.
5. Final RMSNorm + lm_head → logits.

**Particularité prefill** : tous les tokens sont traités en parallèle. Les producers remplissent `producer_kv` à pleine longueur T. Les readers attaquent directement le cache complet. Une seule passe forward sur tous les tokens.

## 8. Decode lifecycle (fermé via § 5 + § 6)

Pour le token T+1 (après prefill de T tokens) :

1. Embedding + PLE du **nouveau token seul** (shape `[B, 1, H]`).
2. Layers 0..34 séquentielles, chacune avec `seq_len=1` pour le compute :
   - **Producers** : recalculent q/k/v sur le token seul, **append** au slot `producer_kv[layer_idx]` (taille passe de T à T+1).
   - **Readers** : recalculent q seul, lisent `producer_kv[target_idx]` (taille T+1) tel quel.
3. Le cache `producer_kv` est l'unique état persistant inter-tokens. Pas de buffer éphémère shared.
4. Sliding mask par layer sliding : limite l'attention aux derniers 512 tokens du cache (cache reste full, masque coupe).

**Coût marginal par token decode** : 15 producers × 1-token qkv + 20 readers × 1-token q + 35 layers attention compute (sur cache T+1, avec masque sliding pour 28 layers et masque full pour 7).

## 9. ZML cache layout — Q9 FERMÉE

Voir § 6.6 pour la décision. Récap exécutable :

```
struct KVCache {
  // 15 slots producer, alloués une fois
  producer_kv: [15]struct {
    K: Tensor([B, S_max, n_kv_heads, head_dim], fp16/bf16),
    V: Tensor([B, S_max, n_kv_heads, head_dim], fp16/bf16),
  },
  current_len: usize,  // longueur logique courante (commune aux 15 slots)
}

// Table d'accès statique, calculée à l'init depuis config.layer_types
fn cache_target(layer_idx: usize, layer_types: []const LayerType) usize {
  if (layer_idx < 15) return layer_idx;
  const t = layer_types[layer_idx];
  // Pour E2B : full_attention → 14, sliding_attention → 13
  // En général : dernière occurrence de `t` dans layer_types[0..15]
  // (précalculée à l'init, pas runtime)
  return last_occurrence_of(t, layer_types[0..15]);
}
```

**Contraintes ZML capitalisées P4.4.2 (toujours valables)** :
- `.withTags(...)` obligatoire après `reshape` (piège #1).
- `weight.broad(shape)` pour broadcast explicite sur `mul`/`add` (piège #2).
- Pattern RMSNorm Llama (`normalized.mul(weight)`) — pas Qwen `(1+weight)` (piège #3).

**Standard de comparaison** (validé Gate J) :
```zig
var ref_slice = try buffers.<reference_tensor>.toSliceAlloc(allocator, io);
defer ref_slice.free(allocator);
```

## 10. Open questions / BLOCKERS

1. ~~**(BLOCKER P5.0)** Si tous les poids K/V sont présents, comment Transformers décide-t-il quelles couches sont source vs consumer ?~~ **FERMÉ § 5.6** : `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers = 15`, layers `< 15` producers (sources), layers `>= 15` consumers. Et **seulement 15 jeux de poids K/V** instanciés en Python (les 20 du haut ne créent pas leurs modules).
2. ~~Est-ce que le partage est intra-bloc ou inter-bloc ?~~ **FERMÉ § 5.6** : inter-bloc strict. Préfixe `[0..14]` = producers, suffixe `[15..34]` = consumers. Aucune notion de bloc 5-couches dans le sharing.
3. Le PLE injection (validé P4.4.2) se fait à chaque couche : est-elle indépendante du sharing K/V ou y a-t-il un couplage ? **PARTIELLEMENT FERMÉ § 5.7** : PLE est appliqué dans `Gemma4TextDecoderLayer.forward` *après* l'attention (lignes 739-746), indépendamment de `is_kv_shared_layer`. **Pas de couplage architectural visible** entre PLE et shared KV. La 4ᵉ phase PLE (`per_layer_input_gate` + `per_layer_projection` + `post_per_layer_input_norm`) tourne pour les 35 couches identiquement.
4. Quel est le comportement attendu si on charge le modèle "naïvement" et qu'on exécute les 35 couches avec leurs propres K/V ? **NON FORMELLEMENT TESTÉ** : les modules n'existent pas (cf § 5.9), donc impossible en l'état sans patcher `__init__`. À tester en P5.1 comme oracle de divergence (charger les 35 poids brutes du safetensors et faire 35 forwards K/V indépendants pour comparer).

**Nouvelle question P5.0** (à étape 4 vLLM) :
5. ~~vLLM implémente-t-il un layout `[L, ...]` uniforme avec aliasing, ou maintient-il deux structures séparées comme Transformers ?~~ **FERMÉ § 6.3** : ni l'un ni l'autre — vLLM utilise un **string-pointer statique par layer** (`kv_sharing_target_layer_name`) résolu par le backend d'attention au niveau du `kv_cache` pool. Pas de buffer Python éphémère, pas d'aliasing implicite : la table de redirection est explicite et calculée à `__init__`.

## Critères de clôture P5.0 (rappel)

P5.0 est fermé quand on peut répondre à ces 9 questions :

1. Quelles couches full_attention ? **OK : [4, 9, 14, 19, 24, 29, 34]**
2. Quelles couches sliding_attention ? **OK : les 28 restantes**
3. Quelles couches possèdent leurs propres K/V ? **OK nuancé : 35 sur safetensors disque, 15 en Python instancié (cf § 5.9 nota bene).**
4. Quelles couches partagent K/V ? **OK § 5.6 + § 6.2 : layers 15..34 (les 20 dernières). Confirmé identique dans Transformers et vLLM.**
5. Quelle couche écrit dans `shared_kv_states` ? **OK § 5.4 + § 6.3 : pour Transformers, layer 14 écrit `["full_attention"]`, layer 13 écrit `["sliding_attention"]`. Pour vLLM, le concept de "writer dans un dict" n'existe pas — les layers 13/14 écrivent dans leur propre slot du `kv_cache` pool, et les readers pointent vers ces slots.**
6. Quelle couche lit `shared_kv_states` ? **OK § 5.5 + § 6.2 : layers full readers = [19, 24, 29, 34] ; layers sliding readers = [15-18, 20-23, 25-28, 30-33]. Mêmes layer-sets dans les deux runtimes.**
7. Comment `past_key_values` interagit avec `shared_kv_states` ? **OK § 5.8 + § 6.3 : dans Transformers, deux structures disjointes (cache standard + UserDict éphémère). Dans vLLM, une seule structure (`kv_cache` pool) avec redirection par pointer. La sémantique est équivalente : les readers lisent le cache complet d'une layer source de même `layer_type`, jamais le cache d'une layer shared.**
8. Que change prefill vs decode ? **OK § 5.4/5.5 + § 6.4 : rien dans le dispatch du sharing K/V (identique dans les deux runtimes, identique entre prefill et decode). vLLM ajoute une optimisation orthogonale optionnelle (`kv_sharing_fast_prefill`) qui compile deux subgraphs séparés et ne fait tourner le cross-decoder que sur les logits indices en prefill — **pas requis pour la correction**, optionnel pour la perf.**
9. Quel layout de cache faudra-t-il prévoir en ZML ? **OK § 6.6 + § 9 : Layout A explicite — 15 slots `producer_kv` full-length + table d'accès statique de 35 entrées (identity pour 0..14, target_idx pour 15..34 selon layer_type). Zéro buffer éphémère. Sliding window appliqué via masque, pas via troncation cache. Couvre prefill et decode. Fast prefill différé à P5.2+.**

**9/9 répondues. P5.0 prêt à fermer.** Suggestion : commit + tag `p5.0-cartography-done` (ou équivalent), puis P5.1 = première implémentation ZML du cache + des shared layers.
