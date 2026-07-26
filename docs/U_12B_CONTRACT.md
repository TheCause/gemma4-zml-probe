# U0 — le contrat 12B Unified, vérifié sur pièce

Jalon J2 (Gemma 4 12B `google/gemma-4-12B-it-qat-w4a16-ct` w4a16 sur la 3090). Cette carte
est committée AVANT tout code moteur (règle J1 restaurée par l'Amendement 2026-07-24) —
elle fige les faits D1 (D1/D7 = décisions figées du plan J2, cf. renvoi ci-dessous) sur
lesquels `engine.zig` (géométrie comptime `Geom`, Task 2) et le runner `g12.zig` (Task 8)
seront bâtis. Source de vérité : `scripts/62_u0_contract.py`, exécuté sur la VM
(`/data/venvs/g12b`), manifest auto-vérifié `fixtures/u0_contract_manifest.json`.

**Spec :** `docs/superpowers/specs/2026-07-18-w4-poids-4bit-12b-design.md` (§4.2 corrigée par ce contrat)
**Plan :** `docs/superpowers/plans/2026-07-24-w4-j2-12b-unified.md` (décisions D1-D13 + Amendement 2026-07-24)

## 1. Les deux géométries d'attention (D1)

| | 40 couches sliding | 8 couches full (5,11,…,47) |
|---|---|---|
| Q | 16 × hd **256** | 16 × hd **512** (`global_head_dim`) |
| KV | **8** KV × 256 → GQA **groupe 2** | **1** KV × 512 → **MQA** (groupe 16, broadcast) |
| v_proj | présent | **ABSENT** (`attention_k_eq_v`) |
| V | `v_norm(v_proj(x))` | **`v_norm(k_proj(x))` — capturé AVANT k_norm et AVANT RoPE** |
| RoPE | default, theta 1e4, 256 dims pleines | proportional, theta 1e6, partial 0.25 sur 512 (== chemin full E2B) — **64 fréquences actives, padding zéros dans `inv_freq`, `rotate_half` par moitiés** |

Commun aux deux géométries : scaling d'attention 1.0, pas de softcap d'attention texte,
RMSNorm `x*weight` fp32 (**PAS** `1+weight`), q_norm/k_norm avec poids (256/512), v_norm
**sans** poids (`with_scale=False`), sandwich norms, MLP gelu-tanh inter 15360 partout,
`layer_scalar` bf16 `[1]` × 48 au checkpoint, softcap final 30, tied lm_head, vocab 262144,
eps 1e-6, fenêtre sliding 1024, ni PLE ni YOCO ni MoE, préfixes `model.language_model.*`,
multimodal ignoré.

## 2. Extrait du forward K=V (source de vérité)

`transformers` 5.9.0+, `models/gemma4/modeling_gemma4.py` l.1226-1249 :

```python
query_states = self.q_proj(hidden_states).view(hidden_shape)
query_states = self.q_norm(query_states)
query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
...
key_states = self.k_proj(hidden_states).view(hidden_shape)
value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
key_states = self.k_norm(key_states)
key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
key_states = key_states.transpose(1, 2)
value_states = self.v_norm(value_states)
value_states = value_states.transpose(1, 2)
```

Le point décisif pour les couches full (K=V, `v_proj is None`) : `value_states` part de
`key_states` **AVANT** `k_norm` et **AVANT** `apply_rotary_pos_emb` — la capture doit se faire
sur la sortie brute de `k_proj`, pas sur le `key_states` normé/roté qui alimente les scores QK.

## 3. Shapes sondes du checkpoint (script 62)

```
model.language_model.layers.0.self_attn.q_proj.weight_packed  : [4096, 480]   # 16x256, in 3840
model.language_model.layers.0.self_attn.q_proj.weight_scale   : [4096, 120]
model.language_model.layers.5.self_attn.q_proj.weight_packed  : [8192, 480]   # 16x512 (full)
model.language_model.layers.5.self_attn.k_proj.weight_packed  : [512, 480]    # 1x512 MQA
model.language_model.layers.5.self_attn.o_proj.weight_packed  : [3840, 1024]  # in 8192
model.language_model.layers.0.mlp.down_proj.weight_packed     : [3840, 1920]  # in 15360
model.language_model.embed_tokens.weight                      : [262144, 3840]
lm_head.weight                                                 : [262144, 3840]
model.language_model.layers.5.self_attn.q_norm.weight          : [512]
model.language_model.layers.0.self_attn.q_norm.weight          : [256]
model.language_model.layers.0.layer_scalar                     : [1]
```

Confirmé sur pièce : 328 `weight_packed` au total (48×6 tenseurs projetés + 40 `v_proj`,
les 8 couches full n'en ayant pas), `v_proj` présent uniquement sur les 40 couches sliding,
absent sur les 8 full, aucun `.v_norm.` pondéré, ni `weight_zero_point` ni `weight_g_idx`
(quantization `pack-quantized`, group_size 32, symmetric, 4 bits).

## 4. Motif des couches full

`(i+1) % 6 == 0` → couches full = `{5, 11, 17, 23, 29, 35, 41, 47}` (indices 0-based, 48
couches). Vérifié bit contre `config.json.text_config.layer_types` du checkpoint.

## 5. Pièges hérités applicables (repo)

- **8** — scalings d'embeddings implicites : `embed_tokens` (`×√1536` E2B / `×62.0` bf16 12B,
  cf. D12 §7) ET `embed_tokens_per_layer` (`×√256=16`, **sans objet ici : `ple_dim=0`** au 12B —
  ni PLE ni YOCO, cf. §1) portent chacun un scale (`Gemma4TextScaledWordEmbedding`) ; ne pas
  en oublier un côté 12B.
- **10** — attention scaling = 1.0, **pas** de softcap d'attention (softcap uniquement sur
  les logits finaux, 30.0) ; masque additif, softmax fp32 (repris tel quel du piège d'origine,
  aucune omission volontaire côté 12B).
- **12** — « oracle » distingue deux choses à ne pas confondre : le **modèle HF officiel**
  (`transformers` 5.14.1 sur le checkpoint réel — la référence, **jamais en cause**) vs le
  **script oracle Python** (réimplémentation/reproduction locale — **peut** avoir des bugs,
  cf. les 2 bugs oracle E2B documentés en D12 §7). Le piège officiel : dériver le script
  oracle du code source réel de `modeling_gemma4.py`, **jamais d'une hypothèse** ré-encodée —
  toute divergence moteur/script-oracle se diagnostique en suspectant d'abord le script, pas
  le modèle HF.
- **13** — comparer des logits, pas des argmax seuls (les flips top-1 masquent des dérives
  numériques réelles).
- **17** — marges/égalités (ties) : la comparaison bit doit gérer les cas de scores à égalité
  sans faux-négatif. Piège d'origine formulé en contexte batching (`docs/DOCUMENTATION.md`
  §8.17) — généralisé ici aux ties de toute comparaison, pas seulement B>1.
- **19** — `zml.nn.sdpa` scale K par 1/√hd par défaut (`engine.zig:473-478`, `AttnKind.sdpa`)
  — `.scale = 1.0` obligatoire si ce chemin est un jour activé pour le 12B.
- **20** — scale-invariance : un contre-test sur `gate_proj` (MLP) doit vérifier que le moteur
  n'est pas invariant à une renormalisation qui masquerait un bug de câblage.

## 6. Faits U0 constatés par le run (`fixtures/u0_contract_manifest.json`)

Run officiel VM (`/data/venvs/g12b/bin/python3 scripts/62_u0_contract.py`), **exit=0, PASS U0**.

- **`lm_head_eq_embed` : `True`** — `lm_head.weight` est bit-identique à
  `model.language_model.embed_tokens.weight` (comparé par tranches de 16384 lignes sur les
  262144). Le lm_head est bien tied ; la voie d'échec D7 (STOP décision humaine) ne s'est
  **pas** déclenchée.
- **`templates_match` : `False`.** `chat_template.jinja` du snapshot 12B
  (sha256 `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`) **diffère** du
  template E2B de référence (sha256 `2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082`).
  **Conséquence actée** : le chat template côté Zig (clone/prompt-formatting utilisé par le
  runner de décode) devra être **adapté au template 12B spécifiquement** en Task 8 — ne pas
  réutiliser tel quel le template E2B câblé aujourd'hui. Vérification manuelle du diff textuel
  des deux templates à faire avant d'écrire ce câblage.
- **Snapshot checkpoint** : `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee` (figé, cf Amendement).
- Toutes les autres assertions D1 (328 packed, shapes sondes, motif full, config `text_config`,
  `quantization_config`, sha tokenizer == E2B) sont tenues — cf. §3-4 ci-dessus.

## 7. D12 — downcast embed scale (E2B vs 12B)

`zml_runner/engine.zig:38` définit `const EMBED_SCALE: f64 = @sqrt(1536.0);` (constante
**comptime Zig, double précision** — pas de buffer bf16). Elle est appliquée en 4 sites
(`engine.zig:591,630,673,711`, dont `forwardStep` l.673) via
`embeds_step.convert(.f32).scale(EMBED_SCALE)` : les embeddings bf16 sont **d'abord convertis
en f32**, **puis** multipliés par la constante f64 exacte (`√1536 = 39.191835884…`, promue au
type du tenseur) — le moteur n'arrondit jamais ce scale en bf16.

**Historique (`docs/ENGINE_LOG.md`, entrée du 2 juin, validation E2B)** : le vrai module HF
`Gemma4TextScaledWordEmbedding` construit son buffer `embed_scale` **casté au dtype des
poids** (bf16), ce qui arrondit `√1536 = 39.1918…` à **39.25**. Le premier oracle Python
(script 39) reproduisait fidèlement ce comportement HF et produisait un écart mesurable
(STAGE0 `max_abs 0.032`, catalogué WARN) contre le moteur ZML fp32. Diagnostic : ce n'était
**pas** un bug moteur mais un **bug oracle** — l'oracle a été corrigé en forçant
`model.embed_tokens.embed_scale = torch.tensor(√1536, dtype=fp32)`, c'est-à-dire en
**contournant délibérément** l'arrondi bf16 réel de HF pour aligner l'oracle sur le
comportement (déjà fp32) du moteur E2B. Ce choix a été validé numériquement bout-en-bout
(producer+reader PASS, argmax identique 4/4 vs HF).

**Pourquoi le 12B fait le choix inverse** : pour le 12B, `hidden_size = 3840`, donc
`√3840 = 61.96773…`. En bf16 (ULP = 0.25 sur cette plage de magnitude), la valeur
représentable la plus proche est **62.0 exactement** — un arrondi net et significatif
(différence relative ~0.05 %), contrairement à E2B où l'écart 39.1918→39.25 avait été
neutralisé sans remise en cause du chemin de validation existant. Le gate U8 du 12B compare
le décodage du moteur **directement au checkpoint officiel via HF CPU** (pas un oracle
retravaillé comme pour E2B) — pour matcher cette source de vérité **au bit**, le 12B doit
donc **reproduire fidèlement l'arrondi bf16 réel** de `Gemma4TextScaledWordEmbedding`, soit
une constante bf16 **`62.0`**, et non `√3840` en pleine précision.

**Où l'appliquer** : cette constante `62.0` doit être câblée **uniquement dans le chemin
runner 12B** (`g12.zig` / assemblage `EngineModel` instancié `Geom.g12`, Task 8) — **jamais**
dans `engine.zig`, dont la neutralité vis-à-vis d'E2B est prouvée par md5 HLO
`before_optimizations` (gate U1, Task 2) : modifier `EMBED_SCALE` dans `engine.zig` casserait
cette preuve de non-régression pour E2B. Le paramètre `Geom` devra donc porter un scale
d'embedding spécifique par instanciation (E2B garde `√1536` fp64 tel quel, 12B reçoit `62.0`
bf16-exact), pas une formule générique `@sqrt(hidden_size)` recalculée pour les deux.

## 8. Versions figées (Amendement 2026-07-24)

- `transformers` **5.14.1**, `torch` **2.13.0**, `compressed-tensors` **0.17.1** — fixés dans
  les deux venvs `g12b` (VM **et** M4).
- Snapshot checkpoint figé : `1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee`.
- **Dérive d'API 5.14** : `apply_chat_template(..., return_tensors="pt")` renvoie désormais
  un **`BatchEncoding`** (pas un `Tensor` nu). Tous les scripts 62-70 doivent utiliser
  `return_dict=True` puis `model.generate(**enc)` (pas `model.generate(enc)`).

---

*Généré à partir du run officiel `scripts/62_u0_contract.py` (exit=0, PASS U0) et de
`fixtures/u0_contract_manifest.json`. Le script contenait initialement un bug de résolution
de chemin (`os.path.realpath` traversait les 2 niveaux de symlinks HF-cache et atterrissait
dans `blobs/`, sans `config.json`) — corrigé par le contrôleur (commit `524e7a8` sur le plan)
en `snap = os.path.dirname(os.readlink(CKPT))` (un seul hop, le symlink `weights_12b/model.safetensors`
étant absolu depuis le Step 0.5).*
