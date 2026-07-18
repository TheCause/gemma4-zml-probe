# Design — Chantier W4 : poids 4-bit dans ZML → Gemma 4 12B sur la 3090

**Date** : 2026-07-18
**Statut** : spec approuvée, **rangée au tiroir** (décision Régis 18 juil 2026 : design + spec
maintenant, exécution plus tard — « nous plantons une graine »). Aucun plan d'implémentation
n'existe encore ; le jour venu, invoquer `writing-plans` depuis cette spec.
**Prérequis de lecture** : `docs/DOCUMENTATION.md` (méthode + 19 pièges), `docs/BATCHING_RESULTS.md`
(gates T0/S1, enveloppe), `docs/G2_3_OP_SENSITIVITY.md` (méthode de l'enveloppe).

---

## 1. Contexte & objectif

Le portage E2B est clos (== HF, runtime autonome, batching ×18,5, backlog vide). Le désir
d'origine de ce chantier était « porter Gemma 4 12B en ZML » ; le cadrage a montré que :

- le 12B (`google/gemma-4-12B-it`, 11,95 B params) est une **autre architecture**
  (`Gemma4UnifiedForConditionalGeneration`) : ni PLE ni YOCO, dense classique ;
- en bf16, ses poids seuls font ~24 Go = la VRAM totale de la 3090 → **infaisable sans
  quantization des poids** ;
- ZML n'a aucun chemin poids-quantifiés aujourd'hui (les dtypes `i4/u4/f4e2m1` existent dans
  `zml/dtype.zig`, rien ne les consomme ; TurboQuant n'a quantifié que le cache V).

**Reformulation actée** : le chantier est « **poids 4-bit dans ZML** » ; le 12B qui décode sur la
3090 en est le point d'arrivée, pas le point de départ. Justification alambic examinée et
**écartée** (le teacher T_logits HF NF4 suffit ; le dump teacher-forcing sature déjà le GPU, les
×18,5 du batching ne s'appliquent pas à ce régime). Chantier assumé comme projet de maîtrise
technique + déblocage matériel.

**Approche retenue (A)** : format unique w4a16 compressed-tensors, deux jalons — J1 brique W4
prouvée sur E2B (terrain connu), J2 portage 12B Unified par-dessus. Ordre strict J1 → J2.

## 2. Périmètre

**Dans le scope** :
- Brique ZML « poids int4 groupés → dequant en graphe → GEMM bf16 » (format §3.1).
- E2B-W4 : décode complet == oracle HF sur le même checkpoint quantifié.
- Paramétrage comptime du moteur (`EngineConfig`) avec preuve de neutralité HLO pour E2B.
- 12B Unified texte seul : op-par-op → décode complet W4 sur la 3090, mono-séquence greedy,
  génération type 1020 tokens (fenêtre sliding 1024 franchie).

**Hors scope explicite** :
- Quantization maison (TurboQuant poids, Hadamard/PolarQuant) — chantier papier distinct ;
  J1 lui prépare le terrain, rien de plus.
- Route q4_0/GGUF (parser + oracle llama.cpp = friction pour zéro gain vs w4a16-ct).
- Multimodal (le 12B Unified projette patches image / waveform audio — ignoré, comme les
  towers vision/audio d'E2B), sampling, batching continu/serving, contexte 256K.
- Le batching statique existant n'est **pas** un livrable J2 (s'il survit gratuitement au
  paramétrage, tant mieux ; le gate final est mono-séquence).

## 3. Jalon 1 — la brique W4, prouvée sur E2B

### 3.1 Format cible (= celui du checkpoint Google 12B, vérifié 18 juil 2026)

`google/gemma-4-12B-it-qat-w4a16-ct` : **compressed-tensors, format `pack-quantized`** —
int4 **symétrique** (pas de zero-point), scales par **groupe de 32**, stratégie `group`,
**Linear uniquement** ; lm_head, embeddings, norms restent bf16. Les 8 poids int4 sont packés
par int32. Concrètement, par couche linéaire : `weight_packed` (i32) + `weight_scale`
(une échelle par groupe de 32). Shapes exactes et ordre des nibbles = à figer au gate W1,
l'oracle Python fait foi.

### 3.2 Artefact d'entrée J1 : E2B quantifié maison

Google ne publie pas de w4a16-ct pour E2B → on le fabrique **à la même recette** :
- Source : `google/gemma-4-E2B-it-qat-q4_0-unquantized` (bf16 *entraîné pour* le 4-bit).
- Outil : `llm-compressor` (outil officiel compressed-tensors). **Repli si l'archi
  `Gemma4ForConditionalGeneration` n'est pas supportée** : script numpy maison produisant le
  même layout (le format §3.1 est simple : quantize sym par groupe de 32 + packing).
- Validation de l'artefact **avant tout Zig** : gate W0 (round-trip HF, §5).

### 3.3 Brique ZML `dequantW4`

Fonction de graphe : unpack (shifts + masks ; ops vérifiées présentes dans `zml/tensor.zig` —
`bitCast` L282, `shiftLeft` L379, `shiftRightArithmetic/Logical` L384/389, `logical` L1031,
`convert` L1050), conversion bf16, `mul` par la scale broadcastée par groupe, reshape
`[out, in]`. Sortie = Tensor poids bf16 ordinaire consommé par les `dot` existants. Les poids
résident en VRAM en i32 (÷4 mémoire) ; le bf16 n'existe que transitoirement pendant le forward.

### 3.4 Intégration : pattern wrapper, moteur intact

Même philosophie que L3/StepTok : la brique vit **en amont** du moteur, côté runner. Le runner
W4 charge packed+scales via le `TensorStore` habituel, applique `dequantW4` à la construction du
graphe, et tend au moteur des poids symboliques bf16 — `engine.zig` reste témoin des gates
existants. Point d'accroche exact (champ par champ) = décision de plan, sources sous les yeux ;
si le wrapper pur s'avère impossible et qu'engine.zig doit bouger, la preuve de neutralité HLO
(md5 `before_optimizations`, méthode T0/S1) devient obligatoire.

### 3.5 Oracles J1 (trois étages, du plus petit au plus gros)

- **Étage 0** : 1 groupe de 32 poids choisis à la main, packés en Python → unpack ZML comparé
  **au bit près**. Tue le risque n°1 (ordre nibbles, signe) avant qu'il coûte.
- **Étage 1** : script numpy de référence dequantant le checkpoint entier → comparaison par
  tranches (pattern scripts 07/14).
- **Étage 2** : HF transformers chargeant **le même checkpoint** décompressé bf16 → décode
  greedy 48 tokens, référence de bout en bout. Device : **CPU de la VM** par défaut (E2B bf16
  ≈ 10 Go, ça passe ; déterminisme préférable, cohérent avec les oracles historiques) ; GPU
  seulement si trop lent, alors comparaison sous enveloppe.

**Critère épistémique** : on ne compare JAMAIS E2B-W4 à E2B-bf16 pour la correction (deux
modèles différents). Vérité = ZML-W4 vs HF-lisant-le-même-checkpoint-W4.

## 4. Jalon 2 — 12B Unified

### 4.1 Paramétrage du moteur : `EngineConfig` comptime + neutralité

`engine.zig` est E2B au marteau (`NUM_LAYERS=35`, `D=1536`, slots YOCO, `HD_SLIDING/FULL`,
`EMBED_SCALE`…). Design : struct de config **comptime** passé à `EngineModel` ; le 12B = autre
instanciation du même moteur. **Gate U1 obligatoire** : avec la config E2B, HLO
`before_optimizations` **byte-identique** (md5) à l'actuel → les gates E2B restent valides par
construction (méthode qui a réussi pour le shape-polymorphisme, gate T0/S1).

### 4.2 Carte des diffs 12B vs E2B (config vérifiée 18 juil 2026)

| Disparaît | Change | Neuf (à gater) |
|---|---|---|
| PLE (`hidden_size_per_layer_input: 0`) — branche comptime-morte | 48 couches, H=3840 (scale √3840) | **GQA 16 Q / 8 KV** (E2B = MQA KVH=1) : repeat K/V par groupe de 2 têtes Q |
| YOCO (`num_kv_shared_layers: 0`) — 48 caches K/V ordinaires, plus de policy table | motif **5 sliding + 1 full ×8**, fenêtre **1024** | **p-RoPE** couches full : `partial_rotary_factor 0.25` (seul le 1er quart des 256 dims tourne, theta 1e6) ; sliding = RoPE standard theta 1e4 |
| head_dim dual 256/512 → **256 uniforme** | softcap 30.0, vocab 262144 — identiques | |

Solde net favorable : PLE+YOCO (la moitié du portage E2B) disparaissent ; GQA + p-RoPE sont
deux briques bien bornées.

### 4.3 Gate zéro : cartographie du contrat (U0)

Avant toute op, lire `modeling_gemma4_unified.py` + manifest du checkpoint. À vérifier
explicitement : **préfixes des noms de poids** (E2B = `model.language_model.*` ; Unified = ?),
**variante RMSNorm** (`weight` vs `1+weight`, piège Qwen), **lm_head tied ou non**, layout
exact `weight_packed`/`weight_scale`. Règle maison : « oracle = source de vérité, pas
hypothèse » (cf bug v_norm D.0→D.0b).

### 4.4 Méthode & oracle full-decode

Méthode E2B inchangée : op-par-op contre oracle PyTorch par **tranches** extraites via
`safe_open` (jamais les 24 Go chargés), puis couche 0 → N couches → prefill → logits+softcap →
décode court → décode long. Chaque gate committé/taggé.

**Oracle full-decode 12B** : le bf16 ne rentre pas sur la 3090 → référence = HF **CPU**,
checkpoint w4a16 décompressé bf16, ~48 tokens greedy (lent mais suffisant). RAM CPU de la VM à
vérifier au plan ; **repli** : vLLM w4a16 sur GPU + méthode de l'enveloppe (kernels différents).

**Cache KV 12B** : 40 couches sliding bornées à 1024 positions, 8 full pleine longueur ;
~1-2 Go aux contextes de travail — pas un sujet.

## 5. Gates & critères de PASS

**Série W (J1)** :
- **W0** — artefact valide : quantifier E2B, recharger dans HF, décoder 48 tokens cohérents.
  Aucun Zig avant W0 PASS.
- **W1** — unpack **bit-exact** sur 1 groupe de 32 poids connus.
- **W2** — dequant d'une matrice complète == référence numpy (tranches).
- **W3** — premier GEMM W4 (q_proj couche 0) == oracle.
- **W4g** — décode complet E2B-W4 **48/48 == HF-même-checkpoint** + mesure débit/VRAM
  (observation, pas critère).
- **WN** (conditionnel) — si engine.zig touché : HLO md5 identique pour gen_auto bf16.

**Série U (J2)** :
- **U0** cartographie du contrat (doc committée) → **U1** `EngineConfig` + HLO byte-identique
  E2B → **U2** embeddings ×√3840 (tranches) → **U3** attention sliding couche 0 →
  **U4** GQA repeat-KV → **U5** p-RoPE partial 0.25 (sanity : identité en position 0) →
  **U6** couche 0 complète puis N couches → **U7** prefill + logits softcap (top-k overlap) →
  **U8** décode court == oracle CPU → **U9** décode long, fenêtre 1024 **franchie**,
  non-vacuité prouvée par les logits (leçon « vacuité de l'antécédent ») →
  **U10** débit + VRAM finals (informatif).

**Critères** : bit-exact partout où déterministe (CPU, unpack, tranches) ; `==` greedy pour les
décodes ; **enveloppe ≤ 2× le bruit de la référence** pour toute comparaison GPU recompilée
(méthode G2.3 ; jamais de bit-à-bit entre deux compiles XLA-GPU — pièges 15/17). Un gate qui ne
peut pas échouer ne compte pas.

## 6. Risques & parades

| # | Risque | Parade |
|---|---|---|
| R1 | Ordre nibbles / signe int4 faux → charabia plausible en aval | W1 en premier, bit-exact |
| R2 | `llm-compressor` ne digère pas l'archi E2B | Script numpy maison, même format |
| R3 | Checkpoint maison illisible par HF | W0 round-trip avant tout Zig |
| R4 | XLA matérialise mal le dequant → débit décevant | Mesure ≠ gate ; correctness first |
| R5 | GQA repeat-KV erroné | U4 dédié par tranche |
| R6 | p-RoPE partial faux | U5 + oracle scripté + sanity position 0 |
| R7 | RMSNorm variante / préfixes / lm_head tied | U0 cartographie obligatoire |
| R8 | **Disque VM : ~16 Go libres** vs E2B-unquantized ~10 Go + 12B-w4a16 ~8 Go | `df` avant download, ménage `/data` d'abord ; jamais sur M1 (lui-même critique) |
| R9 | Non-déterminisme XLA-GPU | Enveloppe G2.3, jamais bit-à-bit inter-compiles |
| R10 | RAM CPU VM insuffisante pour l'oracle 12B | Vérifier au plan ; repli vLLM w4a16 + enveloppe |

## 7. Budget & infra

- **VRAM 12B-W4 estimée** : ~6 Go linears i32-packés + ~2 Go embeddings bf16 + cache KV +
  activations ≈ **10-12 Go** — marge confortable sur 24 Go (le pic E2B mesuré était 16,3 GiB,
  dominé par la compile XLA — à re-mesurer, la garde 20 GiB de gen_auto sert de référence).
- **Machines** : quantization + oracles Python sur VM 3090 (`/data`, venv dédié à créer ou
  réutiliser `/data/venvs/gemma4-probe` + `llm-compressor`/`compressed-tensors`), build ZML
  `/data/rqz_workspace/zml`, pilotage M1. Chemins connus : poids E2B
  `/data/gemma4-zml-probe/weights/model.safetensors`, tokenizer
  `/data/gemma4-zml-probe/gemma4-e2b-it-meta/tokenizer.json`. Le tokenizer 12B arrive avec le
  checkpoint `w4a16-ct` (même famille de vocab 262144 — identité exacte à confirmer en U0).
- **Taille du chantier** : comparable au portage E2B par jalon — plusieurs sessions, nuits
  3090. J1 seul a de la valeur (brique réutilisable, E2B en ~2,5 Go, terrain TurboQuant-poids) ;
  **règle d'arrêt possible après J1**.

## 8. Références

- Checkpoints HF : `google/gemma-4-12B-it-qat-w4a16-ct` (cible J2),
  `google/gemma-4-E2B-it-qat-q4_0-unquantized` (source J1),
  `google/gemma-4-12B-it` (config archi vérifiée 18 juil 2026).
- Méthode & pièges : `docs/DOCUMENTATION.md` (§8 pièges 1-19),
  `docs/G2_3_OP_SENSITIVITY.md` (enveloppe), `docs/BATCHING_RESULTS.md` (T0/S1, HLO md5),
  `docs/ZML_UPSTREAM_AUDIT_2026-07-12.md` (pas de bump ZML — vaut aussi pour ce chantier).
- Écartés au cadrage : justification alambic (T_logits HF NF4 suffit), E4B comme étape
  intermédiaire (le paramétrage moteur U1 en capture l'essentiel sans porter un 3e modèle),
  q4_0/GGUF, quantization maison.
