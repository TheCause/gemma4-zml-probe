# Runtime autonome — Design (spec approuvée — IMPLÉMENTÉE, 4 gates rendus)

> **Statut** : spec validée par Régis le 10 juil 2026 (brainstorming 4 sections, 4/4 OK) ;
> **implémentée les 10-11 juil** (branche `gen-autonome`, exécution subagent-driven).
> **But** : un binaire ZML texte→texte SANS oracle HF à l'usage — le banc, lui, reste validé
> CONTRE HF (les gates ci-dessous). Ferme l'item backlog « Runtime 100 % autonome ».

## Résultats (10-11 juil 2026)

| Gate | Verdict | Mesures clés |
|---|---|---|
| A0 tokenizer+template | **PASS** (tag `gate/gen-auto-a0-pass`) | ids == HF bit-exact sur 2 prompts ; round-trip détok ; ⚠ tokens de tour RÉELS `<|turn>`/105, `<turn|>`/106 (pas start/end_of_turn) ; EOT extrait = 106 (piège : lookup `<end_of_turn>` → `<unk>` silencieux) |
| A1 prefill-par-decode | **PASS** (tag `gate/gen-auto-a1-pass`) | **48/48 == HF, autonome complet, zéro input fixture** ; backend cuda, compile mono forwardStep GPU 16.7-17.4 s (risque nommé : non matérialisé), 75-94 tok/s |
| A2 bout-en-bout long | **critère pré-enregistré N/N : FAIL — requalifié PASS au critère différentiel (décision Régis 10 juil)** | Autonome : 916/999, 1re bifurcation step gen 590 (pos 615), marge top1-top2 **0.006** (15.1260 vs 15.1197). **Contrôle replay (inputs HF exacts) : MÊME bifurcation step 589, mêmes tokens (12266 vs 25680), 997/999** (le replay se resynchronise par steps forcés ; l'autonome propage). Lecture : l'autonomie n'ajoute AUCUNE dégradation mesurable (590 steps parfaits ≥ replay) ; c'est ZML-fp32 vs HF-fp32-CUDA qui ne tient pas le N/N sur séquence à marges fines — le 1020/1020 de S46 était une propriété de S46. **Critère requalifié : A2-diff = « autonome ≥ replay sur la même fixture » → PASS (590 ≥ 589)** ; le FAIL du critère original est publié ici même (null-result, pattern G2.0 : mesurer ce que la référence s'autorise) |
| A3 early-stop EOS | **PASS** (tag `gate/gen-auto-a3-pass`) | Free-run : stop après exactement **i+2 = 2 tokens** (EOT à l'index 0 d'`expected`), dernier = EOT ; EOT strippé avant détok ; **stdout : `réponse : "Paris"`** — pipeline texte→texte complet |
| Non-régression | **PASS** (11 juil) | E1 4/4 (`[1018,6398,25967,53121]` == decode4 == HF) ; replay GPU 48/48 — le moteur n'a pas bougé d'un octet |
| Non-vacuité | **PASS** (11 juil) | Template perturbé (`\n` après `user` retiré) → ids `{2,105,2364,3689,…}` (20 ids, le 107 disparaît) ≠ référence `{2,105,2364,107,…}` (21 ids) — le gate A0 discrimine ; restauration vérifiée (ids == référence, round-trip PASS, tree propre) |
| Validation réelle (post-merge) | **PASS** (11 juil, Régis) | Prompt libre FR hors fixtures → 110 tokens, early-stop EOT naturel, réponse correcte stdout, 54,5 tok/s (prefill inclus). Incident opérationnel : OOM VRAM (Hermès/Ollama 22/24 Go) → garde-fou contention documenté au PLANNING |

**Piège découvert (coût réel : 1 run tué en thrash)** : sans le flag de build
`--@zml//platforms:cuda=true`, `bazel run` ET l'exécution directe du binaire retombent en
**CPU silencieux** (libpjrt_cuda absente des runfiles) → compile/boucle XLA-CPU du mono =
mur mémoire. Mitigé en dur : `error.CudaRequired` sauf `--allow-cpu` (débogage), ligne de
log `backend = …` obligatoire. Cf ENGINE_LOG.md § Validation GPU 28 juin (cause racine).

## 1. But et périmètre (décisions de cadrage)

| Décision | Choix | Alternative écartée |
|---|---|---|
| Périmètre | **Texte→texte complet** : tokenizer + prefill + decode + EOS + détok dans le binaire ; zéro Python, zéro fixture par prompt (poids + `tokenizer.json` suffisent) | Decode autonome seul (fixture par prompt restait nécessaire) ; streaming stdout (hors scope) |
| Cible | **GPU 3090 seul** (le régime validé : fp32, 109 tok/s). Moteur device-agnostic — le portage M4 sera un chantier ultérieur | CPU (350× plus lent, pas d'usage) ; M4 direct (mélange deux chantiers) |
| Prefill | **A. Prefill-par-decode** : le graphe decode S=1 validé sert de prefill (tokens du prompt injectés un à un, argmax ignoré) | B. Vrai prefill S>1 (graphe neuf à valider pour gagner ~0,5 s/prompt — YAGNI, ouvrable plus tard) |
| Sampling | **Greedy seul** (invariant du banc : oracle HF greedy déterministe) | — |

**Ce qui existe déjà** (rien à refaire) : moteur `EngineModel` validé E1/E2 + G2/G2.3 ;
decode GPU mono 1020/1020 == HF (`gemma4_gen_long_gpu`, replay) ; autonomie host prouvée
côté CPU chunké (`gemma4_gchunk_auto` : argmax → gather streaming → réinjection).
**Ce chantier = porter l'autonomie sur le runner GPU + fermer les deux bouts texte.**

## 2. Architecture et flux

Un nouveau runner **`gemma4_gen_auto.zig`** (nom court — quota comptime pjrt, piège connu).
Le moteur et ses graphes ne changent **pas d'un octet** — tout est host-side.

```
prompt (CLI) → chat template Gemma (Zig) → tokenizer ZML → ids[]
→ boucle unique sur le graphe decode S=1 :
     gather embeds host (lignes BRUTES bf16, streaming safetensors) → forwardStep GPU
     → logits → argmax host
     · i < len(ids)-1  (prefill) : token suivant injecté = ids[i+1], argmax ignoré
     · sinon (génération)        : token suivant injecté = argmax
     · arrêt : argmax == EOS  OU  steps == --max-tokens
→ ids générés → décodeur ZML → texte sur stdout
```

- Moteur : `EngineModel(struct{}, .{ .two_masks = true, .kmax_sliding = 1024, .kmax_full = 1024 })`,
  fp32 (PrecRt défaut), identique à `gemma4_gen_long_gpu`. **Point d'entrée compilé =
  `forwardStep` (engine.zig:632)** — l'entrée créée pour l'autonomie L2, qui prend les
  embeds token-dépendants (PAS `forward(Packed)` du replay, dont les embeds viennent du
  buffer packé de la fixture).
- **Inputs host par step — code Zig NEUF à écrire** (le L2 CPU actuel les lit de la
  fixture, il n'existe pas de pattern host à reprendre) : cos/sin **full** (RoPE
  theta=1e6, hd=512, **partial rotary 0.25, scaling "proportional"** — la formule exacte
  de l'oracle 46/49 `Gemma4TextRotaryEmbedding(layer_type="full_attention")`, PAS un RoPE
  standard ; le RoPE sliding est calculé in-graph depuis la position, engine.zig:112-114),
  positions, masques bande/causal, et **cache initial à zéros** (les runners existants
  chargent tous un `cache0` prefill HF depuis la fixture).
- **Champs `Packed(true)` non consommés** : `forwardStep` n'utilise pas `embeds`/`embptls`
  du `Packed`, mais la struct les déclare — le runner devra les fournir (buffers factices
  ~21 Mo bf16, ou variante allégée à trancher au plan). À budgéter, pas à découvrir.
- CLI : `gemma4_gen_auto <model.safetensors> <tokenizer.json> --prompt "…" [--max-tokens 200]`.

## 3. Composants

1. **Tokenizer** : `zml.tokenizer.Tokenizer.fromFile(allocator, io, tokenizer.json)` —
   module natif ZML (`zml/tokenizer`, utilisé par `examples/llm`). Encoder pour le prompt,
   decoder pour la sortie. Prérequis vérifiable au plan (spike) : `tokenizer.json` de
   `google/gemma-4-E2B-it` présent dans le cache HF de la 3090 **ET le module ZML le
   parse** (format SentencePiece/Gemma) — c'est le premier step du plan, avant tout
   le reste.
2. **Chat template** : reproduit en Zig (il ne vit PAS dans `tokenizer.json`) —
   **mesuré (Task 1, repr() HF)** : `<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n` —
   les tokens de tour de CE tokenizer sont `<|turn>` (105) / `<turn|>` (106), pas les
   `<start_of_turn>`/`<end_of_turn>` classiques de la famille Gemma. Conformité vérifiée
   caractère près contre l'oracle 49 (**gate A0**).
3. **Gather embeds host** : lecture streaming depuis `model.safetensors` (le pattern qui a
   résolu l'OOM du L2 CPU, repris de `gemma4_gchunk_auto`) : par token, lignes
   `embed_tokens[id]` et `embed_tokens_per_layer[id]` injectées **BRUTES (bf16, sans
   scaling)** — les scalings ScaledWordEmbedding ×√1536 et ×16 sont DÉJÀ dans le graphe
   (`EMBED_SCALE` engine.zig:644, `SQRT_PLE` engine.zig:535 ; doc `forwardStep` : « AVANT
   scale √1536, brut »). Les appliquer host serait un DOUBLE scaling → divergence garantie.
4. **EOS** : id du token de fin de tour extrait du tokenizer au démarrage (pas hardcodé).
   **Mesuré (Task 1)** : `<turn|>` = **106**. ⚠ Piège consigné : `convert_tokens_to_ids("<end_of_turn>")`
   retourne silencieusement `<unk>` (3) — ce token n'existe pas dans ce vocab ; extraire
   l'EOT du champ `eot_token` de la special_tokens_map, jamais par lookup de chaîne supposée.
   Early-stop + garde `--max-tokens` (défaut 200) + garde
   `len(prompt_ids) + max_tokens ≤ L_MAX` (sinon erreur claire au lancement).
5. **Détokenisation** : decoder ZML sur les ids générés ; round-trip sanity (re-encode du
   texte produit == ids) en mode verbose.

## 4. Gates (un commit/tag par gate — pattern du projet)

| Gate | Critère PASS | Oracle |
|---|---|---|
| **A0 — tokenizer+template** | ids ZML (chat template inclus) == ids HF sur ≥2 prompts ; round-trip détok OK | dump ids de `49_gen_custom_oracle.py` |
| **A1 — prefill-par-decode** | prompt de réf (« What is the capital of France? Answer in one word. », 48 tokens de génération) : **48/48 == HF** en partant du prompt injecté token par token (plus AUCUN input de fixture) | fixture 49 recyclée en **oracle de comparaison** (`expected`), plus jamais en source d'inputs |
| **A2 — bout-en-bout long** | fixture 49 **longue** (prompt court templaté ~13 tokens, `--n-tokens` ≈ 1010, jusqu'à L_MAX) : **N/N == HF** en autonome complet. NB : S46 est inatteignable en texte pur (son prompt = ids bruts `[2,105,2048,4095]` sans antécédent par le chat template) — S46 reste couvert par la non-régression replay | fixture 49 longue (`expected`) |
| **A3 — early-stop EOS** | sur un prompt court : la génération s'arrête à l'index du premier `<end_of_turn>` dans `expected` (l'oracle 49 ne s'arrête PAS à EOS, il génère `n_tokens` fixes — le critère se lit dans `expected`, pas dans la longueur de la fixture) | fixture 49 (`expected`) |
| **Non-régression** | `gemma4_gen_long_gpu` (replay) et E1 re-PASS — les runners replay restent les oracles du banc | HLO/tokens existants |
| **Non-vacuité** | chat template perturbé (ex. `\n` manquant) → A0 ou A1 FAIL — le gate discrimine | — |

## 5. Risques et limites assumées

- **Drift prefill batch vs token-par-token** : HF préfille en une passe (S=48), nous en 48
  passes S=1 — mêmes maths causales, ordre de matmuls différent → drift numérique possible
  sur les caches. Critère = argmax-match (a déjà encaissé ce type de drift : fp32 GPU
  1020/1020) ; **si A1 FAIL, diagnostic au niveau LOGITS** (leçon méthodo : argmax trop
  robuste pour diagnostiquer, comparer les logits).
- Greedy seul, fp32, L_MAX=1024, batch-1 : les régimes validés du banc, rien au-delà.
- Prefill à ~109 tok/s (≈0,5 s pour 50 tokens) : assumé (approche A) ; l'optimisation
  (vrai prefill S>1) est un chantier ultérieur documenté §1.
- **Compile `forwardStep` mono sur GPU : jamais fait** (seul le chunké CPU l'a compilé).
  Risque faible : `gemma4_gen_long_gpu` compile déjà le mono `.forward` 35 couches
  op-identique sur GPU sans incident — le « ~33 Go, thrash » de la doc engine.zig:667
  était le compile **XLA-CPU** sur la VM 23 Go, pas un plafond GPU. Si surprise : le
  précédent chunké existe.
- Le chat template Zig est spécifique Gemma 4 chat mono-tour (pas de multi-tour, pas de
  system prompt) — YAGNI, extensible au besoin.

## 6. Hors scope

Streaming stdout ; sampling non-greedy ; bf16 (config G2.3 disponible mais chantier
distinct) ; multi-tour ; portage M4/macOS (alambic ch. 11, chantier ultérieur) ;
batching/flash-attention ; L3 in-graph.

## 7. Références

- Moteur & gates existants : `docs/ZML_MODULAR_ENGINE_DESIGN.md`, `docs/GENERATION_LONGUE_PLAN.md`
  (L2 : `gemma4_gchunk_auto`, gather streaming), `docs/GPU_PORT_PLAN.md` (runner GPU mono).
- Oracles : `scripts/46_gen_long_oracle.py` (S46), `scripts/49_gen_custom_oracle.py`
  (prompt custom + chat template), `scripts/48_detokenize.py` (détok + round-trip, rendu
  optionnel par ce chantier).
- Tokenizer ZML : `zml/tokenizer/main.zig` (workspace 3090), usage `examples/llm/main.zig:123-175`.
- Contexte 3090 : `ssh ia@192.168.1.163`, workspace `/data/rqz_workspace/zml/examples/rqz/`,
  checkpoint `/data/gemma4-zml-probe/weights/model.safetensors`, deploy `zml_runner/deploy_to_3090.sh`,
  build `./bazel.sh` (piège : cible au nom court ; patch local `@setEvalBranchQuota` pjrt.zig).
