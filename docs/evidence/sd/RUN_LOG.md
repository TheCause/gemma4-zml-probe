# Journal des runs GPU — chantier SD

- 24 sept 2026 16:17 CEST — 3090 libérée par Régis (ComfyUI arrêté par lui), VRAM 0 MiB ; début des runs SD.
- 24 sept 2026 16:18 CEST — contrôle avant deploy : VRAM 0 MiB ; rsync à blanc : le deploy supprimerait `examples/rqz/.claude-handoff.md` (distant seul) → deploy suspendu, décision demandée.
- 24 sept 2026 16:19 CEST — deploy sauté (option A : sources distantes identiques par rsync -nc, rien supprimé) ; build gemma4_gen_auto ReleaseFast+CUDA, 302 s, rc=0.
- 24 sept 2026 16:24 CEST — run A1+dump (chemins relatifs) : error: InvalidPath avant compile, aucun calcul GPU ; relancé en chemins absolus.
- 24 sept 2026 16:25 CEST — run A1 + dump HLO de gemma4_gen_auto (sources == main 81d98f8, binaire sha256 3d34c0b0…) : A1 PASS 48/48, compile 13,8 s ; empreinte sha256 9465c21d… (module_0001.zml). Fin 16:26 CEST.
- 24 sept. 2026 16:29 CEST — contrôle avant run 84 : VRAM 0 MiB, aucun processus GPU ; deploy 49+84+sd_cases.json (rsync sans --delete) ; lancement 84_sd0_oracle.py.
- 24 sept. 2026 16:32 CEST — run 84_sd0_oracle.py : C-SD-A PASS (48 lettre × 4, 24 JSON), enable_thinking présent → False ; longueurs lettre 69/77/93, JSON 78/86/102 ; manifest + oracle rapatriés.
- 24 sept. 2026 16:33 CEST — run 1 : model_revisions_in_cache VIDE (scan_cache_dir écarte le dépôt E2B-it comme corrompu : blob de poids absent) → 84 corrigé (repli sur snapshots/ + source notée), log du run 1 gardé en 84_oracle_try1_revs_vides.log ; VRAM 0 MiB ; relance 84.
- 24 sept. 2026 16:35 CEST — run 2 (84 corrigé) : C-SD-A PASS ; révision 905e84b5… ; longueurs lettre 69/77/93, JSON 78/86/102 ; ids et logits identiques au run 1 ; manifest + oracle rapatriés.
- 24 sept. 2026 16:37 CEST — Task 4 : `zig_test` vérifié présent dans rules_zig 0.12.2 du workspace (`zig/defs.bzl`) → cible `sd_policy_test` gardée (pas de repli sur `--policy-selftest`). ÉCART AU PLAN : `deploy_to_3090.sh` NON utilisé (son rsync --delete supprimerait `examples/rqz/.claude-handoff.md`, distant seul, suppression non autorisée) → copie `rsync -a zml_runner/sd_policy.zig zml_runner/BUILD.bazel` SANS --delete. Commande : `./bazel.sh test -c opt --@rules_zig//zig/settings:mode=release_fast //examples/rqz:sd_policy_test` (test pur : pas de flag CUDA ; ⚠ ce changement de flag invalide le cache d'analyse, le prochain build CUDA le reconstruit).
- 24 sept. 2026 16:37:14–16:37:46 CEST — sd_policy_test, stub : FAIL `panic: not implemented` (attendu), rc=3.
- 24 sept. 2026 16:37:54–16:38:08 CEST — sd_policy_test, evaluate implémenté : PASSED, 2/2 tests, rc=0.
- 24 sept. 2026 16:38:14–16:38:29 CEST — mutant argmin : FAILED (argmax_pos0 : got abstain:low_margin want decision:direct ; 2/2 échouent), rc=3.
- 24 sept. 2026 16:38:29–16:38:43 CEST — mutant retiré : PASSED, 2/2 tests, rc=0. Logs : `docs/evidence/sd/policy_test.log`.
- 24 sept. 2026 16:43 CEST — Task 5 : VRAM 0 MiB. ÉCART AU PLAN maintenu : pas de `deploy_to_3090.sh` ; copie `rsync -a zml_runner/gemma4_decide.zig zml_runner/sd_policy.zig zml_runner/BUILD.bazel` SANS --delete.
- 24 sept. 2026 16:43:28–16:44:38 CEST — build `//examples/rqz:gemma4_decide` (build_3090.sh, ReleaseFast+CUDA), 69,9 s, rc=0, compilé au 1er essai ; sha256 binaire 3460bbb403e0e487…
- 24 sept. 2026 16:44:46 CEST — `gemma4_decide x --policy-selftest` : `BUILD: mode=ReleaseFast`, C-SD-F PASS 9/9, rc=0 (`policy_selftest.log`).
- 24 sept. 2026 16:44:47 CEST — test de démarrage, manifest sans `letter[0].label_ids` : `manifest : champ 'label_ids' absent ou invalide`, `error: BadManifest`, rc=1, en 3 ms (avant plateforme et poids) (`bad_manifest.log`).
