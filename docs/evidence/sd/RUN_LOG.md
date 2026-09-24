# Journal des runs GPU — chantier SD

- 24 sept 2026 16:17 CEST — 3090 libérée par Régis (ComfyUI arrêté par lui), VRAM 0 MiB ; début des runs SD.
- 24 sept 2026 16:18 CEST — contrôle avant deploy : VRAM 0 MiB ; rsync à blanc : le deploy supprimerait `examples/rqz/.claude-handoff.md` (distant seul) → deploy suspendu, décision demandée.
- 24 sept 2026 16:19 CEST — deploy sauté (option A : sources distantes identiques par rsync -nc, rien supprimé) ; build gemma4_gen_auto ReleaseFast+CUDA, 302 s, rc=0.
- 24 sept 2026 16:24 CEST — run A1+dump (chemins relatifs) : error: InvalidPath avant compile, aucun calcul GPU ; relancé en chemins absolus.
- 24 sept 2026 16:25 CEST — run A1 + dump HLO de gemma4_gen_auto (sources == main 81d98f8, binaire sha256 3d34c0b0…) : A1 PASS 48/48, compile 13,8 s ; empreinte sha256 9465c21d… (module_0001.zml). Fin 16:26 CEST.
