# K8 — prédiction PRÉ-ENREGISTRÉE, écrite et committée AVANT de mesurer

> 10 août 2026. Dette K8 (`docs/KVDUMP_RESULTS.md` §6) : le **`0,898 s`** de DC7(ii) est une
> lecture **À CHAUD** — le dump venait d'être écrit ~1 min plus tôt, la lecture des 2,62 GiB a
> très probablement bénéficié du cache de pages. Ce fichier est committé **avant** le premier
> `drop_caches`, pour que la prédiction ne puisse pas être ajustée après coup.

## Les chiffres de référence (publiés, DC7)

| Grandeur | Valeur |
|---|---|
| Atteindre l'état @3 927 positions **par le calcul** | **449,485 s** (compile exclue) |
| Le **restaurer**, lecture à chaud | **0,898 s** (2,62 GiB lus inclus, compile exclue) |
| Gain publié | **×500,5** |
| Taille du dump 4k | **2 818 572 288 octets** (2,62 GiB) |

## Prédiction

Le restore 4k **à froid** tombe entre **3 s et 14,9 s** (2,62 GiB à ~0,2–1 GiB/s de lecture
disque sur la VM, plus le H2D et la reprise déjà comptés dans les 0,898 s).

⇒ **Le gain reste ≥ ×30 sur TOUTE la plage prédite** : `449,485 / 14,98 = ×30,0` exactement. La
claim publiée **C-D (gain ≥ ×30) reste vraie à froid tant que le restore ≤ 14,98 s**.

## Ce qui sera REQUALIFIÉ, et à quelle condition

Le **« ≥ ×130 »** écrit au §6 de `KVDUMP_RESULTS.md` repose sur une hypothèse de 1 GiB/s
(≈ 3,5 s au total). Il **ne tient que si le restore à froid ≤ ~3,457 s** (`449,485 / 130`). Au-delà,
ce chiffre **devra être requalifié** — remplacé par la valeur mesurée, pas défendu.

## Ce qui TUE la claim

Un restore à froid **> 34,6 s** (soit un gain **< ×13**) : la marge de C-D serait alors inférieure
à la moitié de son énoncé, et « ×500 » deviendrait un chiffre de régime chaud uniquement, à
republier comme tel.

## Protocole (pour que la mesure soit lisible)

1. Régénérer le dump 4k (supprimé le 10 août) par un re-run de DC7(i).
2. `sync && echo 3 > /proc/sys/vm/drop_caches` (root ; `sudo -n` est disponible sur la VM —
   le prérequis « accès root » de la dette est donc levé).
3. Run `--load-cache` immédiat, chrono `KVLOAD-PERF` (fenêtre post-compile, comme DC7).
4. **Une mesure à chaud de contrôle** juste après : elle doit retrouver ~0,9 s. Si elle ne la
   retrouve pas, **STOP** — c'est l'instrument qui a changé, et il faut le diffuser avant toute
   requalification (leçon : 2ᵉ requalification du même type ⇒ diff l'instrument, pas la claim).
5. Publication **À CÔTÉ** du 0,898 s, jamais en remplacement : les deux régimes sont vrais.
