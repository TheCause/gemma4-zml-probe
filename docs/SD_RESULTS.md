# SD — Résultats (généré par scripts/86_sd_report.py, ne pas éditer à la main)

| Contrôle | Verdict | Détail |
|---|---|---|
| C-SD-E | **PASS** | bannière présente dans tous les logs chronométrés |
| C-SD-B | **PASS** | max|Δzc|=4.001e-02, max|Δlse|=1.895e-02 (seuil 0.05) ; marge HF < 0,1 : 0 cas [] |
| C-SD-D | **PASS** | bits identiques rep1/rep2 et rep1/inverse sur les 48 prompts |
| C-SD-F(86) | **PASS** | option runner == option recalculée sur toutes les lignes |
| P1 | **PASS** | médiane(t_read_C+t_policy)=0.471 ms, médiane(t_read_B)=0.478 ms, écart 0.007 ms (seuil 1 ms) |
| P2 | **FAIL** | population 24/24 ; (a) G>0 partout : True ; (b) médiane|G−F|=83.5 ms vs tol 78.3 : False ; (c) médiane part=20.6 % : True |
| Qualité | **DESCRIPTIF** | bascules orig↔rev (bras C) : 4/24 ; format_error bras A : 0/24 ; mass_in min/méd/max = 0.945/1/1 |

## Exactitude par bras et par classe

| Bras | Classe | Justes / total |
|---|---|---|
| A | calculate | 6/6 |
| A | direct | 6/6 |
| A | insufficient | 6/6 |
| A | search | 6/6 |
| B | calculate | 60/60 |
| B | direct | 60/60 |
| B | insufficient | 50/60 |
| B | search | 50/60 |
| C | calculate | 60/60 |
| C | direct | 60/60 |
| C | insufficient | 50/60 |
| C | search | 50/60 |

## Descriptifs

- bras A : médiane 835.2 ms, p95 982.3 ms
- bras B : médiane 667.8 ms, p95 810.1 ms
- bras C : médiane 667.6 ms, p95 811.0 ms
- coûts MESURÉS : c_abs = 8.77 ms/pas (publié 14), c_gen = 8.42 ms/pas (publié 9)
- t_reset médian 6.17 ms (hors temps de décision)
- échauffement (froid) : absorption 528.9 ms sur 49 pas
- compile (run_letter.log) : 13.818s
- compile (run_json.log) : 14.149s
- bras B hors étiquettes : 0/240
- rendu du template : enable_thinking passé à False

## Couverture / erreur du bras C en fonction de θ (descriptif, spec §4.4)

| θ | décisions gardées | erreurs parmi gardées |
|---|---|---|
| 0.0 | 240/240 | 20/240 |
| 0.1 | 240/240 | 20/240 |
| 0.2 | 240/240 | 20/240 |
| 0.4 | 240/240 | 20/240 |
| 0.6 | 240/240 | 20/240 |
| 0.8 | 240/240 | 20/240 |

## Gain par cas (bras A − bras C, ms)

| Cas | len JSON | len lettre (moy.) | n_gen A | G mesuré | F prédit | part de t_A |
|---|---|---|---|---|---|---|
| calculate-01 | 92 | 83.0 | 16 | 174.8 | 261.0 | 19.4 % |
| calculate-02 | 102 | 93.0 | 16 | 161.1 | 261.0 | 16.4 % |
| calculate-03 | 87 | 78.0 | 16 | 180.1 | 261.0 | 20.9 % |
| calculate-04 | 87 | 78.0 | 16 | 178.6 | 261.0 | 20.7 % |
| calculate-05 | 86 | 77.0 | 16 | 165.8 | 261.0 | 19.5 % |
| calculate-06 | 102 | 93.0 | 16 | 177.0 | 261.0 | 17.9 % |
| direct-01 | 80 | 71.0 | 16 | 181.9 | 261.0 | 22.7 % |
| direct-02 | 81 | 72.0 | 16 | 177.9 | 261.0 | 22.0 % |
| direct-03 | 81 | 72.0 | 16 | 176.7 | 261.0 | 21.8 % |
| direct-04 | 82 | 73.0 | 16 | 169.6 | 261.0 | 20.9 % |
| direct-05 | 84 | 75.0 | 16 | 167.2 | 261.0 | 20.1 % |
| direct-06 | 84 | 75.0 | 16 | 178.9 | 261.0 | 21.4 % |
| insufficient-01 | 78 | 69.0 | 8 | 113.4 | 189.0 | 15.8 % |
| insufficient-02 | 79 | 70.0 | 8 | 112.4 | 189.0 | 15.5 % |
| insufficient-03 | 79 | 70.0 | 8 | 97.6 | 189.0 | 13.5 % |
| insufficient-04 | 81 | 72.0 | 17 | 180.2 | 270.0 | 22.0 % |
| insufficient-05 | 81 | 72.0 | 8 | 103.0 | 189.0 | 13.9 % |
| insufficient-06 | 80 | 71.0 | 17 | 189.9 | 270.0 | 23.4 % |
| search-01 | 86 | 77.0 | 12 | 149.7 | 225.0 | 18.2 % |
| search-02 | 91 | 82.0 | 16 | 170.3 | 261.0 | 19.0 % |
| search-03 | 87 | 78.0 | 16 | 179.3 | 261.0 | 20.8 % |
| search-04 | 87 | 78.0 | 16 | 174.7 | 261.0 | 20.4 % |
| search-05 | 88 | 79.0 | 16 | 182.8 | 261.0 | 21.1 % |
| search-06 | 87 | 78.0 | 16 | 184.4 | 261.0 | 21.3 % |
