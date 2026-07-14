# Evaluation v0.2 report

`scripts/evaluate_baseline_hybrid.py` evaluates identical KO/EN/JA two-note scores using SoulX RMVPE, WavLM reference cosine, Whisper lyric similarity, boundary energy jump, and held-note F0 coefficient of variation. Raw evidence is `artifacts/reports/baseline_hybrid_evaluation.json`; samples are `artifacts/samples/{baseline,hybrid}_{ko,en,ja}.wav`.

| Language | Renderer | F0 correlation | cents MAE | UV accuracy | boundary jump | WavLM | ASR similarity |
|---|---:|---:|---:|---:|---:|---:|---:|
| KO | baseline | unavailable | 232.28 | 0.1587 | 0.3939 | 0.1876 | 0.0000 |
| KO | hybrid | -0.1638 | 1812.01 | 0.6167 | 0.1172 | 0.4844 | 0.0000 |
| EN | baseline | 0.9574 | 53.43 | 0.4127 | 0.0278 | 0.3421 | 0.0000 |
| EN | hybrid | -0.3146 | 1886.69 | 0.2500 | 0.0943 | 0.4112 | 0.1250 |
| JA | baseline | -0.9973 | 715.08 | 0.2857 | 0.8820 | 0.5979 | 0.1429 |
| JA | hybrid | -0.4096 | 1819.80 | 0.2833 | 0.0413 | 0.4179 | 0.1818 |

Hybrid has lower current boundary-energy jump, but worse pitch and insufficient intelligibility. It is executable phrase neural SVS, not v1-quality singing. Japanese output is only a renderer exercise; Japanese support is not claimed.
