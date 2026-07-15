# External dataset licenses

Raw external audio is never committed or bundled.

| Dataset | License | Decision | Distribution |
|---|---|---|---|
| LibriTTS-R | CC BY 4.0 | selected: dev_clean only; quality-filtered manifest, not full 585-hour corpus | allowed with attribution; checkpoint may be distributed |
| VocalSet 1.2 | CC BY 4.0 | selected: filtered scales, arpeggios, long tones, and excerpts; no blind full-corpus training | allowed with attribution; checkpoint may be distributed |
| Zeroth-Korean SLR40 | CC BY 4.0 | selected: four speakers, 100 utterances each; Korean phoneme/acoustic prior only, not singing | allowed with attribution; checkpoint may be distributed |
| Emilia original | CC BY-NC 4.0 plus gated terms | excluded: none | excluded from production checkpoint to avoid NC/copyright ambiguity |
| Emilia-YODAS | CC BY 4.0 | deferred: none; 2.1 TB source and raw-origin risk make it unnecessary for the first bounded experiment | potentially allowed after per-item provenance review |
| JVS | custom non-commercial research/personal-use terms | excluded: none | excluded from production checkpoint without commercial permission |
| GTSinger | CC BY-NC-SA 4.0 plus indemnity terms | excluded: none | excluded so production checkpoint is not forced into NC-SA terms |
| Children's Song Dataset | CC BY-NC-SA 4.0 | excluded: none | excluded from the redistributable production checkpoint |
| JVS-MuSiC | custom non-commercial research/personal-use terms | excluded: none | excluded from production checkpoint without commercial permission |
| SingNet | no compatible released-data license verified | excluded: none | excluded until an official compatible data/model license is verified |

Evidence URLs and exact use restrictions are preserved in `dataset_registry.json`.
