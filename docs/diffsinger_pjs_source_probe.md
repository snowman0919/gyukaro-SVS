# DiffSinger PJS score-native source probe

This milestone tests whether a redistributable Japanese singing foundation can replace the phrase-TTS content path. It does not claim a GYU release model.

## Reproducible source

- PJS corpus v1.1: 100 Japanese songs, 48 kHz, 26.857 minutes, CC BY-SA 4.0. The official corpus permits commercial and non-commercial use: <https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus>.
- Human-corrected singing labels: `UtaUtaUtau/pjs-manual-labels@cc08bead6bf2b06e88608a8ece12555bcc720ec9`, CC BY-SA 4.0.
- OpenVPI DiffSinger: `753b7cc622aadf802b3145d7bb8f7df4afa213c4`, Apache-2.0. It is the maintained production-compatible fork of the requested MoonInTheRiver DiffSinger architecture: <https://github.com/openvpi/DiffSinger>.
- The converter uses symlinks and never modifies `data/source/` or the downloaded PJS WAV files.

`scripts/prepare_diffsinger_pjs.py` recreates the transcription CSV, vocabulary remap, train/validation split, and bounded experiment configs. PJS rows 091–100 are held out. The Japanese dictionary contains 37 corpus phones plus DiffSinger's AP/SP symbols. New vowel embeddings use the prior model's non-lexical vowel mean; consonants use the Korean-onset mean. Shared embeddings are bit-identical after remapping.

## Exact OpenUtau stress probe

The local-only probe uses the 1.8207-second four-repeat Japanese line from the real OpenUtau project. It supplies one phrase-level phoneme sequence, explicit phoneme durations, and the canonical 50 Hz score F0. It does not use per-note TTS, waveform pitch shifting, or a phase vocoder. Lyrics, DS files, checkpoints, and generated WAV files remain excluded from Git and packages.

| Candidate | Rapid ASR similarity | Pitch p90 cents | Gross >600 cents | Voiced ratio | Decision |
|---|---:|---:|---:|---:|---|
| 69.1M transferred source, step 1000 | 0.1176 | 18.37 | 0.0000 | 0.9130 | reject: lexical collapse |
| 8.1M compact source, step 3000 | 0.2069 | 1152.78 | 1.0000 | 0.4674 | reject: high-pitch failure |
| Compact speed/pitch augmentation, step 3000 | **0.3038** | **47.65** | **0.0000** | 0.7283 | reject: diction and voicing |
| Consonant-weighted continuation, step 1500 | 0.1538 | 24.53 | 0.0000 | 0.8913 | reject: diction regression |
| Longer unweighted continuation, step 2000 | 0.0667 | 38.42 | 0.0000 | 0.8261 | reject: diction regression |
| Compact source through SoulX GYU reference conversion | 0.0000 | 30.51 | 0.0000 | 0.9783 | reject: content destroyed |

The compact model does learn ordinary Japanese singing: held-out PJS091 reaches ASR similarity 0.5116 versus 0.6818 for the real recording. The failure is concentrated in the extreme rapid-repeat condition, not a fake-success WAV or an F0-only model. Speed/key-shift augmentation fixes the octave-scale pitch error but does not meet the required 0.80 lyric gate.

## Decision

The source gate is **FAIL**. GYU identity adaptation, native DiffSinger/OpenUtau export, package creation, and release remain blocked. No candidate is labeled releaseable or human-approved.

The next admissible path needs a larger permissioned Japanese singing foundation or a redistribution-compatible pretrained Japanese acoustic model, followed by the same exact rapid gate before any GYU adaptation. Repeating these bounded PJS-only continuations is not justified by the measurements.

Machine-readable evidence: `artifacts/reports/diffsinger_pjs_source.json` and `artifacts/reports/diffsinger_pjs_rapid_evaluation.json`.
