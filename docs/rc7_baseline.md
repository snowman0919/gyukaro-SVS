# RC7 baseline

Status: accepted experimental baseline. This is not production-ready and is not `v1.0.0`.

## Revisions and runtime

- Source commit: `ae8944070f3dc38e310b33f29d95f4bcd3c81def`
- SoulX-Singer: `81aeb3ae772c70093c3de74dc23c92d983801ae4`
- OmniVoice: `1574e06a767808c9343740ba695e7515c3d484e2`
- OpenUtau upstream/overlay target: `stakira/OpenUtau@27573ac5c888d927119d5f65a207312d79194b1f`
- OpenUtau integration source revision: same source commit as RC7
- Phrase path: RC6 canonical phone/score timeline and SoulX FP32 decode, followed by the RC7 spectral singing refiner at strength `0.5`
- SoulX policy: standard `32 steps / CFG 1.5`; rapid `64 / 2.0`; large interval `32 / 2.0`, seed `21`
- Output: mono 48 kHz PCM-24; no per-note TTS, phase vocoder, or waveform pitch shifting
- RC7 was evaluated as a fixed post-RC6 candidate, not exposed as a production backend alias.

## Checkpoints

| Role | Path | SHA-256 |
|---|---|---|
| Universal spectral parent | `checkpoints/acoustic_refiner_spectral_universal.pt` | `cdaf12342a5aa1cc24b75b609bf8ac4f6f80c09b9d54daed9c58aad0670464a5` |
| Selected singing refiner | `checkpoints/acoustic_refiner_spectral_singing.pt` | `8814e14e3d13400f5098920bc24e6decc080848aa177310f766d50e74d33b7a4` |
| Rejected GYU adapter control | `checkpoints/acoustic_refiner_spectral_gyu.pt` | `a0c8159803653ee31855f1c882f3013d8aa1562cc138ba42b16dc2d0ad828102` |

The selected model is the identity-initialized STFT-mask U-Net with `n_fft=1024`, `hop_length=256`, 16/64 channels, six bottleneck blocks, adapter rank 8, and bounded log gain 0.8. The GYU adapter was not selected.

## Objective result

Nine-file aggregate, RC6 -> RC7:

| Metric | RC6 | RC7 |
|---|---:|---:|
| ASR lyric similarity | 0.924211 | 0.924211 |
| Pitch MAE, cents | 8.468889 | 8.241111 |
| Voicing accuracy | 0.872033 | 0.872711 |
| HF spike p99/median | 458.558778 | 344.181322 |
| Sample jump p99.9 | 0.093596 | 0.073507 |
| WavLM-to-GYU | 0.599993 | 0.617891 |
| ECAPA-to-GYU | 0.112046 | 0.118516 |

Authoritative metrics: `artifacts/reports/acoustic_refiner_spectral_stress_evaluation.json`.

## Listening files

| Case | Path | SHA-256 |
|---|---|---|
| KO neutral | `artifacts/reports/rc7_listening_gate/01_ko_neutral.wav` | `92eac775dcb78e9e77045ccad4ab92423c53701a16b5280f9d5fd293e3af0cdd` |
| KO breathy | `artifacts/reports/rc7_listening_gate/02_ko_breathy.wav` | `21b8ce491b2d036e8791ab497c163f561920a40abe6a226e2063c4db93a06b7a` |
| KO energetic | `artifacts/reports/rc7_listening_gate/03_ko_energetic.wav` | `62676c9207237d64238dd40837ed88219aa9e3b88f8bbc0d9e54316532cb7b01` |
| EN | `artifacts/reports/rc7_listening_gate/04_en.wav` | `6aa89bcf51d27fe42561f4359f2e0f7575a51e7f70412a4d3e4482c94f633693` |
| JA | `artifacts/reports/rc7_listening_gate/05_ja.wav` | `d283f1611cff2e48bbb4b1a38d4bbf6e638268a9581db7b05610100dd5e2a22f` |
| Rapid KO | `artifacts/reports/rc7_listening_gate/06_rapid_ko.wav` | `2774ec3eaefa19c9a024467dfcbdbcc6cde6f9f144956d27856462f4637b8380` |
| Sustained KO | `artifacts/reports/rc7_listening_gate/07_sustained_ko.wav` | `db44d39e3b4b701f68bf44b0f2ed11600bb7e174b03f6e9710e154593fa34310` |
| Large interval KO | `artifacts/reports/rc7_listening_gate/08_large_interval_ko.wav` | `a7e0aa543a3a1a8afc3e7d3d5655953db1b30057e714fe3d1ddd3eca0284b50e` |
| Phrase boundary | `artifacts/reports/rc7_listening_gate/09_phrase_boundary.wav` | `e158044e7fb5c009a213ecc4ad02104c56c3d5bc3cc22db4a2a0de1d2c594b0b` |

The Git commit and hashes are the byte-for-byte preservation boundary. RC8 must write to new paths.

## Human observations

- RC7 is clearly better than RC6, especially on headphones.
- Rapid KO is practically usable and is a protected non-regression case.
- Remaining: sustained-note noise, unnatural English transitions, excessive Korean continuity, occasional weak/unnatural Japanese phonemes, and an initial dual-trajectory mechanical artifact in Large Interval KO.
