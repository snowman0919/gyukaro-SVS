Overall status: incomplete experimental milestone; semantic v0.6 acceptance gates A–F are partially evidenced, G/H require clean-package and stronger independent identity/style evaluation
Current version: gyu-singer-v0.6 experimental
Package: gyu-singer-v0.6-incomplete (not promoted as accepted)
Package SHA-256: f70dd240bb9ed753f8e8a527a64af6c3683eb2d3179d6de03f0f8d03f67b6372
Git commit: bd2e2dd45478768cbc6ec74a4e0f9b6ed64ae9a0
Manual verified score phrases: 24 independently transcribed/reviewed rows; no target RMVPE used for score construction
Independent prosody evaluation: completed; v0.6 MAE 58.27c vs nominal 56.75c and correlation 0.6844 vs 0.6846, no decisive gain
Teacher internal paired rows: 191 Fish+MOSS (KO 99, EN 77, JA 15), 382 extracted vectors; Higgs hidden unavailable
Fish internal representation: Fish-S2-Pro-DAC.encoder_hidden, mean pooled
MOSS internal representation: MOSS-Audio-Tokenizer-Nano.encoder_hidden_states, mean pooled
Higgs internal representation: unavailable; waveform evidence only
Shared identity space: 64D checkpoint trained with weighted teacher agreement
Identity adapter affects final audio: yes; WavLM 0.67271 enabled vs 0.64184 zero on same KO phrase
Latent acoustic-style adapter: SoulX gt_decoder_inp gated FiLM; style-zero WavLM 0.60861; calibration-only, not production style claim
Pseudo singing used: v0.5 accepted low-trust corpus only; no new v0.6 candidates admitted
Phrase-level generation: KO/EN/JA smoke WAVs produced
Per-note TTS used: no
Waveform pitch shifting used: no
v0.5 fallback used by v0.6 backend: no; v0.6 prosody checkpoint used, v0.5 retained as baseline
OpenUtau readiness: protocol fields documented; native bridge deferred
Korean: real GYU prosody evidence and phrase render; independent score gain unproven
English: generic singing prosody plus GYU identity/style adaptation; no real GYU EN singing claim
Japanese: generic singing prosody plus GYU identity/style adaptation; no real GYU JA singing claim

The goal is not marked achieved because clean-package reproduction, ECAPA/multi-phrase identity confidence intervals, and a genuinely trained latent style objective remain open.
