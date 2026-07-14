Overall status: incomplete experimental milestone; semantic v0.6 acceptance gates A–F are partially evidenced, G/H require clean-package and stronger independent identity/style evaluation
Current version: gyu-singer-v0.6 experimental
Package: not promoted (no clean v0.6 package yet)
Package SHA-256: n/a
Git commit: recorded after this report is committed
Manual verified score phrases: 24 independently transcribed/reviewed rows; no target RMVPE used for score construction
Independent prosody evaluation: completed; v0.5 does not beat nominal verified score in aggregate
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
v0.5 fallback used by v0.6 backend: no; v0.5 prosody checkpoint retained
OpenUtau readiness: protocol fields documented; native bridge deferred
Korean: real GYU prosody evidence and phrase render; independent score gain unproven
English: generic singing prosody plus GYU identity/style adaptation; no real GYU EN singing claim
Japanese: generic singing prosody plus GYU identity/style adaptation; no real GYU JA singing claim

The goal is not marked achieved because clean-package reproduction, ECAPA/multi-phrase identity confidence intervals, and a genuinely trained latent style objective remain open.
