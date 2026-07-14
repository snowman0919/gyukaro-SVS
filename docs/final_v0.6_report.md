Overall status: audit in progress — not achieved; earlier A–H verification claim withdrawn
Current version: gyu-singer-v0.6 experimental
Package: gyu-singer-v0.6-experimental (unpacked smoke passed with the pinned external model cache)
Package SHA-256: 4abc1b08ccb9e45beb7bb41abc8bc2476cc6093e2b92588e9cad63665c6dc0d5
Git commit: audit worktree, uncommitted
Manual verified score phrases: 24 independently transcribed rows; no target RMVPE used for score construction; not human-certified source MIDI
Independent prosody evaluation: completed; v0.6 MAE 134.41c vs nominal 137.87c, but correlation 0.7283 vs 0.7337 — no consistent gain
Teacher internal paired rows: 191 Fish+MOSS style-conditioned rows (KO 99, EN 77, JA 15), but only 44 unique text/reference groups; Higgs hidden unavailable
Fish internal representation: Fish-S2-Pro-DAC.encoder_hidden, mean pooled
MOSS internal representation: MOSS-Audio-Tokenizer-Nano.encoder_hidden_states, mean pooled
Higgs internal representation: unavailable; waveform evidence only
Shared identity space: 64D checkpoint trained with weighted teacher agreement; semantic-group split corrected and reevaluation in progress
Identity adapter affects final audio: yes, six causal phrase renders change WavLM embeddings; however GYU similarity is not consistently improved (WavLM delta +0.00517, 95% bootstrap −0.00057..+0.01385; ECAPA delta −0.00223)
Latent acoustic-style adapter: SoulX gt_decoder_inp gated FiLM changes final audio (latent-vs-spectral-only WavLM L2 0.001663/0.004911 on two KO phrases); weak 32-row style supervision is insufficient to claim calibrated all-preset control
Pseudo singing used: v0.5 accepted low-trust corpus only; targeted v0.6 expansion is not complete
Phrase-level generation: KO/EN/JA smoke WAVs produced
Per-note TTS used: no
Waveform pitch shifting used: no
v0.5 fallback used by v0.6 backend: no; v0.6 prosody checkpoint used, v0.5 retained as baseline
OpenUtau readiness: protocol fields documented; native bridge deferred
Korean: real GYU prosody evidence and phrase render; independent score gain unproven
English: generic singing prosody plus GYU identity/style adaptation; no real GYU EN singing claim
Japanese: generic singing prosody plus GYU identity/style adaptation; no real GYU JA singing claim

Blocking conditions: only 44 unique text/reference groups, incomplete targeted pseudo-singing expansion, external-cache rather than self-contained packaging, weak style calibration, and no consistent independent-score prosody gain. The package remains experimental; pinned external caches are required, and English/Japanese prosody is generic rather than real-GYU supervised.
