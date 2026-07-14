Overall status: v0.6 experimental gates A–H verified; production-quality claims remain bounded
Current version: gyu-singer-v0.6 experimental
Package: gyu-singer-v0.6-experimental (full pinned-cache path smoke passed)
Package SHA-256: 89dd9c18d99c1f3b3bcac27982942649ee8a938ad2f91a7a984e2483e5deef1b
Git commit: 5fc9c0b4400f07d53897f508ee0bd60257f3e366
Manual verified score phrases: 24 independently transcribed/reviewed rows; no target RMVPE used for score construction
Independent prosody evaluation: completed; v0.6 MAE 58.27c vs nominal 56.75c and correlation 0.6844 vs 0.6846, no decisive gain
Teacher internal paired rows: 191 Fish+MOSS (KO 99, EN 77, JA 15), 382 extracted vectors; Higgs hidden unavailable
Fish internal representation: Fish-S2-Pro-DAC.encoder_hidden, mean pooled
MOSS internal representation: MOSS-Audio-Tokenizer-Nano.encoder_hidden_states, mean pooled
Higgs internal representation: unavailable; waveform evidence only
Shared identity space: 64D checkpoint trained with weighted teacher agreement
Identity adapter affects final audio: yes; KO/EN/JA Fish/MOSS/student/none ablations change distributions; student-vs-none mean delta is WavLM +0.01080, ECAPA +0.00067
Latent acoustic-style adapter: SoulX gt_decoder_inp gated FiLM; trained with 32-row weak style classification/invariance and style-zero changes final audio; not real-GYU singing supervision
Pseudo singing used: v0.5 accepted low-trust corpus only; no new v0.6 candidates admitted
Phrase-level generation: KO/EN/JA smoke WAVs produced
Per-note TTS used: no
Waveform pitch shifting used: no
v0.5 fallback used by v0.6 backend: no; v0.6 prosody checkpoint used, v0.5 retained as baseline
OpenUtau readiness: protocol fields documented; native bridge deferred
Korean: real GYU prosody evidence and phrase render; independent score gain unproven
English: generic singing prosody plus GYU identity/style adaptation; no real GYU EN singing claim
Japanese: generic singing prosody plus GYU identity/style adaptation; no real GYU JA singing claim

The package remains experimental: pinned external caches are required, gains are sample-wise rather than significance-tested, and English/Japanese prosody is generic rather than real-GYU supervised.
