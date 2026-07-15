Overall status: RC5 human listening PASS; final `v1.0.0` remains unpublished
Highest truthful release: `v1.0.0-rc.5` prerelease
Source commit: `d94d8b3eac8556332f11fa55756e769d7a31e044`
OpenUtau upstream commit: `27573ac5c888d927119d5f65a207312d79194b1f`
Git tag: `v1.0.0-rc.5`; final `v1.0.0` not created
GitHub release URL: `https://github.com/snowman0919/gyukaro-SVS/releases/tag/v1.0.0-rc.5`
Package: `artifacts/package/gyu-singer-v1.0.0-rc5.zip`
Package SHA-256: `baffa2a24ae7aa42394d17f176d486643240fdfd8e6fe297c5082b4582e3b7ac`
Human listening: PASS on immutable candidate4 nine-case set and matched RC4 clips
Clean package install: PASS
Long-form OpenUtau render: PASS; 119.983 seconds, 136 notes, 17 phrases, KO/EN/JA
Per-note TTS: not used
Waveform pitch shifting: not used
RC4 baseline: preserved as backend `gyu-singer-v0.8` and tag `v1.0.0-rc.4`

# RC5 artifact-fix report

## Artifact source

Primary source: RC4 forced nonzero F0 through silence and unvoiced consonants while OmniVoice content timing did not follow score phoneme timing. This content/F0 conflict produced metallic buzz, smeared transitions, and collapse around fast syllables and large intervals.

Secondary sources: 16-step, high-CFG SoulX decode increased discontinuities; hard note steps worsened unedited transitions. Controlled identity/style ablations did not identify either adapter as primary source.

## Changes

- One canonical 50 Hz timeline now carries phoneme, note, voicing, nominal F0, GYU residual, and user pitch.
- Silence and unvoiced consonants receive F0=0. Voiced regions receive score F0, validated GYU prosody, and authoritative OpenUtau pitch.
- Only explicit slurs without an editor pitch curve receive a minimal 60 ms transition. Hard attacks remain hard.
- Rapid Korean uses MMS CTC phoneme-hold mapping inside SoulX content hidden state. English uses 0.25 CTC latent timing correction.
- SoulX uses FP32. Standard KO/EN uses 32 steps, CFG 1.5; rapid uses 64, CFG 2.0; large intervals use 32, CFG 2.0, seed 21. Other stress/style cases use 64, CFG 2.0.
- Output peak safety gain is applied only above 0.97. No denoise, dereverb, UVR, phase vocoder, or waveform pitch shift was added.

Identity and latent style adapters remain enabled. Same-input adapter gates showed no primary artifact responsibility, and the approved stress set includes neutral, breathy, and energetic outputs. RC4 remains callable and unchanged.

## Before/after results

Core four versus RC4:

- high-frequency spike ratio: -15.3%;
- spectral flux p95: -7.0%;
- sample jump p99.9: -24.4%;
- pitch MAE: +1.81 cents;
- ASR similarity: -0.025;
- clipping: 0 for all nine RC5 stress files;
- lyric coverage: at least 0.8 for all nine.

Objective diagnostics did not decide listening quality. Human listening explicitly passed the exact hashes in `artifacts/reports/rc5_stress_candidate4/human_review.json`.

Listening files:

- after: `artifacts/reports/rc5_stress_candidate4/listening/`;
- matched RC4/RC5 clips: `artifacts/reports/rc5_stress_candidate4/before_after/`;
- measurements: `artifacts/reports/rc5_stress_candidate4/evaluation.json`.

## Runtime and package proof

The exposed `gyu-singer-rc5` backend reproduced KO neutral, EN, rapid KO, and large-interval KO approved WAVs byte-for-byte. Clean package KO/EN/JA smoke outputs also matched approved hashes exactly.

The final archive was unpacked outside the repository. Installer verification covered pinned OmniVoice, SoulX, RMVPE, MMS, project checkpoints, and OpenUtau. Real installed renders produced 48 kHz mono KO/EN/JA WAVs. A restored xUnit integration test rendered the complete 17-phrase project with zero failures or retries, then hit all 17 cache entries. Public GitHub re-download matched package SHA-256.

## Remaining defects

- Rapid Korean retains elevated high-frequency energy.
- English ASR hears “Sing” as “Sink” in one stress phrase.
- Isolated sustained “아” is transcribed as “아멘” despite full lyric coverage.
- Large-interval target voicing accuracy remains 0.65 even though human listening passed the selected output.
- Personalized prosody evidence remains Korean-only. EN/JA use generic multilingual prosody plus GYU identity/style.
- Linux NVIDIA CUDA is the only release-tested platform. Initial pinned download is roughly 10 GB.

## Release recommendation

RC5 is suitable as the next experimental release candidate and is clearly more stable than failed RC4. Do not infer production-grade quality from this pass. Final `v1.0.0` remains unpublished; resume final release only as a separate explicit decision after broader use confirms no regression.

## Evidence

- `artifacts/reports/rc5_isolation/matrix.json`
- `artifacts/reports/rc5_stress_candidate4/manifest.json`
- `artifacts/reports/rc5_stress_candidate4/evaluation.json`
- `artifacts/reports/rc5_stress_candidate4/human_review.json`
- `artifacts/reports/rc5_runtime_smoke/verification.json`
- `artifacts/reports/rc5_package_smoke.json`
- `docs/rc4_artifact_isolation.md`
- `docs/rc5_reference_audit.md`
- `docs/rc5_audio_quality_candidate.md`
