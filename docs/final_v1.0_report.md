Overall status: RC4 human listening failed; candidate4 passes objective stress gates and awaits human listening
Highest truthful release: failed public v1.0.0 RC4 prerelease; candidate4 is not tagged or released
Source commit: `ce7a1e088de0b38777551ba583c993817dc9d020` (validated RC4 source; report changes follow)
OpenUtau upstream commit: `27573ac5c888d927119d5f65a207312d79194b1f`
Git tag: `v1.0.0-rc.4`; final `v1.0.0` pending
GitHub release URL: `https://github.com/snowman0919/gyukaro-SVS/releases/tag/v1.0.0-rc.4`; final release pending
Package: `artifacts/package/gyu-singer-v1.0.0-rc4.zip`
Package SHA-256: `507e51cc9199a248f217eb359b2790e7f56797ccfc15b3f639cc6ad5fb2d2702`
Fresh OpenUtau clone: pass, official `stakira/OpenUtau` master
OpenUtau build: pass; unmodified upstream 205/206 tests, one reproduced unrelated upstream JaPresamp failure
Native GYU renderer: pass, 7/7 mappings plus actual resident and RenderMixdown tests
Fresh install: pass in `/tmp`, no source `PYTHONPATH`; pinned cache imported only through documented checksum/revision-verifying installer option
Long-form render: pass, 119.983 seconds, 136 notes, 17 phrases, KO/EN/JA
Phrase-boundary quality: pass, all 16 boundaries
KO: pass; personalized real-GYU prosody, mean ASR lyric similarity 1.000
EN: pass; generic multilingual prosody plus GYU identity/style, mean ASR lyric similarity 0.9449
JA: pass; generic multilingual prosody plus GYU identity/style, mean ASR lyric similarity 0.6826
Note pitch edit: pass, actual OpenUtau renderer +2 semitones produced +201.58 cents
User pitch edit: pass, actual OpenUtau renderer +1 semitone PITD produced +102.86 cents
Lyric edit: pass, actual OpenUtau renderer changed Whisper from “하늘빛 노래 불러요” to “사랑을 담아서 마음을 전해요”
Breathy: objective proxy pass; RC4 human quality review failed overall
Energetic: objective proxy pass; RC4 human quality review failed overall
Repeated rendering: pass, 20/20 identical hashes; long-form cache rerender 0.104 seconds
Memory stability: pass; repeated stress growth -1313.66 MiB, long-form peak unified-memory growth 15.40 GiB
Public release re-download verification: RC4 bytes verified but audio quality failed; final v1.0.0 prohibited

Post-RC4 candidate: `artifacts/reports/rc5_stress_candidate4/`; objective candidate, human pending
Post-RC4 decoder: measured 32-step/CFG1.5 standard path; 64-step/CFG2 rapid path; 32-step/CFG2 large-interval path
Post-RC4 core delta vs RC4: HF-spike -15.3%, spectral flux -7.0%, sample jump -24.4%, pitch MAE +1.81 cents, ASR similarity -0.025

## What changed from v0.9

v1.0 validates the native renderer against a fresh official OpenUtau clone, fixes real phrase loading and cache invalidation, hardens the resident service, adds a realistic two-minute project and boundary gates, and replaces the manual development setup with a pinned installer and launcher. RC testing found and fixed three distribution defects: downloader invocation, OpenUtau project path, and omitted English lexicon package data. A completion audit also replaced split mapping/direct-HTTP behavior evidence with audio measured after the real C# `GyuSingerRenderer.Render` path.

## Production path

OpenUtau sends a multi-note `RenderPhrase` to `GyuSingerRenderer`. The resident `gyu-singer-v0.8` backend combines score, editor pitch, the validated GYU prosody controller, multilingual OmniVoice phrase content, real SoulX latent identity/style conditioning, and the SoulX phrase decoder. OpenUtau alone performs its normal 48 kHz phrase-to-44.1 kHz editor conversion and mix export. Per-note TTS and waveform pitch shifting are not used, and v0.9 is not a silent fallback.

## OpenUtau and installation

The supported flow is `./install.sh`, then `./launch-openutau.sh examples/openutau_v10_longform.ustx`. The installer creates `.runtime/`, verifies pinned OmniVoice/SoulX revisions and checkpoint hashes, installs the packaged project checkpoints, automatically clones and overlays OpenUtau, builds it, and renders KO/EN/JA 48 kHz smoke WAVs. Training teachers Fish, MOSS, and Higgs are absent from inference.

The exact RC4 archive was unpacked outside the repository into a new runtime. All three language smokes passed. Its installed OpenUtau rendered the complete long-form project with 17 cache misses followed by 17 hits, zero failed phrases, zero retries, and an output SHA-256 exactly matching the development reference: `728b02c18ed99f9336d3621c212aa2d984eb0e22f87a62d051c832a35e52f4c3`.

## Audio and runtime findings

The final backend evaluation uses two phrases per language. Mean pitch MAE was 32.78 cents KO, 28.28 EN, and 23.36 JA. No clipping, format, or silence gate failed. Every long-form phrase was present and unique. All 16 boundaries passed click, energy, spectrum, silence, and expected-continuity F0 gates; maximum continuous-boundary F0 movement was 72.9 cents and maximum sample click was 0.000183.

The resident loads each model once, reports worker state from `/health`, serializes concurrent model requests, survives a malformed request, restarts deterministically, and leaves no child process after shutdown. Twenty identical phrase renders produced one unique hash.

## Known limitations

Linux NVIDIA CUDA is the only release-tested platform. Initial model download is roughly 9 GB. Peak measured unified-memory growth was about 15.4 GB. Korean alone has personalized prosody evidence. EN/JA intelligibility varies, particularly the held-out Japanese phrase. GYU identity gains are small and their confidence intervals cross zero. RC4 human listening found unacceptable metallic/robotic artifacts, intermittent harsh buzzing/noise, and smeared or collapsing transitions, worst on fast syllables, high notes, and large intervals. The final tag is prohibited until a later RC passes human review.

## Evidence

- `artifacts/reports/openutau_upstream_v10.json`
- `artifacts/reports/openutau_v10/behavior.json`
- `artifacts/reports/runtime_v10_stress.json`
- `artifacts/reports/longform_v10_quality.json`
- `artifacts/reports/longform_v10_supervised.json`
- `artifacts/reports/release_audio_v10.json`
- `artifacts/reports/release_candidate_v10.json`
- `artifacts/reports/listening_v10/manifest.json`

## Remaining release steps

Listen to `artifacts/reports/rc5_stress_candidate4/listening/` and the matched `before_after/` clips. Only an explicit human pass may promote this engineering candidate into a real RC5. Runtime integration, clean packaging, and the full install/long-form smoke remain intentionally deferred until that pass; final `v1.0.0` remains prohibited.
