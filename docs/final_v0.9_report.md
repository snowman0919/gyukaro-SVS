Overall status: Achieved; all v0.7, v0.8, and v0.9 semantic gates passed with real measured paths
Highest achieved version: v0.9
Package: artifacts/package/gyu-singer-v0.9-openutau.zip
Package SHA-256: 91839be510c478dde3bfcbad0fcfd238c31d8f828cf87074f942a0bbcafaaf63
Git commit: bb6dc624ef4f6318342cb3441bf3059b786a2ed8
Real SoulX latent training: 117 actual `SoulXSingerSVC.infer_segment.gt_decoder_inp` tensors; real GYU 24, accepted pseudo 45, teacher 48
Identity adapter: v0.7 separable real-latent FiLM, active in production; final-audio effect measurable but quality gain modest
Style adapter: v0.7 separable real-latent FiLM, active; breathy and energetic proxy directions validated, other presets relative/unverified
Production prosody: v0.5 real-GYU controller, selected over v0.6 on 24 independent-score rows
KO: 48 kHz phrase render and OpenUtau project case pass
EN: 48 kHz phrase render and OpenUtau project case pass; generic prosody plus GYU identity/style, not real-GYU EN prosody
JA: 48 kHz phrase render and OpenUtau project case pass; generic prosody plus GYU identity/style, not real-GYU JA prosody
OpenUtau integration: executable maintained fork overlay at official commit 27573ac5c888d927119d5f65a207312d79194b1f
OpenUtau phrase rendering: real C# `IRenderer.Render` -> resident v0.8 -> non-silent 44.1 kHz editor samples pass
User pitch control: authoritative PITD edit produced +92.52 cents RMVPE median shift
Style control: neutral/energetic final RMS 0.078379/0.079411 with all other inputs fixed
Clean package smoke: pass; clean unpack direct 48 kHz render, clean OpenUtau build/tests, resident C# render, and behavioral suite

# What changed after v0.6

v0.7 replaced synthetic/dummy adapter inputs with cached decoder-condition tensors from the actual SoulX production hook. Identity and style became separate trained paths. v0.8 retained only measured components: v0.5 prosody, v0.7 real-latent identity/style, multilingual phrase content, and SoulX phrase decoding. v0.9 added a native multi-note OpenUtau renderer, virtual singer, request cache, resident service, three-language USTX, and behavioral editor tests.

# Actual v0.7 training path

`scripts/cache_soulx_latents_v07.py` captured 117 nonzero real tensors at `SoulXSingerSVC.infer_segment.gt_decoder_inp`: 24 real GYU, 45 accepted pseudo-singing, and 48 selected teacher rows; KO/EN/JA counts are 87/17/13. Each manifest row records audio, source type, language, style, identity target, trust, SoulX revision, tensor path, shape, and statistics. `scripts/train_real_latent_adapters_v07.py` trained separately supervised `GYUIdentityAdapter` and `GYUStyleAdapter`; checkpoint `checkpoints/gyu_real_latent_adapters_v0.7.pt` is injected before the existing SoulX decoder.

# Actual v0.8 production path

```text
lyrics -> KO/EN/JA frontend -> phrase content
score -> nominal F0 + v0.5 GYU residual + user pitch curve
content + v0.7 GYU identity + v0.7 style -> SoulX latent adapter
-> SoulX phrase decoder -> 48 kHz WAV
```

Backend `gyu-singer-v0.8` uses no per-note TTS, waveform pitch shifting, or phase-vocoder note control. Clean-unpack `render.sh` produced mono 48 kHz audio, 476160 frames, 9.92 seconds.

# Measured selection

Identity uses six phrases, two per KO/EN/JA. Student minus identity-off is WavLM +0.00200 (95% interval -0.00932..+0.01124) and ECAPA +0.00134 (-0.00605..+0.00835). The confidence intervals cross zero, so this is a real causal effect and architecture improvement, not strong proof of perceptual quality improvement. WavLM cross-language consistency moves 0.71532 -> 0.74530 on quality phrases and 0.89438 -> 0.89848 held out.

Style holds score, content source, F0, reference, and identity fixed. Breathy increases the 4 kHz high-frequency proxy and energetic increases RMS across quality, held-out, and locked Korean cases. Soft, dark, and bright did not validate consistently and remain explicitly named relative controls.

On the 24-row independent score set, v0.5 has the lowest pitch MAE (134.5467 cents) and established production behavior. v0.6 slightly wins log-F0 RMSE and transition measures but not consistently, so v0.5 remains production prosody. The independent scores are independently transcribed, not human-certified notation.

# OpenUtau integration and behavior

Current OpenUtau has an internal `IRenderer` API but no external renderer-registration API. The smallest supported solution is a pinned maintained fork. `install_fork.sh` applies three registration hunks and installs `GyuSingerRenderer.cs`; it adds a virtual singer whose dummy OTOs form phrases but never synthesize audio. The renderer maps notes, lyrics, tempo, phonemes, final OpenUtau pitch including vibrato/PITD, dynamics, breathiness, tension, and GYUS style. Complete request SHA-256 controls phrase caching and edit invalidation.

`examples/openutau_v09.ustx` loads three GYU tracks. Seven native tests pass. A separate test invokes the real C# renderer and receives resident phrase audio. `scripts/test_openutau_v09_behavior.py` reports:

- note pitch +2 semitones -> +200.41 cents RMVPE;
- PITD +1 semitone -> +92.52 cents RMVPE;
- lyric edit changes Whisper output from “하늘빛 노래” to “사랑을 담아서 내준 내 마음”;
- neutral -> energetic changes final RMS 0.078379 -> 0.079411;
- KO, EN, and JA are all 48 kHz mono phrase WAVs.

# Known limitations

- Identity gains are small and heterogeneous; the intervals cross zero.
- Only breathy and energetic have acoustic-proxy semantic evidence; no formal listening panel was run.
- Real personalized singing prosody is Korean-only. EN/JA use generic multilingual prosody plus GYU identity/style.
- OpenUtau integration is a pinned fork because upstream lacks external renderer registration.
- The archive contains project-trained inference checkpoints and reference data, but expects the documented pinned SoulX/OmniVoice cache and compatible SoulX Python externally. Training-only Fish, MOSS, and Higgs weights are not bundled.

# Install and run

```sh
unzip gyu-singer-v0.9-openutau.zip
cd gyu-singer-v0.9-openutau
export GYU_SINGER_CACHE=/absolute/path/to/pinned/model-cache
export GYU_SOULX_PYTHON=/absolute/path/to/.venv-soulx/bin/python
./serve.sh

git clone https://github.com/stakira/OpenUtau.git
git -C OpenUtau checkout 27573ac5c888d927119d5f65a207312d79194b1f
./integrations/openutau/install_fork.sh OpenUtau
dotnet build OpenUtau/OpenUtau.csproj -c Release
export GYU_RENDERER_URL=http://127.0.0.1:8765/render
```

Open `examples/openutau_v09.ustx` in the built editor. Backend status is available at `http://127.0.0.1:8765/model`; health is at `/health`.
