# Model card: RC5

GYU Singer v1.0 is a personalized experimental singing system for one consented target voice. Production inference is:

```text
OpenUtau RenderPhrase
-> notes/lyrics/final editor pitch and expressions
-> canonical phoneme/note voicing timeline; silence/unvoiced F0 = 0
-> v0.5 real-GYU Korean prosody residual
-> OmniVoice multilingual phrase content
-> MMS CTC latent timing correction for rapid and English phrases
-> v0.7 GYU identity and latent style conditioning
-> FP32 SoulX-Singer phrase decoder with measured stress-specific steps/CFG
-> 48 kHz mono phrase WAV
```

OpenUtau mixes and exports at 44.1 kHz stereo. There is no per-note TTS, waveform pitch shifting, phase-vocoder note control, or older-backend fallback.

Validated data and metrics:

- 24 independently transcribed Korean score rows select the v0.5 prosody controller.
- Two final phrases per KO/EN/JA had mean pitch MAE 32.78/28.28/23.36 cents and mean ASR lyric similarity 1.000/0.945/0.683.
- A native 136-note, 17-phrase, 119.983-second OpenUtau project exported with 0 failed phrases; repeat cache render was 0.107 seconds.
- Stable controls: neutral, breathy, energetic. Soft, dark, and bright are experimental relative controls.
- RC5 nine-case listening and matched RC4 before/after review passed; final `v1.0.0` remains unpublished.

Korean has personalized GYU prosody. English and Japanese use generic multilingual prosody plus GYU identity/style. Speaker-identity gains are measured but modest and heterogeneous; this is not a claim of perfect voice cloning.
