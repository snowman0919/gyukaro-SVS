# GYU Hybrid Singer v0.4 quality runtime

Whole-phrase neural path: TriSinger score/content/timbre/style conditioner -> flow-predicted expressive F0 residual -> OmniVoice duration-locked multilingual lyric content -> SoulX-Singer SVC. SoulX receives the complete 50 Hz score-plus-controller F0 contour. No per-note TTS, pitch-shift, time-stretch, or waveform concatenation.

The package includes the trained TriSinger pitch controller but excludes upstream weights. On a CUDA host with Git and Python 3, run `sh bootstrap.sh /path/to/cache`; export the paths it prints; then run `sh run.sh`. Bootstrap pins OmniVoice `1574e06` and SoulX-Singer `81aeb3a`.
