# v1 architecture

`JSON score -> resident fine-tuned MOSS Nano vocalizer -> per-note GYU-reference speech -> explicit MIDI pitch shift -> phase-vocoder duration control -> 48 kHz WAV`

The trained component is `gyu_moss_nano_sft/checkpoint-last`: MOSS-TTS-Nano SFT on 64 ASR-confirmed real GYU Korean singing phrases. The audio tokenizer remains frozen. Runtime preserves GYU conditioning through an authorized reference phrase and exposes note pitch, start, duration, lyric, and dynamics.

This is one coherent v1 path, not a concatenation of SVS repositories. It is a neural multilingual vocalizer with DSP score control, not a flow-matching score-to-mel singing model. Isolated one-character prompts can emit empty Nano token sequences; runtime deliberately falls back to real-GYU reference vocal material rather than failing.
