# v1 architecture

`JSON score -> resident Renderer -> nearest stable real-GYU voiced loop -> pitch-ratio resampling -> overlap/fade WAV`

The model checkpoint contains eight selected periodic voiced loops and their measured F0. Pitch is direct MIDI control. Note durations, timing, dynamics, and arbitrary lyric fields are accepted. Lyrics do not condition waveform in v1; Korean/English/Japanese status is therefore protocol-only, not intelligible synthesis.

Next neural architecture: language-aware phonemes and note relations -> score/content encoder -> real-anchor timbre adapter + expression projection -> conditional-flow acoustic decoder with explicit target F0/residual and singing vocoder. Train only after manual transcript/alignment plus pretrained SVS legal checkpoint selection.
