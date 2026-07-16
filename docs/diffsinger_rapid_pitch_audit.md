# Rapid Japanese pitch and intelligibility audit

Status: corrected source candidate passed objective waveform/STT gates; human listening pending.

The previously supplied DiffSinger depth 0.4 and 0.6 probes are invalid. They were rendered from an octave-high score and their free Whisper transcripts did not recover the lyric. They must not be used as RC8/RC9 or OpenUtau evidence.

The local independent UST encodes note number 72, but the matching reference waveform does not have a 523 Hz fundamental. On the locally separated 2.432 s vocal, YIN measured a 259.54 Hz median. The spectrum contains the 261/523/786/1045 Hz series; energy at odd multiples of 261 Hz proves that the strong 523 Hz ridge is the second harmonic. The audited render score is therefore MIDI 60, with a documented -12 semitone correction. The local reference and generated test audio remain excluded from Git and packaging.

At corrected C4, six PJS checkpoints still failed free transcription. The best PJS result was only a partial lyric. The GTSinger source checkpoint at step 15000 produced the exact free Whisper transcript four times:

`息が詰まる 息が詰まる 息が詰まる 息が詰まる`

Its waveform metrics were 261.31 Hz median F0, 35.83 cents p90 absolute pitch error, 89.34% observed voiced frames, and 0 clipping. The rapid-curriculum checkpoints regressed transcription and are rejected. This objective pass does not replace human listening.

Every future promoted audio candidate must include:

- a free, unprompted Whisper transcript;
- expected-text similarity without teacher forcing;
- F0 distribution and score error;
- voiced ratio, clipping, RMS/peak, high-frequency energy, spectral flatness/flux, and sample discontinuity;
- waveform and spectrogram review;
- explicit human-listening status.

Evidence:

- `artifacts/reports/diffsinger_reference_pitch_octave_audit.json`
- `artifacts/reports/diffsinger_pjs_independent_c4_evaluation.json`
- `artifacts/reports/diffsinger_gtsinger_independent_c4_evaluation.json`
- local-only listening candidate: `artifacts/reports/diffsinger_independent_c4_candidate/listening/rapid_ja_c4_source15000.wav`
