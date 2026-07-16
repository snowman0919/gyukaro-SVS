# RC9 local reference-song analysis

Status: local evaluation complete; source audio, separated audio, lyrics, and the reconstructed project are excluded from Git and packages.

The user-provided mix (208.469 s, 48 kHz), off-vocal, and local lyrics were used without downloading replacement material. The off-vocal was aligned by cross-correlation (`-95.5 ms`) and removed with complex-STFT subtraction. The resulting vocal estimate is an imperfect analysis aid, not clean ground truth.

## Analysis definitions

| Scale | FFT / hop at 48 kHz | Purpose |
|---|---:|---|
| short | 512 / 128 (10.7 ms window) | attacks and consonants |
| medium | 2048 / 512 (42.7 ms) | phoneme and formant structure |
| long | 8192 / 2048 (170.7 ms) | harmonics and sustained notes |

Tempo analysis measured `151.999081 BPM`. Pitch evidence combines residual-vocal RMVPE, original-mix RMVPE, and pYIN. A frame is accepted only when at least two estimators agree within 100 cents. This produced 6,237 accepted 50 Hz frames and 566 musically merged note candidates. The score labels remain explicitly **inferred**; they are not manual ground truth.

## Local OpenUtau project

The reproducible builder created a 203.72 s editable project with 55 parts, 566 notes, 6,200 PITD points, and 17 vibrato-marked notes. Lyrics came only from the user-provided local file. OpenUtau later generated 566 phonemes with zero validation errors.

Reproduce locally:

```bash
PYTHONPATH=src python scripts/analyze_reference_song_rc9.py
PYTHONPATH=src python scripts/build_reference_ustx_rc9.py
```

Evidence: `artifacts/reports/reference_song_rc9_analysis.json` and `artifacts/reports/reference_song_rc9_project.json`. Local paths named by those reports are intentionally absent from the package.
