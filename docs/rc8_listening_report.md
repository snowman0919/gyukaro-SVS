# RC8 listening report

Status: **superseded — later human review failed; RC8 is not accepted.**

This file preserves the historical nine-case decision, but it is no longer authoritative. The later user report of excessive pitch and unintelligibility overrides it, and the fresh machine re-audit does not establish material improvement. Any future promotion requires a new listening decision on files that have first passed free Whisper, RMVPE F0/voicing, and waveform/spectral analysis.

Listening directory: `artifacts/reports/rc8_listening_gate/`

Objective report: `artifacts/reports/rc8_listening_gate/evaluation.json`

The directory contains nine numbered RC8 files and six RC7-before/RC8-after pairs for KO neutral, EN, JA, sustained KO, Large Interval KO, and protected Rapid KO. Use headphones and compare at matched playback volume.

| # | Case | Required check | Verdict | Observation |
|---:|---|---|---|---|
| 1 | KO neutral | less overconnection, no staccato, exact lyrics | PASS | user accepted |
| 2 | KO breathy | breathiness and intelligibility preserved | PASS | user accepted |
| 3 | KO energetic | attacks and identity preserved | PASS | user accepted |
| 4 | EN | transition naturalness and intelligibility | PASS | user accepted |
| 5 | JA | weak phonemes improved without invalid devoicing | PASS | user accepted |
| 6 | Rapid KO | no practical regression from accepted RC7 | PASS | user accepted |
| 7 | Sustained KO | materially less noise without vibrato/harmonic damage | PASS | user accepted |
| 8 | Large Interval KO | less initial dual-trajectory/mechanical tearing | PASS | fixed actual-backend render accepted by user |
| 9 | Phrase boundary | no fade/staccato boundary regression | PASS | user accepted |

Historical RC8 suitability: **PASS at the time; subsequently invalidated**.

Large Interval retest: an 80 ms score-domain onset ramp passed the objective dual-trajectory gate and is integrated into the actual RC8 backend when no user PITD curve is present. The accepted actual-backend render is `artifacts/reports/rc8_interval_actual_backend/listening/large_interval_ko.wav` (SHA-256 `63a1e8a77d6de1a501c4f01920fc2c3927c607b233c86cbcc81220c26633c105`). The earlier failed RC8 file remains preserved separately.

RC9 OpenUtau work: **not authorized for release by this historical pass**.

Final `v1.0.0` tag/release: **not allowed**.

## Candidate 3 listening request

Status: **machine gate complete; human listening pending.**

Directory: `artifacts/reports/rc8_candidate3_full/listening/`

The nine files were rendered through one actual `gyu-singer-rc8` resident path. Free Whisper transcripts were preserved for every case; the aggregate waveform/RMVPE result is recorded in `artifacts/reports/rc8_candidate3_full/evaluation.json`. This candidate replaces the rejected stronger stationary spectral gate with a bounded `64 steps / CFG 1.5` policy only for long neutral Korean notes.

Listen to all nine files against the frozen RC7 files. Pay particular attention to sustained noise, EN and JA transitions, Large Interval tearing/identity, and protected Rapid KO. No human verdict has been inferred from machine metrics.

EN decoder A/B (same OmniVoice source, F0, identity, style, CTC warp, and seed):

- current: `artifacts/reports/rc8_en_decode_sweep/listening/s32_c1.5.wav`
- alternative: `artifacts/reports/rc8_en_decode_sweep/listening/s32_c2.wav`

Free Whisper favors the alternative, but waveform diagnostics are mixed. Human review should select neither unless the transition improvement outweighs its increased noise and discontinuity.
