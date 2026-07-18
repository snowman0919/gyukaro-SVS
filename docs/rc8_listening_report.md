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

## Candidate 3 failure-evidence files

Status: **rejected; not human-pending.**

Directory: `artifacts/reports/rc8_candidate3_full/listening/`

The nine files were rendered through one actual `gyu-singer-rc8` resident path. Free Whisper transcripts were preserved for every case; the aggregate waveform/RMVPE result is recorded in `artifacts/reports/rc8_candidate3_full/evaluation.json`. This candidate replaces the rejected stronger stationary spectral gate with a bounded `64 steps / CFG 1.5` policy only for long neutral Korean notes.

The nine files are preserved for confirming the failure modes against frozen RC7. They are not a promotion listening set and cannot authorize RC8.

EN decoder A/B (same OmniVoice source, F0, identity, style, CTC warp, and seed):

- current: `artifacts/reports/rc8_en_decode_sweep/listening/s32_c1.5.wav`
- alternative: `artifacts/reports/rc8_en_decode_sweep/listening/s32_c2.wav`

Free Whisper favors the alternative, but waveform diagnostics are mixed. Neither decoder setting is selected; the files are diagnostic evidence only.

## JA duplicate-span diagnostic listening files

Status: **machine reject; do not request promotion listening.**

Directories:

- `artifacts/reports/rc8_ja_duplicate_span/quality_ja/listening/`
- `artifacts/reports/rc8_ja_duplicate_span/heldout_ja/listening/`

Each contains `current_rc8.wav`, `global_ctc_025.wav`, `chunked_single_decode.wav`, and `duplicate_span_candidate.wav`. The duplicate candidate is byte-identical to current RC8 because both JA alignments failed the bounded CTC confidence gate; no source interval was removed. Held-out Whisper remains the repeated `新しい歌を風に乗せて新しい歌を風に乗せて届ける` at similarity `0.7222`. Chunking reaches only `0.8966` and materially worsens HF spike, sample jump, and voicing. A is rejected and not integrated.

Direct waveform/STFT review images are `quality_ja/waveform_multires_stft.png` and `heldout_ja/waveform_multires_stft.png` under the same report directory. These files cannot approve RC8 or authorize RC9/OpenUtau work.
