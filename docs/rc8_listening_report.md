# RC8 listening report

Status: eight cases passed human review; Large Interval failed. RC8 is not accepted, and RC9 must not start until the remaining case passes.

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
| 8 | Large Interval KO | less initial dual-trajectory/mechanical tearing | FAIL | further engineering required |
| 9 | Phrase boundary | no fade/staccato boundary regression | PASS | user accepted |

Overall RC8 suitability: **FAIL — Large Interval only**.

Large Interval retest: an 80 ms score-domain onset ramp passed the objective dual-trajectory gate and is now integrated into the actual RC8 backend when no user PITD curve is present. The actual-backend render at `artifacts/reports/rc8_interval_actual_backend/listening/large_interval_ko.wav` is byte-identical (SHA-256 `63a1e8a77d6de1a501c4f01920fc2c3927c607b233c86cbcc81220c26633c105`) to the selected sweep candidate. The failed RC8 file remains unchanged. Human listening of the new actual-backend file is still required before RC8 can pass.

RC9 OpenUtau work: **blocked by the required RC8 listening verdict, not started**.

Final `v1.0.0` tag/release: **not allowed**.
