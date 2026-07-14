# v1.0 listening review

Status: pending human review. No subjective score has been inferred from automated metrics.

Review the nine files in `artifacts/reports/listening_v10/` with headphones at a comfortable fixed level. Report a short concrete observation for each item and whether it is acceptable for an experimental v1.0 release.

- `ko_neutral.wav`: lyric clarity, pitch stability, target-singer plausibility
- `ko_breathy.wav`: whether breathiness increases without damaging lyrics
- `ko_energetic.wav`: whether energy increases without clipping or harshness
- `en.wav`: English intelligibility and identity consistency
- `ja.wav`: Japanese intelligibility and identity consistency
- `sustain_ko.wav`: held-note stability, vibrato, metallic/buzzy artifacts
- `rapid_ko.wav`: consonants, timing, missing/repeated syllables
- `large_interval_ko.wav`: transition accuracy and unnatural glides
- `phrase_boundary.wav`: click, gap, overlap, or clipped consonant at the center

Accepting the release requires no severe content corruption, missing phrase, obvious boundary click, or unusable sustained/rapid/interval behavior. Style differences may be subtle, but breathy and energetic must not be reversed. If an item fails, identify its filename and defect; do not average it away.
