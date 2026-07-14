# Pseudo-singing coverage (v0.5)

{
  "real_rows": 76,
  "accepted_pseudo_rows": 27,
  "real_pitch_range": [
    36,
    65
  ],
  "real_duration_range": [
    0.08,
    1.6
  ],
  "real_script_shapes": {
    "free": 22,
    "same": 18,
    "ascending": 18,
    "descending": 18
  },
  "pseudo_languages": {
    "ko": 11,
    "ja": 9,
    "en": 7
  },
  "coverage_targets": {
    "new_candidates": "200-500 after gap review",
    "languages": [
      "ko",
      "en",
      "ja"
    ],
    "quality_gates": [
      "RMVPE agreement",
      "duration ratio",
      "speaker similarity",
      "ASR",
      "language ID",
      "audio quality",
      "degeneration"
    ]
  },
  "status": "analysis_only_no_blind_generation"
}

Accepted pseudo-singing remains low-trust (`trust_weight: 0.2`). New candidates are not admitted without the listed gates.