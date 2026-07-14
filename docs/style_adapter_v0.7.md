# v0.7 latent acoustic-style adapter

{
  "protocol": "Sign calibration selected on quality_ko only. heldout_ko was observed during development; korean.json is the new locked confirmation. Same score/content/reference/F0; only latent style condition changes.",
  "semantic_evidence_is_proxy_not_listening_proof": true,
  "calibration": {
    "energetic": -0.5,
    "dark": -2.0
  },
  "semantic_status": {
    "neutral": "neutral",
    "soft": "relative_style_A_unverified",
    "breathy": "breathiness_proxy",
    "energetic": "energy_proxy",
    "dark": "relative_style_C_unverified",
    "bright": "relative_style_B_unverified"
  },
  "paths": {
    "quality_neutral": "artifacts/reports/v07_style_semantics_quality_neutral.wav",
    "quality_soft": "artifacts/reports/v07_style_semantics_quality_soft.wav",
    "quality_breathy": "artifacts/reports/v07_style_semantics_quality_breathy.wav",
    "quality_energetic": "artifacts/reports/v07_style_semantics_quality_energetic.wav",
    "quality_dark": "artifacts/reports/v07_style_semantics_quality_dark.wav",
    "quality_bright": "artifacts/reports/v07_style_semantics_quality_bright.wav",
    "heldout_neutral": "artifacts/reports/v07_style_semantics_heldout_neutral.wav",
    "heldout_soft": "artifacts/reports/v07_style_semantics_heldout_soft.wav",
    "heldout_breathy": "artifacts/reports/v07_style_semantics_heldout_breathy.wav",
    "heldout_energetic": "artifacts/reports/v07_style_semantics_heldout_energetic.wav",
    "heldout_dark": "artifacts/reports/v07_style_semantics_heldout_dark.wav",
    "heldout_bright": "artifacts/reports/v07_style_semantics_heldout_bright.wav",
    "korean_neutral": "artifacts/reports/v07_style_semantics_korean_neutral.wav",
    "korean_soft": "artifacts/reports/v07_style_semantics_korean_soft.wav",
    "korean_breathy": "artifacts/reports/v07_style_semantics_korean_breathy.wav",
    "korean_energetic": "artifacts/reports/v07_style_semantics_korean_energetic.wav",
    "korean_dark": "artifacts/reports/v07_style_semantics_korean_dark.wav",
    "korean_bright": "artifacts/reports/v07_style_semantics_korean_bright.wav"
  },
  "measurements": {
    "quality_neutral": {
      "spectral_centroid_hz": 1067.923306,
      "rms": 0.15750377,
      "high_frequency_ratio_4khz": 0.01357493
    },
    "quality_soft": {
      "spectral_centroid_hz": 1055.404743,
      "rms": 0.16937326,
      "high_frequency_ratio_4khz": 0.01390527
    },
    "quality_breathy": {
      "spectral_centroid_hz": 1048.415629,
      "rms": 0.19101402,
      "high_frequency_ratio_4khz": 0.02309543
    },
    "quality_energetic": {
      "spectral_centroid_hz": 1062.311024,
      "rms": 0.15781406,
      "high_frequency_ratio_4khz": 0.01345857
    },
    "quality_dark": {
      "spectral_centroid_hz": 1053.874908,
      "rms": 0.15769961,
      "high_frequency_ratio_4khz": 0.01250905
    },
    "quality_bright": {
      "spectral_centroid_hz": 1023.421113,
      "rms": 0.17992514,
      "high_frequency_ratio_4khz": 0.01487559
    },
    "heldout_neutral": {
      "spectral_centroid_hz": 1022.750562,
      "rms": 0.09499443,
      "high_frequency_ratio_4khz": 0.00293024
    },
    "heldout_soft": {
      "spectral_centroid_hz": 1079.575749,
      "rms": 0.10016793,
      "high_frequency_ratio_4khz": 0.00583992
    },
    "heldout_breathy": {
      "spectral_centroid_hz": 1019.725746,
      "rms": 0.10197236,
      "high_frequency_ratio_4khz": 0.00909466
    },
    "heldout_energetic": {
      "spectral_centroid_hz": 1014.116175,
      "rms": 0.09530416,
      "high_frequency_ratio_4khz": 0.00289549
    },
    "heldout_dark": {
      "spectral_centroid_hz": 1034.162575,
      "rms": 0.09615673,
      "high_frequency_ratio_4khz": 0.00340796
    },
    "heldout_bright": {
      "spectral_centroid_hz": 1051.595394,
      "rms": 0.0974468,
      "high_frequency_ratio_4khz": 0.0078006
    },
    "korean_neutral": {
      "spectral_centroid_hz": 937.806136,
      "rms": 0.11337861,
      "high_frequency_ratio_4khz": 0.00061833
    },
    "korean_soft": {
      "spectral_centroid_hz": 937.267384,
      "rms": 0.11912273,
      "high_frequency_ratio_4khz": 0.00082733
    },
    "korean_breathy": {
      "spectral_centroid_hz": 907.495231,
      "rms": 0.12213451,
      "high_frequency_ratio_4khz": 0.001303
    },
    "korean_energetic": {
      "spectral_centroid_hz": 936.37183,
      "rms": 0.11419648,
      "high_frequency_ratio_4khz": 0.00060161
    },
    "korean_dark": {
      "spectral_centroid_hz": 951.97324,
      "rms": 0.11497946,
      "high_frequency_ratio_4khz": 0.00062303
    },
    "korean_bright": {
      "spectral_centroid_hz": 955.518588,
      "rms": 0.11502954,
      "high_frequency_ratio_4khz": 0.00110288
    }
  },
  "deltas_from_neutral": {
    "quality": {
      "soft": {
        "spectral_centroid_hz": -12.518563,
        "rms": 0.011869,
        "high_frequency_ratio_4khz": 0.00033
      },
      "breathy": {
        "spectral_centroid_hz": -19.507677,
        "rms": 0.03351,
        "high_frequency_ratio_4khz": 0.00952
      },
      "energetic": {
        "spectral_centroid_hz": -5.612282,
        "rms": 0.00031,
        "high_frequency_ratio_4khz": -0.000116
      },
      "dark": {
        "spectral_centroid_hz": -14.048398,
        "rms": 0.000196,
        "high_frequency_ratio_4khz": -0.001066
      },
      "bright": {
        "spectral_centroid_hz": -44.502193,
        "rms": 0.022421,
        "high_frequency_ratio_4khz": 0.001301
      }
    },
    "heldout": {
      "soft": {
        "spectral_centroid_hz": 56.825187,
        "rms": 0.005173,
        "high_frequency_ratio_4khz": 0.00291
      },
      "breathy": {
        "spectral_centroid_hz": -3.024816,
        "rms": 0.006978,
        "high_frequency_ratio_4khz": 0.006164
      },
      "energetic": {
        "spectral_centroid_hz": -8.634387,
        "rms": 0.00031,
        "high_frequency_ratio_4khz": -3.5e-05
      },
      "dark": {
        "spectral_centroid_hz": 11.412013,
        "rms": 0.001162,
        "high_frequency_ratio_4khz": 0.000478
      },
      "bright": {
        "spectral_centroid_hz": 28.844832,
        "rms": 0.002452,
        "high_frequency_ratio_4khz": 0.00487
      }
    },
    "korean": {
      "soft": {
        "spectral_centroid_hz": -0.538752,
        "rms": 0.005744,
        "high_frequency_ratio_4khz": 0.000209
      },
      "breathy": {
        "spectral_centroid_hz": -30.310905,
        "rms": 0.008756,
        "high_frequency_ratio_4khz": 0.000685
      },
      "energetic": {
        "spectral_centroid_hz": -1.434306,
        "rms": 0.000818,
        "high_frequency_ratio_4khz": -1.7e-05
      },
      "dark": {
        "spectral_centroid_hz": 14.167104,
        "rms": 0.001601,
        "high_frequency_ratio_4khz": 5e-06
      },
      "bright": {
        "spectral_centroid_hz": 17.712452,
        "rms": 0.001651,
        "high_frequency_ratio_4khz": 0.000485
      }
    }
  },
  "direction_checks": {
    "quality": {
      "soft_rms_lower": false,
      "breathy_high_frequency_ratio_higher": true,
      "energetic_rms_higher": true,
      "dark_centroid_lower": true,
      "bright_centroid_higher": false
    },
    "heldout": {
      "soft_rms_lower": false,
      "breathy_high_frequency_ratio_higher": true,
      "energetic_rms_higher": true,
      "dark_centroid_lower": false,
      "bright_centroid_higher": true
    },
    "korean": {
      "soft_rms_lower": false,
      "breathy_high_frequency_ratio_higher": true,
      "energetic_rms_higher": true,
      "dark_centroid_lower": false,
      "bright_centroid_higher": true
    }
  },
  "all_directions_pass": {
    "quality": false,
    "heldout": false,
    "korean": false
  },
  "validated_semantic_directions_pass": {
    "quality": true,
    "heldout": true,
    "korean": true
  }
}

All semantic claims above are inferred acoustic proxies. Listening validation remains separate.
