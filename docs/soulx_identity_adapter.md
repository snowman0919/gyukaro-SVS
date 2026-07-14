# v0.6 SoulX identity adapter

{
  "scores": {
    "quality_ko": {
      "language": "ko",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.67829,
          "ecapa_to_gyu": 0.20495,
          "wavlm_l2_to_student": 0.001515
        },
        "fish_only": {
          "wavlm_to_gyu": 0.67287,
          "ecapa_to_gyu": 0.20224,
          "wavlm_l2_to_student": 0.001636
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.68165,
          "ecapa_to_gyu": 0.20443,
          "wavlm_l2_to_student": 0.001638
        },
        "student": {
          "wavlm_to_gyu": 0.68352,
          "ecapa_to_gyu": 0.20317,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "heldout_ko": {
      "language": "ko",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.67977,
          "ecapa_to_gyu": 0.02944,
          "wavlm_l2_to_student": 0.001595
        },
        "fish_only": {
          "wavlm_to_gyu": 0.68039,
          "ecapa_to_gyu": 0.02948,
          "wavlm_l2_to_student": 0.001584
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.68009,
          "ecapa_to_gyu": 0.0307,
          "wavlm_l2_to_student": 0.001462
        },
        "student": {
          "wavlm_to_gyu": 0.6765,
          "ecapa_to_gyu": 0.02951,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "quality_en": {
      "language": "en",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.78722,
          "ecapa_to_gyu": 0.12018,
          "wavlm_l2_to_student": 0.003026
        },
        "fish_only": {
          "wavlm_to_gyu": 0.79488,
          "ecapa_to_gyu": 0.11601,
          "wavlm_l2_to_student": 0.002225
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.78367,
          "ecapa_to_gyu": 0.12246,
          "wavlm_l2_to_student": 0.002943
        },
        "student": {
          "wavlm_to_gyu": 0.78845,
          "ecapa_to_gyu": 0.11735,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "heldout_en": {
      "language": "en",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.88705,
          "ecapa_to_gyu": 0.23445,
          "wavlm_l2_to_student": 0.001236
        },
        "fish_only": {
          "wavlm_to_gyu": 0.88448,
          "ecapa_to_gyu": 0.23654,
          "wavlm_l2_to_student": 0.001579
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.88881,
          "ecapa_to_gyu": 0.23181,
          "wavlm_l2_to_student": 0.001243
        },
        "student": {
          "wavlm_to_gyu": 0.88963,
          "ecapa_to_gyu": 0.22867,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "quality_ja": {
      "language": "ja",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.39711,
          "ecapa_to_gyu": 0.04493,
          "wavlm_l2_to_student": 0.003433
        },
        "fish_only": {
          "wavlm_to_gyu": 0.41974,
          "ecapa_to_gyu": 0.04009,
          "wavlm_l2_to_student": 0.00189
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.41007,
          "ecapa_to_gyu": 0.03957,
          "wavlm_l2_to_student": 0.002456
        },
        "student": {
          "wavlm_to_gyu": 0.4228,
          "ecapa_to_gyu": 0.04379,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "heldout_ja": {
      "language": "ja",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.81858,
          "ecapa_to_gyu": 0.3153,
          "wavlm_l2_to_student": 0.000738
        },
        "fish_only": {
          "wavlm_to_gyu": 0.8228,
          "ecapa_to_gyu": 0.31778,
          "wavlm_l2_to_student": 0.000794
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.82086,
          "ecapa_to_gyu": 0.31586,
          "wavlm_l2_to_student": 0.00082
        },
        "student": {
          "wavlm_to_gyu": 0.81816,
          "ecapa_to_gyu": 0.3134,
          "wavlm_l2_to_student": 0.0
        }
      }
    }
  },
  "student_minus_no_identity": {
    "wavlm_to_gyu": {
      "n": 6,
      "mean": 0.00517,
      "p2_5": -0.00057,
      "p97_5": 0.01385,
      "values": [
        0.00523,
        -0.00327,
        0.00123,
        0.00258,
        0.02569,
        -0.00042
      ]
    },
    "ecapa_to_gyu": {
      "n": 6,
      "mean": -0.00223,
      "p2_5": -0.00382,
      "p97_5": -0.00095,
      "values": [
        -0.00178,
        7e-05,
        -0.00283,
        -0.00578,
        -0.00114,
        -0.0019
      ]
    }
  }
}

Fixed score, lyrics, deterministic OmniVoice/SoulX seeds, and reference; only the named conditioning is varied. Identity uses two phrases per KO/EN/JA. Style compares v0.5 spectral-only, v0.6 without latent injection, and v0.6 latent injection on two KO phrases.
