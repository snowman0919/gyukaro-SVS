# v07 SoulX identity adapter

{
  "scores": {
    "quality_ko": {
      "language": "ko",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.67068,
          "ecapa_to_gyu": 0.21091,
          "wavlm_l2_to_student": 0.002285
        },
        "fish_only": {
          "wavlm_to_gyu": 0.68364,
          "ecapa_to_gyu": 0.22669,
          "wavlm_l2_to_student": 0.001613
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.68343,
          "ecapa_to_gyu": 0.2249,
          "wavlm_l2_to_student": 0.001683
        },
        "student": {
          "wavlm_to_gyu": 0.67112,
          "ecapa_to_gyu": 0.22682,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "heldout_ko": {
      "language": "ko",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.67942,
          "ecapa_to_gyu": 0.02505,
          "wavlm_l2_to_student": 0.00312
        },
        "fish_only": {
          "wavlm_to_gyu": 0.70677,
          "ecapa_to_gyu": 0.03763,
          "wavlm_l2_to_student": 0.002718
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.69734,
          "ecapa_to_gyu": 0.0312,
          "wavlm_l2_to_student": 0.00257
        },
        "student": {
          "wavlm_to_gyu": 0.69429,
          "ecapa_to_gyu": 0.03337,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "quality_en": {
      "language": "en",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.78662,
          "ecapa_to_gyu": 0.13934,
          "wavlm_l2_to_student": 0.004264
        },
        "fish_only": {
          "wavlm_to_gyu": 0.79297,
          "ecapa_to_gyu": 0.14262,
          "wavlm_l2_to_student": 0.001213
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.79492,
          "ecapa_to_gyu": 0.14579,
          "wavlm_l2_to_student": 0.001219
        },
        "student": {
          "wavlm_to_gyu": 0.7897,
          "ecapa_to_gyu": 0.14409,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "heldout_en": {
      "language": "en",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.87827,
          "ecapa_to_gyu": 0.27655,
          "wavlm_l2_to_student": 0.002963
        },
        "fish_only": {
          "wavlm_to_gyu": 0.87304,
          "ecapa_to_gyu": 0.27091,
          "wavlm_l2_to_student": 0.002313
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.87644,
          "ecapa_to_gyu": 0.27727,
          "wavlm_l2_to_student": 0.00144
        },
        "student": {
          "wavlm_to_gyu": 0.8787,
          "ecapa_to_gyu": 0.27136,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "quality_ja": {
      "language": "ja",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.41693,
          "ecapa_to_gyu": 0.05352,
          "wavlm_l2_to_student": 0.002503
        },
        "fish_only": {
          "wavlm_to_gyu": 0.42396,
          "ecapa_to_gyu": 0.04872,
          "wavlm_l2_to_student": 0.00183
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.4387,
          "ecapa_to_gyu": 0.04906,
          "wavlm_l2_to_student": 0.001561
        },
        "student": {
          "wavlm_to_gyu": 0.43402,
          "ecapa_to_gyu": 0.0485,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "heldout_ja": {
      "language": "ja",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.82201,
          "ecapa_to_gyu": 0.30212,
          "wavlm_l2_to_student": 0.002803
        },
        "fish_only": {
          "wavlm_to_gyu": 0.80131,
          "ecapa_to_gyu": 0.29146,
          "wavlm_l2_to_student": 0.001247
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.80613,
          "ecapa_to_gyu": 0.2902,
          "wavlm_l2_to_student": 0.001717
        },
        "student": {
          "wavlm_to_gyu": 0.79812,
          "ecapa_to_gyu": 0.29142,
          "wavlm_l2_to_student": 0.0
        }
      }
    }
  },
  "student_minus_no_identity": {
    "wavlm_to_gyu": {
      "n": 6,
      "mean": 0.002,
      "p2_5": -0.00932,
      "p97_5": 0.01124,
      "values": [
        0.00044,
        0.01487,
        0.00308,
        0.00043,
        0.01709,
        -0.02389
      ]
    },
    "ecapa_to_gyu": {
      "n": 6,
      "mean": 0.00134,
      "p2_5": -0.00605,
      "p97_5": 0.00835,
      "values": [
        0.01591,
        0.00832,
        0.00475,
        -0.00519,
        -0.00502,
        -0.0107
      ]
    }
  },
  "cross_language_identity_consistency": {
    "quality": {
      "none": {
        "wavlm_pairwise_mean": 0.71532,
        "pairs": [
          0.75396,
          0.77253,
          0.61947
        ]
      },
      "fish_only": {
        "wavlm_pairwise_mean": 0.74156,
        "pairs": [
          0.79739,
          0.76886,
          0.65844
        ]
      },
      "fish_moss": {
        "wavlm_pairwise_mean": 0.74336,
        "pairs": [
          0.78644,
          0.7772,
          0.66644
        ]
      },
      "student": {
        "wavlm_pairwise_mean": 0.7453,
        "pairs": [
          0.78765,
          0.78327,
          0.66497
        ]
      }
    },
    "heldout": {
      "none": {
        "wavlm_pairwise_mean": 0.89438,
        "pairs": [
          0.87356,
          0.88357,
          0.92602
        ]
      },
      "fish_only": {
        "wavlm_pairwise_mean": 0.8987,
        "pairs": [
          0.88179,
          0.90479,
          0.90952
        ]
      },
      "fish_moss": {
        "wavlm_pairwise_mean": 0.89756,
        "pairs": [
          0.87798,
          0.89749,
          0.91721
        ]
      },
      "student": {
        "wavlm_pairwise_mean": 0.89848,
        "pairs": [
          0.87776,
          0.9054,
          0.91227
        ]
      }
    }
  }
}

Fixed score, lyrics, deterministic OmniVoice/SoulX seeds, and reference; only named conditioning varies. v07 identity uses two phrases per KO/EN/JA. Style compares available spectral and latent paths on two KO phrases.
