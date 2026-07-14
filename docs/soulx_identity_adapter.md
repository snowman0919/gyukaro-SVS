# v06 SoulX identity adapter

{
  "scores": {
    "quality_ko": {
      "language": "ko",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.67574,
          "ecapa_to_gyu": 0.21374,
          "wavlm_l2_to_student": 0.001886
        },
        "fish_only": {
          "wavlm_to_gyu": 0.67462,
          "ecapa_to_gyu": 0.21651,
          "wavlm_l2_to_student": 0.00138
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.6714,
          "ecapa_to_gyu": 0.21331,
          "wavlm_l2_to_student": 0.002052
        },
        "student": {
          "wavlm_to_gyu": 0.67797,
          "ecapa_to_gyu": 0.21326,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "heldout_ko": {
      "language": "ko",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.69011,
          "ecapa_to_gyu": 0.03022,
          "wavlm_l2_to_student": 0.001992
        },
        "fish_only": {
          "wavlm_to_gyu": 0.69409,
          "ecapa_to_gyu": 0.03411,
          "wavlm_l2_to_student": 0.001971
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.68584,
          "ecapa_to_gyu": 0.02314,
          "wavlm_l2_to_student": 0.002414
        },
        "student": {
          "wavlm_to_gyu": 0.69131,
          "ecapa_to_gyu": 0.02897,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "quality_en": {
      "language": "en",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.7903,
          "ecapa_to_gyu": 0.13698,
          "wavlm_l2_to_student": 0.001033
        },
        "fish_only": {
          "wavlm_to_gyu": 0.78436,
          "ecapa_to_gyu": 0.13405,
          "wavlm_l2_to_student": 0.002404
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.79413,
          "ecapa_to_gyu": 0.13326,
          "wavlm_l2_to_student": 0.000989
        },
        "student": {
          "wavlm_to_gyu": 0.78978,
          "ecapa_to_gyu": 0.13336,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "heldout_en": {
      "language": "en",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.87287,
          "ecapa_to_gyu": 0.26513,
          "wavlm_l2_to_student": 0.001426
        },
        "fish_only": {
          "wavlm_to_gyu": 0.87852,
          "ecapa_to_gyu": 0.26981,
          "wavlm_l2_to_student": 0.001388
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.87664,
          "ecapa_to_gyu": 0.26583,
          "wavlm_l2_to_student": 0.001115
        },
        "student": {
          "wavlm_to_gyu": 0.87639,
          "ecapa_to_gyu": 0.27036,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "quality_ja": {
      "language": "ja",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.42076,
          "ecapa_to_gyu": 0.05145,
          "wavlm_l2_to_student": 0.001193
        },
        "fish_only": {
          "wavlm_to_gyu": 0.42122,
          "ecapa_to_gyu": 0.05327,
          "wavlm_l2_to_student": 0.001602
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.41747,
          "ecapa_to_gyu": 0.05359,
          "wavlm_l2_to_student": 0.002077
        },
        "student": {
          "wavlm_to_gyu": 0.43118,
          "ecapa_to_gyu": 0.0561,
          "wavlm_l2_to_student": 0.0
        }
      }
    },
    "heldout_ja": {
      "language": "ja",
      "variants": {
        "none": {
          "wavlm_to_gyu": 0.81301,
          "ecapa_to_gyu": 0.3067,
          "wavlm_l2_to_student": 0.000905
        },
        "fish_only": {
          "wavlm_to_gyu": 0.81048,
          "ecapa_to_gyu": 0.30389,
          "wavlm_l2_to_student": 0.001345
        },
        "fish_moss": {
          "wavlm_to_gyu": 0.80998,
          "ecapa_to_gyu": 0.29965,
          "wavlm_l2_to_student": 0.00103
        },
        "student": {
          "wavlm_to_gyu": 0.80795,
          "ecapa_to_gyu": 0.30315,
          "wavlm_l2_to_student": 0.0
        }
      }
    }
  },
  "student_minus_no_identity": {
    "wavlm_to_gyu": {
      "n": 6,
      "mean": 0.00196,
      "p2_5": -0.00158,
      "p97_5": 0.00561,
      "values": [
        0.00223,
        0.0012,
        -0.00052,
        0.00352,
        0.01042,
        -0.00506
      ]
    },
    "ecapa_to_gyu": {
      "n": 6,
      "mean": 0.00016,
      "p2_5": -0.00253,
      "p97_5": 0.00301,
      "values": [
        -0.00048,
        -0.00125,
        -0.00362,
        0.00523,
        0.00465,
        -0.00355
      ]
    }
  },
  "cross_language_identity_consistency": {
    "quality": {
      "none": {
        "wavlm_pairwise_mean": 0.72771,
        "pairs": [
          0.77349,
          0.76588,
          0.64377
        ]
      },
      "fish_only": {
        "wavlm_pairwise_mean": 0.73831,
        "pairs": [
          0.78371,
          0.76588,
          0.66534
        ]
      },
      "fish_moss": {
        "wavlm_pairwise_mean": 0.72618,
        "pairs": [
          0.77451,
          0.7722,
          0.63183
        ]
      },
      "student": {
        "wavlm_pairwise_mean": 0.73263,
        "pairs": [
          0.77126,
          0.778,
          0.64865
        ]
      }
    },
    "heldout": {
      "none": {
        "wavlm_pairwise_mean": 0.9017,
        "pairs": [
          0.88111,
          0.89815,
          0.92584
        ]
      },
      "fish_only": {
        "wavlm_pairwise_mean": 0.90167,
        "pairs": [
          0.87704,
          0.90406,
          0.92391
        ]
      },
      "fish_moss": {
        "wavlm_pairwise_mean": 0.89703,
        "pairs": [
          0.87079,
          0.89484,
          0.92546
        ]
      },
      "student": {
        "wavlm_pairwise_mean": 0.9008,
        "pairs": [
          0.88099,
          0.89665,
          0.92476
        ]
      }
    }
  }
}

Fixed score, lyrics, deterministic OmniVoice/SoulX seeds, and reference; only named conditioning varies. v06 identity uses two phrases per KO/EN/JA. Style compares available spectral and latent paths on two KO phrases.
