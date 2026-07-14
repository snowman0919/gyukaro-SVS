# v0.8 production evaluation

{
  "production": {
    "backend": "gyu-singer-v0.8",
    "prosody": "v0.5 real-GYU controller",
    "identity": "v0.7 actual-SoulX-latent identity adapter",
    "style": "v0.7 actual-SoulX-latent style adapter",
    "decoder": "SoulX phrase decoder",
    "per_note_tts": false,
    "waveform_pitch_shift": false
  },
  "prosody_selection": {
    "score_rows": 24,
    "independent_from_target_f0": true,
    "aggregates": {
      "nominal_verified_score": {
        "f0_correlation": 0.7337,
        "pitch_mae_cents": 137.8683,
        "log_f0_rmse": 0.1327,
        "note_onset_residual_cents": 183.5037,
        "transition_contour_error_cents": 205.8404,
        "vibrato_rate_error_hz": 0.781,
        "vibrato_extent_error_cents": 71.1529
      },
      "v0.4_synthetic_controller": {
        "f0_correlation": 0.7328,
        "pitch_mae_cents": 136.0258,
        "log_f0_rmse": 0.1325,
        "note_onset_residual_cents": 183.185,
        "transition_contour_error_cents": 204.9058,
        "vibrato_rate_error_hz": 0.531,
        "vibrato_extent_error_cents": 68.4121
      },
      "v0.5_real_gyu_controller": {
        "f0_correlation": 0.7273,
        "pitch_mae_cents": 134.5467,
        "log_f0_rmse": 0.1317,
        "note_onset_residual_cents": 184.4754,
        "transition_contour_error_cents": 205.6442,
        "vibrato_rate_error_hz": 0.464,
        "vibrato_extent_error_cents": 69.8588
      },
      "v0.6_verified_plus_reconstructed_controller": {
        "f0_correlation": 0.7278,
        "pitch_mae_cents": 135.1762,
        "log_f0_rmse": 0.1315,
        "note_onset_residual_cents": 182.605,
        "transition_contour_error_cents": 204.2392,
        "vibrato_rate_error_hz": 0.4607,
        "vibrato_extent_error_cents": 69.1988
      }
    },
    "selected": "v0.5_real_gyu_controller",
    "reason": "lowest pitch MAE and prior production evidence; v0.6 wins some transition metrics but is not consistent"
  },
  "identity_selection": {
    "v0.6_student_minus_none": {
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
    "v0.7_student_minus_none": {
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
    "v0.7_cross_language_consistency": {
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
    },
    "selected": "v0.7_real_latent",
    "reason": "actual decoder-latent training, positive mean WavLM/ECAPA deltas, and higher student cross-language consistency; confidence intervals cross zero, so quality gain remains modest"
  },
  "style_selection": {
    "selected": "v0.7_real_latent",
    "semantic_status": {
      "neutral": "neutral",
      "soft": "relative_style_A_unverified",
      "breathy": "breathiness_proxy",
      "energetic": "energy_proxy",
      "dark": "relative_style_C_unverified",
      "bright": "relative_style_B_unverified"
    },
    "validated_directions": {
      "quality": true,
      "heldout": true,
      "korean": true
    },
    "limitation": "breathy and energetic proxies validate; soft, dark, and bright remain explicitly relabeled relative styles"
  },
  "renders": {
    "ko": {
      "path": "artifacts/reports/v08_quality_ko.wav",
      "sample_rate": 48000,
      "channels": 1,
      "duration_sec": 9.92
    },
    "en": {
      "path": "artifacts/reports/v08_quality_en.wav",
      "sample_rate": 48000,
      "channels": 1,
      "duration_sec": 9.92
    },
    "ja": {
      "path": "artifacts/reports/v08_quality_ja.wav",
      "sample_rate": 48000,
      "channels": 1,
      "duration_sec": 9.92
    }
  }
}
