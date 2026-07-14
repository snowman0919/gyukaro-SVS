# v0.6 SoulX identity adapter

{
  "paths": {
    "ko:none": "artifacts/reports/v06_ablation_none_ko.wav",
    "ko:fish_only": "artifacts/reports/v06_ablation_fish_only_ko.wav",
    "ko:fish_moss": "artifacts/reports/v06_ablation_fish_moss_ko.wav",
    "ko:student": "artifacts/reports/v06_ablation_student_ko.wav",
    "en:none": "artifacts/reports/v06_ablation_none_en.wav",
    "en:fish_only": "artifacts/reports/v06_ablation_fish_only_en.wav",
    "en:fish_moss": "artifacts/reports/v06_ablation_fish_moss_en.wav",
    "en:student": "artifacts/reports/v06_ablation_student_en.wav",
    "ja:none": "artifacts/reports/v06_ablation_none_ja.wav",
    "ja:fish_only": "artifacts/reports/v06_ablation_fish_only_ja.wav",
    "ja:fish_moss": "artifacts/reports/v06_ablation_fish_moss_ja.wav",
    "ja:student": "artifacts/reports/v06_ablation_student_ja.wav",
    "ko:latent_style_zero": "artifacts/reports/v06_ablation_latent_style_zero.wav"
  },
  "metric": "WavLM and ECAPA speaker cosine; same language score/content/F0 per variant",
  "identity_ablation_modes": [
    "none",
    "fish_only",
    "fish_moss",
    "student"
  ],
  "identity_effect": true,
  "latent_style_effect": true,
  "metrics": {
    "wavlm_to_gyu": {
      "ko:none": 0.67829,
      "ko:fish_only": 0.68047,
      "ko:fish_moss": 0.67197,
      "ko:student": 0.67862,
      "en:none": 0.78722,
      "en:fish_only": 0.79391,
      "en:fish_moss": 0.79697,
      "en:student": 0.79676,
      "ja:none": 0.39711,
      "ja:fish_only": 0.40424,
      "ja:fish_moss": 0.41783,
      "ja:student": 0.41965,
      "ko:latent_style_zero": 0.6837
    },
    "ecapa_to_gyu": {
      "ko:none": 0.20495,
      "ko:fish_only": 0.2043,
      "ko:fish_moss": 0.20286,
      "ko:student": 0.20902,
      "en:none": 0.12018,
      "en:fish_only": 0.11701,
      "en:fish_moss": 0.11758,
      "en:student": 0.11716,
      "ja:none": 0.04493,
      "ja:fish_only": 0.04621,
      "ja:fish_moss": 0.0443,
      "ja:student": 0.04588,
      "ko:latent_style_zero": 0.20585
    },
    "audio_rms": {
      "ko:none": 0.158994,
      "ko:fish_only": 0.159058,
      "ko:fish_moss": 0.159142,
      "ko:student": 0.158595,
      "en:none": 0.13194,
      "en:fish_only": 0.132089,
      "en:fish_moss": 0.132139,
      "en:student": 0.13206,
      "ja:none": 0.131794,
      "ja:fish_only": 0.131635,
      "ja:fish_moss": 0.132066,
      "ja:student": 0.132501,
      "ko:latent_style_zero": 0.15849
    },
    "pairwise_l2_to_student": {
      "ko:none": 0.002839,
      "ko:fish_only": 0.002177,
      "ko:fish_moss": 0.003475,
      "ko:student": 0.0,
      "en:none": 0.003061,
      "en:fish_only": 0.002665,
      "en:fish_moss": 0.00176,
      "en:student": 0.0,
      "ja:none": 0.002144,
      "ja:fish_only": 0.001809,
      "ja:fish_moss": 0.002115,
      "ja:student": 0.0
    }
  },
  "note": "Teacher modes use one held-out-safe paired Fish/MOSS representation as conditioning ablation; student mode uses real-GYU reference projection. Higgs hidden is unavailable.",
  "three_language_summary": {
    "wavlm_to_gyu": {
      "none_mean": 0.62087,
      "student_mean": 0.63168,
      "student_minus_none": 0.0108
    },
    "ecapa_to_gyu": {
      "none_mean": 0.12335,
      "student_mean": 0.12402,
      "student_minus_none": 0.00067
    }
  }
}
