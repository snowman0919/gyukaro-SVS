# Real SoulX latent dataset

{
  "rows": 117,
  "source_types": {
    "real_gyu": 24,
    "accepted_pseudo": 45,
    "teacher_style": 48
  },
  "languages": {
    "ko": 87,
    "en": 17,
    "ja": 13
  },
  "styles": {
    "neutral": 77,
    "soft": 8,
    "breathy": 8,
    "energetic": 8,
    "dark": 8,
    "bright": 8
  },
  "tensor": "actual pre-adapter SoulXSingerSVC.infer_segment.gt_decoder_inp",
  "all_nonzero": true,
  "teacher_audio_proxy_medians_inferred": {
    "neutral": {
      "spectral_centroid_hz": 562.416758,
      "rms": 0.039106,
      "high_frequency_ratio_4khz": 0.000705
    },
    "soft": {
      "spectral_centroid_hz": 630.624761,
      "rms": 0.037273,
      "high_frequency_ratio_4khz": 0.000882
    },
    "breathy": {
      "spectral_centroid_hz": 590.07275,
      "rms": 0.038524,
      "high_frequency_ratio_4khz": 0.000763
    },
    "energetic": {
      "spectral_centroid_hz": 611.8797,
      "rms": 0.075795,
      "high_frequency_ratio_4khz": 0.000587
    },
    "dark": {
      "spectral_centroid_hz": 530.316931,
      "rms": 0.038187,
      "high_frequency_ratio_4khz": 0.001101
    },
    "bright": {
      "spectral_centroid_hz": 594.946481,
      "rms": 0.070391,
      "high_frequency_ratio_4khz": 0.000791
    }
  }
}

Every row preserves audio, source type, language, style, identity target, trust, exact latent path, tensor statistics, and SoulX revision. Teacher/pseudo rows remain lower trust than real GYU singing. Acoustic measurements are explicitly inferred proxies, not proof of perceived semantics.
