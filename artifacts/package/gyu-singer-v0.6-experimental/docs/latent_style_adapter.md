# v0.6 latent acoustic-style adapter

{
  "style_ablation": {
    "quality_ko": {
      "v05_spectral": {
        "dark_minus_neutral_wavlm_l2": 0.005817,
        "dark_minus_neutral_centroid_hz": 24.105
      },
      "v06_spectral_only": {
        "dark_minus_neutral_wavlm_l2": 0.006871,
        "dark_minus_neutral_centroid_hz": 1.393
      },
      "v06_latent": {
        "dark_minus_neutral_wavlm_l2": 0.007594,
        "dark_minus_neutral_centroid_hz": 1.451
      },
      "latent_vs_spectral_only_dark_wavlm_l2": 0.001663
    },
    "heldout_ko": {
      "v05_spectral": {
        "dark_minus_neutral_wavlm_l2": 0.006096,
        "dark_minus_neutral_centroid_hz": 21.138
      },
      "v06_spectral_only": {
        "dark_minus_neutral_wavlm_l2": 0.005287,
        "dark_minus_neutral_centroid_hz": 4.551
      },
      "v06_latent": {
        "dark_minus_neutral_wavlm_l2": 0.005754,
        "dark_minus_neutral_centroid_hz": 3.51
      },
      "latent_vs_spectral_only_dark_wavlm_l2": 0.004911
    }
  },
  "caveat": "This measures output effects, not a claim that the weak teacher-style supervision establishes every style preset semantically."
}
