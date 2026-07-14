# v06 latent acoustic-style adapter

{
  "style_ablation": {
    "quality_ko": {
      "v05_spectral": {
        "dark_minus_neutral_wavlm_l2": 0.005817,
        "dark_minus_neutral_centroid_hz": 24.105
      },
      "v06_spectral_only": {
        "dark_minus_neutral_wavlm_l2": 0.006603,
        "dark_minus_neutral_centroid_hz": 23.612
      },
      "v06_latent": {
        "dark_minus_neutral_wavlm_l2": 0.006392,
        "dark_minus_neutral_centroid_hz": 19.324
      },
      "latent_vs_spectral_only_dark_wavlm_l2": 0.002592
    },
    "heldout_ko": {
      "v05_spectral": {
        "dark_minus_neutral_wavlm_l2": 0.006096,
        "dark_minus_neutral_centroid_hz": 21.138
      },
      "v06_spectral_only": {
        "dark_minus_neutral_wavlm_l2": 0.005029,
        "dark_minus_neutral_centroid_hz": 29.705
      },
      "v06_latent": {
        "dark_minus_neutral_wavlm_l2": 0.00566,
        "dark_minus_neutral_centroid_hz": 26.032
      },
      "latent_vs_spectral_only_dark_wavlm_l2": 0.002966
    }
  },
  "caveat": "Output differences are evidence only; intended semantic direction requires separate acoustic-proxy checks."
}
