# v0.5 evaluation

{
  "checkpoint": "checkpoints/gyu_prosody_v0.5.pt",
  "heldout_rows": 5,
  "target": "real GYU RMVPE F0",
  "metrics": {
    "correlation_mean": 0.2316,
    "pitch_mae_cents_median": 199.73,
    "nominal_mae_cents_median": 197.27,
    "improvement_cents_median": -9.29
  },
  "rows": [
    {
      "id": "gyu_real_000153",
      "correlation": 0.2787,
      "pitch_mae_cents": 161.48,
      "nominal_pitch_mae_cents": 151.2,
      "improvement_cents": -10.28
    },
    {
      "id": "gyu_real_000170",
      "correlation": 0.5986,
      "pitch_mae_cents": 199.73,
      "nominal_pitch_mae_cents": 131.27,
      "improvement_cents": -68.46
    },
    {
      "id": "gyu_real_000187",
      "correlation": 0.1726,
      "pitch_mae_cents": 188.75,
      "nominal_pitch_mae_cents": 322.71,
      "improvement_cents": 133.96
    },
    {
      "id": "gyu_real_000204",
      "correlation": -0.1912,
      "pitch_mae_cents": 217.14,
      "nominal_pitch_mae_cents": 207.85,
      "improvement_cents": -9.29
    },
    {
      "id": "gyu_real_000221",
      "correlation": 0.2991,
      "pitch_mae_cents": 200.17,
      "nominal_pitch_mae_cents": 197.27,
      "improvement_cents": -2.9
    }
  ]
}
