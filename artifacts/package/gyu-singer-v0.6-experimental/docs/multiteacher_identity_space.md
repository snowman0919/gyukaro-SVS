# v0.6 shared GYU identity space

{
  "train_rows": 185,
  "test_rows": 3,
  "test_languages": {
    "en": 1,
    "ja": 1,
    "ko": 1
  },
  "same_gyu_cross_teacher_cosine": {
    "mean": 0.99986,
    "median": 0.99986,
    "values": [
      0.99986,
      0.99988,
      0.99985
    ]
  },
  "same_gyu_cross_language_cosine": {
    "en-en": 1.0,
    "en-ja": 0.99996,
    "en-ko": 0.99996,
    "ja-en": 0.99996,
    "ja-ja": 1.0,
    "ja-ko": 0.99997,
    "ko-en": 0.99996,
    "ko-ja": 0.99997,
    "ko-ko": 1.0
  },
  "teacher_identification_leakage_nearest_centroid": 0.5,
  "language_clustering_nearest_centroid_accuracy": 0.66667,
  "interpretation": "Leakage and clustering are reported as diagnostics; no pretty projection is treated as primary evidence."
}
