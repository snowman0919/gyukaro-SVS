# v0.6 shared GYU identity space

{
  "train_rows": 191,
  "test_rows": 26,
  "test_languages": {
    "en": 12,
    "ja": 6,
    "ko": 8
  },
  "same_gyu_cross_teacher_cosine": {
    "mean": 0.94029,
    "median": 0.94418,
    "values": [
      0.94729,
      0.94841,
      0.89704,
      0.93236,
      0.94141,
      0.94884,
      0.95352,
      0.97262,
      0.93023,
      0.92426,
      0.9414,
      0.91974,
      0.88347,
      0.94901,
      0.88947,
      0.90336,
      0.97132,
      0.94493,
      0.98037,
      0.94242,
      0.92404,
      0.96417,
      0.94344,
      0.97708,
      0.94764,
      0.96973
    ]
  },
  "same_gyu_cross_language_cosine": {
    "en-en": 1.0,
    "en-ja": 0.96139,
    "en-ko": 0.98263,
    "ja-en": 0.96139,
    "ja-ja": 1.0,
    "ja-ko": 0.94154,
    "ko-en": 0.98263,
    "ko-ja": 0.94154,
    "ko-ko": 1.0
  },
  "teacher_identification_leakage_nearest_centroid": 0.5,
  "language_clustering_nearest_centroid_accuracy": 0.42308,
  "interpretation": "Leakage and clustering are reported as diagnostics; no pretty projection is treated as primary evidence."
}
