# v0.6 shared GYU identity space

{
  "paired_rows": 191,
  "split_counts": {
    "train": 165,
    "validation": 9,
    "test": 17
  },
  "language_counts": {
    "ko": 99,
    "en": 77,
    "ja": 15
  },
  "shared_dim": 64,
  "teacher_representations": {
    "fish": "Fish-S2-Pro-DAC.encoder_hidden",
    "moss": "MOSS-Audio-Tokenizer-Nano.encoder_hidden_states"
  },
  "history": [
    {
      "step": 200,
      "loss": 0.000281,
      "teacher_cos": 0.998518
    },
    {
      "step": 400,
      "loss": 5.8e-05,
      "teacher_cos": 0.999721
    },
    {
      "step": 600,
      "loss": 4.9e-05,
      "teacher_cos": 0.999795
    },
    {
      "step": 800,
      "loss": 4.8e-05,
      "teacher_cos": 0.999765
    }
  ],
  "checkpoint": "checkpoints/gyu_identity_space_v0.6.pt",
  "negative_policy": "no fabricated speaker negatives; cross-view consistency only"
}
