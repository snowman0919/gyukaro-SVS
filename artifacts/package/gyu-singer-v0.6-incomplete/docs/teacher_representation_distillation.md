# Teacher representation distillation (v0.5)

{
  "teachers": [
    "fish_s2_pro",
    "moss_local_v15"
  ],
  "paired_rows": 4,
  "representation_shapes": {
    "fish": [
      1024
    ],
    "moss": [
      768
    ],
    "shared": [
      32
    ]
  },
  "loss": "trust_weight * cosine(student_reference, shared_projected_teacher)",
  "trust_weight": "minimum paired manifest trust_weight scales objective",
  "gradient_evidence": [
    {
      "step": 100,
      "loss": 1.1e-05,
      "student_grad": 8.08e-06
    },
    {
      "step": 200,
      "loss": 1e-06,
      "student_grad": 2.49e-06
    },
    {
      "step": 300,
      "loss": 0.0,
      "student_grad": 1.47e-06
    }
  ],
  "checkpoint": "checkpoints/gyu_teacher_identity_v0.5.pt",
  "notes": "Fish DAC and MOSS tokenizer are real internal neural representations; no waveform summary used for this path."
}
