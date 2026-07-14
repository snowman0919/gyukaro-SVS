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
  "loss": "cosine(student_reference, shared_projected_teacher)",
  "trust_weight": "teacher manifest weight applies at dataset selection; paired extraction rows are frozen evidence",
  "gradient_evidence": [
    {
      "step": 100,
      "loss": 5.2e-05,
      "student_grad": 3.525e-05
    },
    {
      "step": 200,
      "loss": 4e-06,
      "student_grad": 1.096e-05
    },
    {
      "step": 300,
      "loss": 1e-06,
      "student_grad": 5.47e-06
    }
  ],
  "checkpoint": "checkpoints/gyu_teacher_identity_v0.5.pt",
  "notes": "Fish DAC and MOSS tokenizer are real internal neural representations; no waveform summary used for this path."
}
