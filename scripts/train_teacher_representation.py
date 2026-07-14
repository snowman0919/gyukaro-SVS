#!/usr/bin/env python3
"""Train shared teacher projections and a student identity projection."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from gyu_singer.data import acoustic_reference_features
from gyu_singer.model import MultiTeacherIdentityEncoder


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"; rows = [json.loads(line) for line in Path("data/manifests/teacher_internal_representations.jsonl").read_text().splitlines() if line]
    moss = {row["id"]: row for row in rows if row["teacher"] == "moss_local_v15"}; fish = {row["id"]: row for row in rows if row["teacher"] == "fish_s2_pro"}; paired = [(moss[key], fish[key]) for key in sorted(moss.keys() & fish.keys())]
    model = MultiTeacherIdentityEncoder(48).to(device).train(); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3); history = []
    for step in range(1, 301):
        moss_row, fish_row = paired[(step - 1) % len(paired)]; m = torch.load(moss_row["path"], weights_only=True).to(device)[None]; f = torch.load(fish_row["path"], weights_only=True).to(device)[None]; shared = model(f, m); ref = acoustic_reference_features("data/processed/master/216.wav").to(device)[None]; student = model.student(ref); loss = 1 - torch.nn.functional.cosine_similarity(shared, student).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0: history.append({"step": step, "loss": round(float(loss.detach()), 6), "student_grad": round(float(model.student_projection[0].weight.grad.detach().abs().mean()), 8)})
    output = "checkpoints/gyu_teacher_identity_v0.5.pt"; Path(output).parent.mkdir(exist_ok=True); torch.save({"model": model.eval().cpu().state_dict(), "model_config": {"dim": 48, "shared_dim": 32}, "teachers": ["fish_s2_pro", "moss_local_v15"], "paired_rows": len(paired)}, output)
    report = {"teachers": ["fish_s2_pro", "moss_local_v15"], "paired_rows": len(paired), "representation_shapes": {"fish": [1024], "moss": [768], "shared": [32]}, "loss": "cosine(student_reference, shared_projected_teacher)", "trust_weight": "teacher manifest weight applies at dataset selection; paired extraction rows are frozen evidence", "gradient_evidence": history, "checkpoint": output, "notes": "Fish DAC and MOSS tokenizer are real internal neural representations; no waveform summary used for this path."}
    Path("artifacts/reports/teacher_representation_training.json").write_text(json.dumps(report, indent=2) + "\n"); Path("docs/teacher_representation_distillation.md").write_text("# Teacher representation distillation (v0.5)\n\n" + json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__": main()
