#!/usr/bin/env python3
"""Train v0.6 shared identity projections on held-out-safe paired views."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gyu_singer.data import acoustic_reference_features
from gyu_singer.model import MultiTeacherIdentityEncoder


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = [json.loads(line) for line in Path("data/manifests/teacher_internal_pairs.jsonl").read_text().splitlines() if line]
    reps = [json.loads(line) for line in Path("data/manifests/teacher_internal_representations.jsonl").read_text().splitlines() if line]
    by_key = {(r["id"], r["teacher"]): r for r in reps}
    usable = [p for p in pairs if p["split"] == "train" and (p["benchmark_id"], "fish_s2_pro") in by_key and (p["benchmark_id"], "moss_local_v15") in by_key]
    model = MultiTeacherIdentityEncoder(96, shared_dim=64).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # One reference is intentional: this corpus supervises the teacher space; student identity
    # is calibrated against the real-GYU reference without inventing speaker labels.
    reference = acoustic_reference_features("data/processed/master/216.wav").to(device)[None]
    history = []
    by_language = {language: [p for p in usable if p["language"] == language] for language in ("ko", "en", "ja")}
    for step in range(1, 801):
        pair = usable[(step - 1) % len(usable)]
        f = torch.load(by_key[(pair["benchmark_id"], "fish_s2_pro")]["path"], weights_only=True).to(device)[None]
        m = torch.load(by_key[(pair["benchmark_id"], "moss_local_v15")]["path"], weights_only=True).to(device)[None]
        fish = F.normalize(model.fish_projection(f), dim=-1)
        moss = F.normalize(model.moss_projection(m), dim=-1)
        shared = F.normalize(fish + moss, dim=-1)
        other_languages = [language for language in by_language if language != pair["language"] and by_language[language]]
        candidates = by_language[other_languages[(step - 1) % len(other_languages)]]
        other = candidates[(step - 1) % len(candidates)]
        of = torch.load(by_key[(other["benchmark_id"], "fish_s2_pro")]["path"], weights_only=True).to(device)[None]
        om = torch.load(by_key[(other["benchmark_id"], "moss_local_v15")]["path"], weights_only=True).to(device)[None]
        other_shared = F.normalize(model.fish_projection(of) + model.moss_projection(om), dim=-1)
        student = model.student(reference)
        weight = torch.tensor([min(pair["fish_trust"], pair["moss_trust"]) * pair["cross_teacher_agreement"]], device=device)
        loss = weight * ((1 - F.cosine_similarity(fish, moss).mean()) + 0.5 * (1 - F.cosine_similarity(shared, student).mean()) + 0.5 * (1 - F.cosine_similarity(shared, other_shared).mean()))
        optimizer.zero_grad(); loss.mean().backward(); optimizer.step()
        if step % 200 == 0:
            history.append({"step": step, "loss": round(float(loss.mean().detach()), 6), "teacher_cos": round(float(F.cosine_similarity(fish, moss).mean().detach()), 6)})
    model.eval().cpu()
    checkpoint = Path("checkpoints/gyu_identity_space_v0.6.pt")
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": {"dim": 96, "shared_dim": 64}, "paired_rows": len(usable), "teachers": ["fish_s2_pro", "moss_local_v15"], "training": "weighted teacher agreement + GYU reference alignment"}, checkpoint)
    report = {"paired_rows": len(pairs), "trained_rows": len(usable), "split_counts": {s: sum(p["split"] == s for p in pairs) for s in ("train", "validation", "test")}, "language_counts": {l: sum(p["language"] == l for p in pairs) for l in ("ko", "en", "ja")}, "shared_dim": 64, "teacher_representations": {"fish": "Fish-S2-Pro-DAC.encoder_hidden", "moss": "MOSS-Audio-Tokenizer-Nano.encoder_hidden_states"}, "history": history, "checkpoint": str(checkpoint), "objective": "weighted teacher agreement + same-GYU cross-language positive + GYU reference alignment", "negative_policy": "no fabricated speaker negatives; cross-view consistency only", "split_policy": "benchmark_id and language stratified; validation/test excluded from optimizer"}
    Path("artifacts/reports/identity_space_v06_training.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/multiteacher_identity_space.md").write_text("# v0.6 shared GYU identity space\n\n" + json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
