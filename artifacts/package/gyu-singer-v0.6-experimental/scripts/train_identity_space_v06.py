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
    torch.manual_seed(606)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = [json.loads(line) for line in Path("data/manifests/teacher_internal_pairs.jsonl").read_text().splitlines() if line]
    reps = [json.loads(line) for line in Path("data/manifests/teacher_internal_representations.jsonl").read_text().splitlines() if line]
    by_key = {(r["id"], r["teacher"]): r for r in reps}
    usable = [p for p in pairs if p["split"] == "train" and (p["benchmark_id"], "fish_s2_pro") in by_key and (p["benchmark_id"], "moss_local_v15") in by_key]
    model = MultiTeacherIdentityEncoder(96, shared_dim=64).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # This corpus has one real speaker, so it cannot supply honest speaker negatives.  Learn
    # matched Fish/MOSS views with a variance-preserving cross-correlation objective instead of
    # pulling every teacher vector directly to the one GYU reference (which collapsed v0.6).
    reference = acoustic_reference_features("data/processed/master/216.wav").to(device)[None]
    history = []
    fish_input = torch.stack([torch.load(by_key[(pair["benchmark_id"], "fish_s2_pro")]["path"], weights_only=True).float() for pair in usable]).to(device)
    moss_input = torch.stack([torch.load(by_key[(pair["benchmark_id"], "moss_local_v15")]["path"], weights_only=True).float() for pair in usable]).to(device)
    weights = torch.tensor([min(pair["fish_trust"], pair["moss_trust"]) * pair["cross_teacher_agreement"] for pair in usable], device=device).clamp_min(.001)
    weights = weights / weights.sum()
    for step in range(1, 801):
        fish_raw = model.fish_projection(fish_input)
        moss_raw = model.moss_projection(moss_input)
        fish = F.normalize(fish_raw, dim=-1)
        moss = F.normalize(moss_raw, dim=-1)
        fish_z = (fish_raw - (weights[:, None] * fish_raw).sum(0)) / fish_raw.std(0).clamp_min(1e-4)
        moss_z = (moss_raw - (weights[:, None] * moss_raw).sum(0)) / moss_raw.std(0).clamp_min(1e-4)
        correlation = fish_z.T @ (moss_z * weights[:, None])
        diagonal = torch.diagonal(correlation).sub(1).pow(2).sum()
        off_diagonal = (correlation - torch.diag(torch.diagonal(correlation))).pow(2).sum()
        barlow = diagonal + .005 * off_diagonal
        alignment = (weights * (1 - F.cosine_similarity(fish, moss))).sum()
        moments = (fish_raw.mean(0) - moss_raw.mean(0)).pow(2).mean() + (fish_raw.std(0) - moss_raw.std(0)).pow(2).mean()
        target = F.normalize((fish + moss).mean(0, keepdim=True).detach(), dim=-1)
        student = model.student(reference)
        student_alignment = 1 - F.cosine_similarity(student, target).mean()
        loss = barlow + .1 * alignment + .05 * moments + .1 * student_alignment
        optimizer.zero_grad(); loss.mean().backward(); optimizer.step()
        if step % 200 == 0:
            history.append({"step": step, "loss": round(float(loss.detach()), 6), "teacher_cos": round(float(F.cosine_similarity(fish, moss).mean().detach()), 6), "barlow": round(float(barlow.detach()), 6)})
    model.eval().cpu()
    checkpoint = Path("checkpoints/gyu_identity_space_v0.6.pt")
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": {"dim": 96, "shared_dim": 64}, "paired_rows": len(usable), "teachers": ["fish_s2_pro", "moss_local_v15"], "training": "trust-weighted Barlow cross-view alignment + student calibration"}, checkpoint)
    report = {"paired_rows": len(pairs), "semantic_text_reference_groups": len({p["semantic_group_id"] for p in pairs}), "trained_rows": len(usable), "split_counts": {s: sum(p["split"] == s for p in pairs) for s in ("train", "validation", "test")}, "language_counts": {l: sum(p["language"] == l for p in pairs) for l in ("ko", "en", "ja")}, "shared_dim": 64, "teacher_representations": {"fish": "Fish-S2-Pro-DAC.encoder_hidden", "moss": "MOSS-Audio-Tokenizer-Nano.encoder_hidden_states"}, "history": history, "checkpoint": str(checkpoint), "objective": "trust-weighted Barlow cross-view agreement + matched-view alignment + student calibration", "negative_policy": "no fabricated speaker negatives; cross-view consistency with variance preservation", "split_policy": "text/reference semantic group and language stratified; validation/test groups excluded from optimizer"}
    Path("artifacts/reports/identity_space_v06_training.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/multiteacher_identity_space.md").write_text("# v0.6 shared GYU identity space\n\n" + json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
