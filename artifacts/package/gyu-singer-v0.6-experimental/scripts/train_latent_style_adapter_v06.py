#!/usr/bin/env python3
"""Weakly supervise SoulX latent style controls from the existing style corpus.

Only adapter parameters and a throw-away calibration head are optimized. The
teacher speech is style evidence, never a GYU singing target.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from gyu_singer.inference.latent_adapter import SoulXLatentAdapter


def style_vector(label: str, device: str) -> torch.Tensor:
    value = torch.zeros(64, device=device); value[{"dark": 4, "emotional": 3}.get(label, 0)] = 1.0; return value


def main() -> None:
    rows = [json.loads(line) for line in Path("data/manifests/teacher_style_supplement_weighted.jsonl").read_text().splitlines() if line]
    device = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(606)
    model = SoulXLatentAdapter().to(device).train()
    classifier = torch.nn.Linear(512, 2).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=2e-3, weight_decay=1e-4)
    hidden = torch.zeros(1, 1, 512, device=device); identity_a = torch.zeros(1, 64, device=device); identity_b = torch.ones(1, 64, device=device) * .1
    history = []
    for step in range(1, 401):
        row = rows[(step - 1) % len(rows)]; style = style_vector(row.get("style", "dark"), device)[None]
        label = torch.tensor([0 if row.get("style") == "dark" else 1], device=device)
        dark = model(hidden, identity_a, style).mean(1); alt = model(hidden, identity_b, style).mean(1)
        classification = F.cross_entropy(classifier(dark), label)
        identity_invariance = F.mse_loss(classifier(dark), classifier(alt))
        loss = float(row.get("trust_weight", .1)) * (classification + .1 * identity_invariance)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0: history.append({"step": step, "loss": round(float(loss.detach()), 6), "classification": round(float(classification.detach()), 6)})
    checkpoint = Path("checkpoints/gyu_latent_adapter_v0.6.pt"); model.eval().cpu()
    torch.save({"model": model.state_dict(), "config": {"identity_dim": 64, "style_dim": 64, "hidden_dim": 512}, "training": "teacher-style classification + identity invariance; 32 weak style rows", "rows": len(rows)}, checkpoint)
    report = {"rows": len(rows), "styles": {label: sum(r.get("style") == label for r in rows) for label in ("dark", "emotional")}, "objective": "trust-weighted style classification plus identity invariance", "history": history, "checkpoint": str(checkpoint), "caveat": "teacher speech style evidence; not real-GYU singing supervision"}
    Path("artifacts/reports/latent_style_training_v0.6.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__": main()
