#!/usr/bin/env python3
"""Held-out shared-space checks: cross-teacher agreement and leakage proxies."""
import collections, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from gyu_singer.model import MultiTeacherIdentityEncoder


def main():
    pairs = [json.loads(x) for x in Path("data/manifests/teacher_internal_pairs.jsonl").read_text().splitlines() if x]
    reps = [json.loads(x) for x in Path("data/manifests/teacher_internal_representations.jsonl").read_text().splitlines() if x]
    lookup = {(r["id"], r["teacher"]): r for r in reps}
    saved = torch.load("checkpoints/gyu_identity_space_v0.6.pt", map_location="cpu", weights_only=False)
    model = MultiTeacherIdentityEncoder(**saved["model_config"]).eval(); model.load_state_dict(saved["model"])
    values = []
    with torch.inference_mode():
        for row in pairs:
            fish = torch.load(lookup[(row["benchmark_id"], "fish_s2_pro")]["path"], weights_only=True).float().flatten()[None]
            moss = torch.load(lookup[(row["benchmark_id"], "moss_local_v15")]["path"], weights_only=True).float().flatten()[None]
            f = F.normalize(model.fish_projection(fish), dim=-1)[0].numpy(); m = F.normalize(model.moss_projection(moss), dim=-1)[0].numpy(); s = F.normalize(torch.from_numpy(f + m), dim=-1).numpy()
            values.append({"id": row["benchmark_id"], "language": row["language"], "split": row["split"], "fish": f, "moss": m, "shared": s})
    train = [v for v in values if v["split"] == "train"]; test = [v for v in values if v["split"] == "test"]
    def centroid(rows, key, label_key, label):
        xs = [v[key] for v in rows if v[label_key] == label]
        return np.mean(xs, axis=0) if xs else None
    same_teacher = [float(np.dot(v["fish"], v["moss"])) for v in test]
    langs = sorted({v["language"] for v in values})
    cross_language = {}
    for a in langs:
        for b in langs:
            ca, cb = centroid(test, "shared", "language", a), centroid(test, "shared", "language", b)
            if ca is not None and cb is not None: cross_language[f"{a}-{b}"] = round(float(np.dot(ca / np.linalg.norm(ca), cb / np.linalg.norm(cb))), 5)
    # Nearest train centroid probes identify leakage without fitting a flexible classifier.
    teacher_centroids = {t: np.mean([v[t] for v in train], axis=0) for t in ("fish", "moss")}
    teacher_samples = [(v["fish"], "fish") for v in test] + [(v["moss"], "moss") for v in test]
    teacher_leakage = sum(max(teacher_centroids, key=lambda t: float(np.dot(sample, teacher_centroids[t]))) == label for sample, label in teacher_samples) / max(1, len(teacher_samples))
    language_centroids = {l: centroid(train, "shared", "language", l) for l in langs}
    language_centroids = {l: c / np.linalg.norm(c) for l, c in language_centroids.items() if c is not None}
    language_accuracy = sum(max(language_centroids, key=lambda l: float(np.dot(v["shared"], language_centroids[l]))) == v["language"] for v in test if language_centroids) / max(1, len(test))
    report = {"train_rows": len(train), "test_rows": len(test), "test_languages": dict(collections.Counter(v["language"] for v in test)), "same_gyu_cross_teacher_cosine": {"mean": round(float(np.mean(same_teacher)), 5), "median": round(float(np.median(same_teacher)), 5), "values": [round(x, 5) for x in same_teacher]}, "same_gyu_cross_language_cosine": cross_language, "teacher_identification_leakage_nearest_centroid": round(float(teacher_leakage), 5), "language_clustering_nearest_centroid_accuracy": round(float(language_accuracy), 5), "interpretation": "Leakage and clustering are reported as diagnostics; no pretty projection is treated as primary evidence."}
    Path("artifacts/reports/identity_space_v06_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/multiteacher_identity_space.md").write_text("# v0.6 shared GYU identity space\n\n" + json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__": main()
