#!/usr/bin/env python3
"""Cache real production-path SoulX gt_decoder_inp tensors with provenance."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from scipy.signal import welch

from gyu_singer.inference.soulx import _Worker


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def acoustic_proxies(path: str) -> dict[str, float]:
    """Measured audio evidence; proxies are inferred, not semantic labels."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    frequency, power = welch(audio, rate, nperseg=min(2048, len(audio)))
    total = max(float(power.sum()), 1e-12)
    return {
        "spectral_centroid_hz": round(float((frequency * power).sum()) / total, 6),
        "rms": round(float(np.sqrt(np.mean(audio * audio))), 8),
        "high_frequency_ratio_4khz": round(float(power[frequency >= 4000].sum()) / total, 8),
    }


def sources() -> list[dict]:
    rows = []
    for row in read("data/manifests/manual_verified_scores.jsonl"):
        rows.append({"id": row["id"], "source_type": "real_gyu", "language": "ko", "audio_path": row["audio_path"], "identity_target": "real_gyu", "style": "neutral", "trust_weight": 1.0, "source_manifest": "manual_verified_scores.jsonl"})
    for row in read("data/manifests/pseudo_singing_v06_accepted.jsonl"):
        rows.append({"id": row["id"], "source_type": "accepted_pseudo", "language": row["language"], "audio_path": row["output_path"], "identity_target": "gyu_reference_transfer", "style": "neutral", "trust_weight": float(row["trust_weight"]), "source_manifest": "pseudo_singing_v06_accepted.jsonl"})
    teacher_rows = read("data/manifests/teacher_weighted_v06.jsonl") + read("data/manifests/teacher_style_supplement_weighted.jsonl")
    for style in ("neutral", "soft", "breathy", "energetic", "dark", "bright"):
        seen = set()
        candidates = sorted((row for row in teacher_rows if row.get("style", "neutral") == style and Path(row["output_path"]).exists()), key=lambda row: (-float(row.get("trust_weight", 0)), row["id"], row["teacher"]))
        for row in candidates:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            rows.append({"id": f"teacher_{style}_{len(seen):02d}_{row['id']}", "source_id": row["id"], "source_type": "teacher_style", "language": row["language"], "audio_path": row["output_path"], "identity_target": "shared_gyu_identity_space", "style": style, "trust_weight": float(row["trust_weight"]), "teacher": row["teacher"], "source_manifest": "teacher_weighted_v06.jsonl" if style != "dark" else "teacher_style_supplement_weighted.jsonl"})
            if len(seen) == 8:
                break
        if len(seen) < 8:
            raise RuntimeError(f"need 8 existing teacher outputs for style={style}, found {len(seen)}")
    assert len({row["id"] for row in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_type"], row["style"])].append(row)
    for group in groups.values():
        for index, row in enumerate(sorted(group, key=lambda value: value["id"])):
            row["split"] = "test" if index == 0 else ("validation" if index == 1 else "train")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="data/manifests/soulx_real_latents_v07.jsonl")
    args = parser.parse_args()
    rows = sources()[:args.limit] if args.limit else sources()
    root = Path("data/cache/soulx_latents_v07"); root.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("GYU_SINGER_CACHE", "data/cache")).resolve()
    soulx = cache / "soulx-singer"
    # Keep venv symlink path intact; resolving it drops pyvenv.cfg discovery.
    command = [str(Path.cwd() / ".venv-soulx/bin/python"), "scripts/probe_soulx_score.py", "--worker", "--reference", "data/processed/master/216.wav", "--model", str(soulx / "pretrained_models/SoulX-Singer/model-svc.pt"), "--config", str(soulx / "soulxsinger/config/soulxsinger.yaml"), "--rmvpe", str(soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt")]
    worker = _Worker(command, Path.cwd(), os.environ | {"GYU_SINGER_CACHE": str(cache)})
    manifest = []
    try:
        for index, row in enumerate(rows, 1):
            latent = root / f"{row['id']}.pt"; rendered = root / "renders" / f"{row['id']}.wav"
            if not latent.exists():
                worker.request({"source": row["audio_path"], "output": str(rendered), "latent_output": str(latent)})
            tensor = torch.load(latent, map_location="cpu", weights_only=True)
            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != 512 or not torch.isfinite(tensor).all() or float(tensor.std()) <= 1e-5:
                raise RuntimeError(f"invalid SoulX latent {latent}: shape={tuple(tensor.shape)} std={float(tensor.std())}")
            manifest.append(row | {"audio_acoustic_proxies_inferred": acoustic_proxies(row["audio_path"]), "latent_path": str(latent), "latent_module": "SoulXSingerSVC.infer_segment.gt_decoder_inp", "latent_shape": list(tensor.shape), "latent_mean": round(float(tensor.mean()), 6), "latent_std": round(float(tensor.std()), 6), "soulx_revision": "81aeb3ae772c70093c3de74dc23c92d983801ae4"})
            print(f"{index}/{len(rows)} {row['id']}", flush=True)
    finally:
        worker.close()
    Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    style_evidence = {}
    for style in ("neutral", "soft", "breathy", "energetic", "dark", "bright"):
        evidence = [row["audio_acoustic_proxies_inferred"] for row in manifest if row["source_type"] == "teacher_style" and row["style"] == style]
        style_evidence[style] = {key: round(float(np.median([row[key] for row in evidence])), 6) for key in evidence[0]}
    report = {"rows": len(manifest), "source_types": dict(Counter(row["source_type"] for row in manifest)), "languages": dict(Counter(row["language"] for row in manifest)), "styles": dict(Counter(row["style"] for row in manifest)), "tensor": "actual pre-adapter SoulXSingerSVC.infer_segment.gt_decoder_inp", "all_nonzero": all(row["latent_std"] > 1e-5 for row in manifest), "teacher_audio_proxy_medians_inferred": style_evidence}
    Path("artifacts/reports/soulx_real_latents_v07.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/real_latent_dataset.md").write_text("# Real SoulX latent dataset\n\n" + json.dumps(report, indent=2) + "\n\nEvery row preserves audio, source type, language, style, identity target, trust, exact latent path, tensor statistics, and SoulX revision. Teacher/pseudo rows remain lower trust than real GYU singing. Acoustic measurements are explicitly inferred proxies, not proof of perceived semantics.\n")
    print(report)


if __name__ == "__main__":
    main()
