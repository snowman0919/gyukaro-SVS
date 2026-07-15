#!/usr/bin/env python3
"""Build clean/current-SoulX pairs without synthetic corruption."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gyu_singer.data import acoustic_reference_features
from gyu_singer.inference.soulx import _Worker
from gyu_singer.model import MultiTeacherIdentityEncoder


SOULX_REVISION = "81aeb3ae772c70093c3de74dc23c92d983801ae4"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def public_pairs(rows: list[dict], dataset: str, per_speaker: int = 3) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["dataset"] == dataset and row["accepted"]:
            grouped[row["speaker"]].append(row)
    output = []
    for speaker, members in sorted(grouped.items()):
        members.sort(key=lambda row: row["id"])
        if dataset == "vocalset":
            reference = next((row for technique in ("excerpts/straight", "long_tones/straight", "long_tones/forte") for row in members if row.get("technique") == technique), members[0])
            priorities = ("scales/fast_forte", "arpeggios/straight", "scales/vibrato", "long_tones/straight", "scales/breathy", "arpeggios/fast_piano")
            sources = [row for technique in priorities for row in members if row is not reference and row.get("technique") == technique][:per_speaker]
        else:
            reference, sources = members[0], members[1:1 + per_speaker]
        for source in sources:
            output.append({
                "id": f"pair_{source['id']}", "dataset": dataset, "domain": source["domain"], "language": source["language"],
                "speaker": speaker, "source_id": source["id"], "clean_target": source["audio"], "reference": reference["audio"],
                "split": source["split"], "trust_weight": source["trust_weight"], "identity_adapter": False, "style_adapter": False,
                "technique": source.get("technique"),
            })
    return output


def gyu_pairs(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    output = []
    for index, row in enumerate(rows):
        output.append({
            "id": f"pair_{row['id']}", "dataset": "real_gyu", "domain": "singing", "language": "ko", "speaker": "GYU",
            "source_id": row["id"], "clean_target": row["audio_path"], "reference": "data/processed/master/216.wav",
            "split": "test" if index % 5 == 0 else "validation" if index % 5 == 1 else "train", "trust_weight": 1.0,
            "identity_adapter": True, "style_adapter": True, "score_ground_truth": "independently_verified",
        })
    return output


def identity_files(root: Path) -> tuple[Path, Path]:
    checkpoint = torch.load("checkpoints/gyu_identity_space_v0.6.pt", map_location="cpu", weights_only=False)
    encoder = MultiTeacherIdentityEncoder(**checkpoint["model_config"]).eval(); encoder.load_state_dict(checkpoint["model"])
    with torch.inference_mode():
        identity = encoder.student(acoustic_reference_features("data/processed/master/216.wav")[None])[0].numpy()
    style = np.zeros(64, dtype="float32"); style[0] = 1
    root.mkdir(parents=True, exist_ok=True); identity_path, style_path = root / "gyu_identity.npy", root / "neutral_style.npy"
    np.save(identity_path, identity); np.save(style_path, style)
    return identity_path, style_path


def decoder_options(row: dict) -> dict:
    if row["dataset"] == "vocalset" and row.get("technique", "").endswith("fast_forte"):
        return {"n_steps": 64, "cfg": 2.0, "seed": 21}
    if row["dataset"] == "vocalset" and row.get("technique", "").startswith("arpeggios"):
        return {"n_steps": 32, "cfg": 2.0, "seed": 21}
    return {"n_steps": 32, "cfg": 1.5, "seed": 21}


def normalize_output(path: Path) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True); mono = audio.mean(1)
    if rate != 48000:
        mono = resample_poly(mono, 48000, rate).astype("float32")
    peak = float(np.max(np.abs(mono))); mono *= min(1.0, .97 / max(peak, 1e-8))
    sf.write(path, mono, 48000, subtype="PCM_24")
    return {"sample_rate": 48000, "duration_sec": round(len(mono) / 48000, 4), "peak": round(float(np.max(np.abs(mono))), 6)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/external/manifests/pipeline_degradation_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/external/work/degradation_pairs"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pair-plan", type=Path)
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/pipeline_degradation_pairs.json"))
    args = parser.parse_args()
    if args.pair_plan:
        rows = read_jsonl(args.pair_plan)
    else:
        clean = read_jsonl(Path("data/external/manifests/acoustic_clean.jsonl"))
        rows = public_pairs(clean, "libritts_r") + public_pairs(clean, "vocalset") + gyu_pairs(Path("data/manifests/manual_verified_scores.jsonl"))
    if args.limit:
        rows = rows[:args.limit]
    identity = style = None
    if any(row["identity_adapter"] for row in rows):
        identity, style = identity_files(args.output / "conditioning")
    root = Path.cwd(); cache = Path(os.environ.get("GYU_SINGER_CACHE", "data/cache")).resolve(); soulx = cache / "soulx-singer"
    command = [str(root / ".venv-soulx/bin/python"), "scripts/probe_soulx_score.py", "--worker", "--precision", "fp32",
               "--reference", str((root / "data/processed/master/216.wav").resolve()), "--model", str(soulx / "pretrained_models/SoulX-Singer/model-svc.pt"),
               "--config", str(soulx / "soulxsinger/config/soulxsinger.yaml"), "--rmvpe", str(soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
               "--latent-adapter", str((root / "checkpoints/gyu_real_latent_adapters_v0.7.pt").resolve())]
    worker = _Worker(command, root, os.environ | {"GYU_SINGER_CACHE": str(cache)})
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    output = []
    try:
        for index, row in enumerate(rows, 1):
            target = args.output / row["dataset"] / f"{row['id']}.wav"; target.parent.mkdir(parents=True, exist_ok=True)
            options = decoder_options(row)
            if not target.exists():
                request = {"source": str((root / row["clean_target"]).resolve()), "reference": str((root / row["reference"]).resolve()), "output": str(target.resolve())} | options
                if row["identity_adapter"]:
                    assert identity is not None and style is not None
                    request |= {"identity_npy": str(identity.resolve()), "style_npy": str(style.resolve())}
                worker.request(request); normalize_output(target)
            info = sf.info(target)
            output.append(row | {
                "degraded_input": str(target), "degradation": "actual SoulX reconstruction; no random-noise synthesis",
                "pipeline_git_commit": commit, "soulx_revision": SOULX_REVISION, "precision": "fp32", "decoder": options,
                "input_sample_rate": info.samplerate, "target_sample_rate": sf.info(row["clean_target"]).samplerate,
                "duration_sec": round(info.duration, 4), "pipeline_f0": "RMVPE extracted from clean source and applied as SoulX conditioning",
            })
            print(f"{index}/{len(rows)} {row['id']}", flush=True)
    finally:
        worker.close()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    counts = {domain: sum(row["domain"] == domain for row in output) for domain in ("speech", "singing")}
    report = {"rows": len(output), "domains": counts, "datasets": {name: sum(row["dataset"] == name for row in output) for name in ("libritts_r", "vocalset", "real_gyu")},
              "splits": {split: sum(row["split"] == split for row in output) for split in ("train", "validation", "test")}, "pipeline_git_commit": commit,
              "soulx_revision": SOULX_REVISION, "real_pipeline_degradation_only": True, "random_noise_used": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
