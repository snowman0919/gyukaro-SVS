#!/usr/bin/env python3
"""Freeze RC8 content/F0 inputs for the truncated identity diagnostic."""
from __future__ import annotations

import shutil
import gc
import hashlib
import json
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np


class CaptureComplete(Exception):
    pass


class CaptureWorker:
    def __init__(self, output: Path):
        self.output = Path(output)
        self.options = {}

    def request(self, body):
        self.output.mkdir(parents=True, exist_ok=True)
        for key, name in {
            "source": "source.wav", "f0_npy": "f0.npy", "identity_npy": "identity.npy",
            "style_npy": "style.npy", "content_warp_npy": "content_warp.npy",
        }.items():
            if body.get(key):
                shutil.copy2(body[key], self.output / name)
        self.options = {
            key: body[key]
            for key in ("content_warp_strength", "n_steps", "cfg", "seed")
            if key in body
        }
        raise CaptureComplete

    def close(self):
        pass


def fixed_split():
    return {
        "train": ["korean", "english", "japanese"],
        "validation": ["quality_ko", "quality_en", "quality_ja"],
        "heldout": [
            "heldout_ko", "heldout_en", "review_sustain_ko",
            "review_large_interval_ko", "review_phrase_boundary_ko",
        ],
        "protected": ["review_rapid_ko"],
        "excluded": ["heldout_ja"],
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compact(text):
    return "".join(character.lower() for character in text if character.isalnum())


def _repeated_span(expected, observed):
    for size in range(len(expected), 1, -1):
        for start in range(len(expected) - size + 1):
            span = expected[start:start + size]
            if observed.count(span) > expected.count(span):
                return span
    return None


def main():
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    root = Path.cwd()
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], text=True).strip()).resolve()
    runtime_root = common.parent
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from gyu_singer.inference.rc8 import GyuSingerRC8Renderer
    from gyu_singer.score import normalize_score

    output = root / "artifacts/reports/truncated_identity_training"
    corpus = output / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    split = fixed_split()
    rows = []
    renderer = GyuSingerRC8Renderer(runtime_root / "data/processed/master/216.wav", root=runtime_root)
    renderer.soulx.close()
    try:
        for split_name, names in split.items():
            for name in names:
                case_dir = corpus / name
                shutil.rmtree(case_dir, ignore_errors=True)
                capture = CaptureWorker(case_dir)
                renderer.soulx = capture
                score_path = root / "examples" / f"{name}.json"
                score = normalize_score(json.loads(score_path.read_text()))
                try:
                    renderer.render(score)
                except CaptureComplete:
                    pass
                else:
                    raise RuntimeError(f"SoulX input capture did not stop {name}")
                source = case_dir / "source.wav"
                f0 = case_dir / "f0.npy"
                identity = case_dir / "identity.npy"
                if not all(path.exists() for path in (source, f0, identity)):
                    raise RuntimeError(f"incomplete SoulX inputs for {name}")
                expected = "".join(note["lyric"] for note in score["notes"])
                row = {
                    "id": name,
                    "split": split_name,
                    "language": score["language"],
                    "score_path": str(score_path.relative_to(root)),
                    "score_sha256": _sha(score_path),
                    "expected_lyrics": expected,
                    "source_path": str(source.relative_to(root)),
                    "source_sha256": _sha(source),
                    "f0_path": str(f0.relative_to(root)),
                    "f0_sha256": _sha(f0),
                    "identity_path": str(identity.relative_to(root)),
                    "identity_sha256": _sha(identity),
                    "content_warp_path": str((case_dir / "content_warp.npy").relative_to(root)) if (case_dir / "content_warp.npy").exists() else None,
                    "content_warp_strength": capture.options.get("content_warp_strength", 0.0),
                    "captured_decoder_steps": capture.options.get("n_steps", 16),
                    "diagnostic_decoder_steps": 64,
                    "cfg": capture.options.get("cfg", 2.5),
                    "fixed_seeds": [7, 21, 42],
                    "identity_training_eligible": split_name == "train",
                }
                info = sf.info(source)
                row["source_duration_seconds"] = round(info.frames / info.samplerate, 4)
                row["f0_frames"] = int(len(np.load(f0)))
                rows.append(row)
                print(f"captured {name} {split_name}", flush=True)
    finally:
        renderer.close()
    del renderer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(runtime_root / "data/cache/whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        runtime_root / "data/cache/whisper-large-v3-turbo", dtype=torch.float16,
    ).cuda().eval()
    for index, row in enumerate(rows, 1):
        audio, rate = sf.read(root / row["source_path"], dtype="float32", always_2d=True)
        mono = audio.mean(1)
        audio16 = resample_poly(mono, 16_000, rate).astype("float32") if rate != 16_000 else mono
        values = processor(audio16, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(
                values.input_features.cuda().half(), language=row["language"],
                task="transcribe", max_new_tokens=96,
            )
        transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        expected, observed = _compact(row["expected_lyrics"]), _compact(transcript)
        similarity = SequenceMatcher(None, expected, observed).ratio()
        repeated = _repeated_span(expected, observed)
        row |= {
            "source_whisper_transcript": transcript,
            "source_lyric_similarity": round(similarity, 4),
            "source_repeated_expected_span": repeated,
            "source_gate_pass": bool(similarity >= 0.80 and repeated is None),
        }
        print(f"whisper {index}/{len(rows)} {row['id']} {transcript}", flush=True)
    del model
    torch.cuda.empty_cache()

    manifest = root / "data/manifests/truncated_identity_diagnostic.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    core = [row for row in rows if row["split"] in {"train", "validation"}]
    heldout = [row for row in rows if row["split"] in {"heldout", "protected"}]
    report = {
        "status": "source_gate_pass" if all(row["source_gate_pass"] for row in core) else "diagnostic_reject",
        "split_counts": {name: sum(row["split"] == name for row in rows) for name in split},
        "fixed_seeds": [7, 21, 42],
        "source_gate": {
            "minimum_train_validation_similarity": 0.80,
            "train_validation_pass": all(row["source_gate_pass"] for row in core),
            "heldout_all_pass": all(row["source_gate_pass"] for row in heldout),
            "excluded_heldout_ja_retained": any(row["id"] == "heldout_ja" and row["split"] == "excluded" for row in rows),
        },
        "rows": rows,
        "runtime_integration": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "split_counts": report["split_counts"], "source_gate": report["source_gate"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "source_gate_pass" else 1)


if __name__ == "__main__":
    main()
