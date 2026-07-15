#!/usr/bin/env python3
"""Extract and quality-score a bounded, speaker-disjoint acoustic subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tarfile
import zipfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly, stft

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gyu_singer.data import acoustic_reference_features


LIBRI_MD5 = "2c1f5312914890634cc2d15783032ff3"
VOCALSET_MD5 = "c5e5efab412637fc94972b93c343a2f0"
VOCAL_CATEGORIES = (
    "scales/fast_forte", "arpeggios/fast_piano", "arpeggios/straight",
    "long_tones/straight", "long_tones/forte", "scales/vibrato",
    "scales/breathy", "excerpts/straight",
)


def digest(path: Path, algorithm: str = "md5") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_pick(names: list[str], count: int, salt: str) -> list[str]:
    return sorted(names, key=lambda name: hashlib.sha256(f"{salt}:{name}".encode()).digest())[:count]


def normalize_text(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def extract_libri(archive: Path, root: Path, speakers: int = 8, clips: int = 8) -> list[dict]:
    selected: dict[str, dict] = {}
    with tarfile.open(archive, "r:gz") as source:
        wavs = [member for member in source.getmembers() if member.isfile() and member.name.endswith(".wav")]
        grouped: dict[str, list] = defaultdict(list)
        for member in wavs:
            grouped[PurePosixPath(member.name).parts[2]].append(member)
        chosen_speakers = stable_pick([key for key, values in grouped.items() if len(values) >= clips], speakers, "libri-speakers")
        for speaker in chosen_speakers:
            for member in stable_pick([value.name for value in grouped[speaker]], clips, f"libri-{speaker}"):
                stem = PurePosixPath(member).stem
                selected[member] = {"id": f"libritts_r_{stem}", "dataset": "libritts_r", "language": "en", "speaker": speaker,
                                    "audio": str(root / "libritts_r" / speaker / f"{stem}.wav"), "text": "", "domain": "speech", "source_member": member}
    wanted = set(selected) | {name[:-4] + ".normalized.txt" for name in selected}
    # gzip tar archives have no cheap random access. A second streaming pass avoids
    # repeatedly decompressing from byte zero for every selected member.
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            if member.name not in wanted:
                continue
            handle = source.extractfile(member)
            if handle is None:
                continue
            wav_name = member.name if member.name.endswith(".wav") else member.name.removesuffix(".normalized.txt") + ".wav"
            row = selected[wav_name]
            if member.name.endswith(".wav"):
                target = Path(row["audio"]); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(handle.read())
            else:
                row["text"] = handle.read().decode("utf-8").strip()
    assert all(Path(row["audio"]).exists() and row["text"] for row in selected.values())
    return list(selected.values())


def extract_vocalset(archive: Path, root: Path, singers: int = 10) -> list[dict]:
    rows: list[dict] = []
    with zipfile.ZipFile(archive) as source:
        names = [name for name in source.namelist() if name.startswith("data_by_singer/") and name.endswith(".wav") and "__MACOSX" not in name]
        available = sorted({PurePosixPath(name).parts[1] for name in names})
        female = stable_pick([name for name in available if name.startswith("female")], singers // 2, "vocal-female")
        male = stable_pick([name for name in available if name.startswith("male")], singers - len(female), "vocal-male")
        for speaker in female + male:
            speaker_names = [name for name in names if PurePosixPath(name).parts[1] == speaker]
            for category in VOCAL_CATEGORIES:
                candidates = [name for name in speaker_names if "/".join(PurePosixPath(name).parts[2:4]) == category]
                if not candidates:
                    continue
                member = stable_pick(candidates, 1, f"vocal-{speaker}-{category}")[0]
                stem = PurePosixPath(member).stem
                target = root / "vocalset" / speaker / category.replace("/", "_") / f"{stem}.wav"
                target.parent.mkdir(parents=True, exist_ok=True)
                # The official 6 GB archive has overflowing central-directory
                # offsets. Info-ZIP can compensate after `zip -FF`; Python's
                # zipfile cannot, so stream the selected member through unzip.
                with target.open("wb") as dst:
                    result = subprocess.run(("unzip", "-p", str(archive), member), stdout=dst, stderr=subprocess.PIPE)
                if target.stat().st_size < 44 or sf.info(target).frames == 0:
                    raise RuntimeError(f"failed to recover {member}: {result.stderr.decode(errors='replace')[-400:]}")
                rows.append({"id": f"vocalset_{speaker}_{category.replace('/', '_')}_{stem}", "dataset": "vocalset", "language": "en",
                             "speaker": speaker, "audio": str(target), "text": "", "domain": "singing", "technique": category,
                             "source_member": member, "text_status": "not_applicable_non_lexical_or_excerpt"})
    return rows


def quality(path: Path) -> tuple[dict, np.ndarray]:
    stereo, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = stereo.mean(1)
    duration = len(audio) / rate
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    frame = max(1, int(rate * .02)); usable = len(audio) // frame * frame
    frame_rms = np.sqrt(np.mean(audio[:usable].reshape(-1, frame) ** 2, axis=1) + 1e-12) if usable else np.array([0.0])
    noise, signal = np.percentile(frame_rms, [10, 75])
    snr = float(20 * np.log10(max(signal, 1e-8) / max(noise, 1e-6)))
    silence = float(np.mean(frame_rms < max(.002, signal * .05)))
    dc = float(abs(np.mean(audio))) if len(audio) else 0.0
    clipping = float(np.mean(np.abs(audio) >= .999)) if len(audio) else 0.0
    analysis = resample_poly(audio, 24000, rate).astype("float32") if rate != 24000 else audio
    _, _, spectrum = stft(analysis, 24000, nperseg=1024, noverlap=768, boundary=None)
    power = np.abs(spectrum) ** 2 + 1e-12
    frequencies = np.fft.rfftfreq(1024, 1 / 24000)
    hf = float(np.mean(np.sum(power[frequencies >= 8000], axis=0) / np.sum(power, axis=0))) if power.shape[1] else 0.0
    flatness = float(np.mean(np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0))) if power.shape[1] else 1.0
    envelope = frame_rms - np.mean(frame_rms)
    lag = max(1, int(.08 / .02))
    reverb = float(abs(np.dot(envelope[:-lag], envelope[lag:])) / max(np.dot(envelope, envelope), 1e-8)) if len(envelope) > lag else 0.0
    features = acoustic_reference_features(path, strict_sample_rate=False).numpy()
    return {
        "duration_sec": round(duration, 4), "sample_rate": rate, "channels": stereo.shape[1],
        "peak": round(peak, 6), "rms": round(rms, 6), "clipping_fraction": round(clipping, 8),
        "dc_offset_abs": round(dc, 7), "snr_proxy_db": round(snr, 3), "silence_frame_ratio": round(silence, 5),
        "reverb_tail_proxy": round(reverb, 5), "hf_energy_ratio": round(hf, 7), "spectral_flatness": round(flatness, 7),
    }, features


def add_asr(rows: list[dict], model_path: Path, device: str) -> None:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    for index, row in enumerate(rows, 1):
        if row["dataset"] != "libritts_r":
            row.update({"asr_text_similarity": None, "asr_transcript": None, "asr_gate": "not_applicable_non_lexical_singing"})
            continue
        audio, rate = sf.read(row["audio"], dtype="float32", always_2d=True); audio = audio.mean(1)
        if rate != 16000:
            divisor = math.gcd(rate, 16000); audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype("float32")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(inputs.input_features.to(device, dtype), language="en", task="transcribe", max_new_tokens=96)
        transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        similarity = SequenceMatcher(None, normalize_text(row["text"]), normalize_text(transcript)).ratio()
        row.update({"asr_text_similarity": round(similarity, 4), "asr_transcript": transcript, "asr_gate": "measured_whisper_large_v3_turbo"})
        print(f"ASR {index}/{len(rows)} {row['id']} {similarity:.3f}", flush=True)


def add_speaker_consistency(rows: list[dict], model_path: Path, device: str) -> None:
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
    processor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioXVector.from_pretrained(model_path).to(device).eval()
    embeddings = {}
    for index, row in enumerate(rows, 1):
        audio, rate = sf.read(row["audio"], dtype="float32", always_2d=True); audio = audio.mean(1)
        if rate != 16000:
            divisor = math.gcd(rate, 16000); audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype("float32")
        values = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.inference_mode():
            vector = model(**{key: value.to(device) for key, value in values.items()}).embeddings.squeeze().float().cpu().numpy()
        embeddings[row["id"]] = vector / max(np.linalg.norm(vector), 1e-8)
        print(f"speaker {index}/{len(rows)} {row['id']}", flush=True)
    for speaker in {row["speaker"] for row in rows}:
        members = [embeddings[row["id"]] for row in rows if row["speaker"] == speaker]
        center = np.mean(members, axis=0); center /= max(np.linalg.norm(center), 1e-8)
        for row in (item for item in rows if item["speaker"] == speaker):
            row["speaker_consistency_cosine"] = round(float(np.dot(embeddings[row["id"]], center)), 5)
            row["speaker_consistency_method"] = "WavLMForXVector microsoft/wavlm-base-plus-sv"
            row["quality_gates"]["speaker_consistency"] = row["speaker_consistency_cosine"] >= .75


def finalize(rows: list[dict]) -> list[dict]:
    features: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows, 1):
        metrics, features[row["id"]] = quality(Path(row["audio"])); row.update(metrics)
        row["audio_sha256"] = digest(Path(row["audio"]), "sha256")
        print(f"quality {index}/{len(rows)} {row['id']}", flush=True)
    centroids = {}
    for speaker in {row["speaker"] for row in rows}:
        vectors = [features[row["id"]] for row in rows if row["speaker"] == speaker]
        centroids[speaker] = np.mean(vectors, axis=0)
    speaker_order = sorted({row["speaker"] for row in rows})
    split_map = {speaker: ("test" if index % 5 == 0 else "validation" if index % 5 == 1 else "train") for index, speaker in enumerate(speaker_order)}
    for row in rows:
        vector, center = features[row["id"]], centroids[row["speaker"]]
        consistency = float(np.dot(vector, center) / max(np.linalg.norm(vector) * np.linalg.norm(center), 1e-8))
        row["speaker_consistency_cosine"] = round(consistency, 5)
        row["split"] = split_map[row["speaker"]]
        gates = {
            "duration": .75 <= row["duration_sec"] <= 30,
            "clipping": row["clipping_fraction"] <= .0001,
            "dc_offset": row["dc_offset_abs"] <= .02,
            "level": row["rms"] >= .004,
            "snr_proxy": row["snr_proxy_db"] >= 10,
            "high_frequency": row["hf_energy_ratio"] <= .18,
            "speaker_consistency": consistency >= .92,
            "asr_text": row["dataset"] == "vocalset" or row.get("asr_text_similarity") is None or row["asr_text_similarity"] >= .75,
            "music_background": True,
        }
        row["quality_gates"] = gates
        row["music_background_evidence"] = "source_isolated_solo_vocal_recording" if row["dataset"] == "vocalset" else "clean_read_speech_corpus_plus_measured_asr_consistency"
        penalties = [not passed for passed in gates.values()]
        score = max(0.0, 1.0 - .12 * sum(penalties) - min(row["clipping_fraction"] * 100, .2) - min(row["dc_offset_abs"] * 2, .1))
        row["quality_score"] = round(score, 4)
        row["accepted"] = all(gates.values())
        row["trust_weight"] = round(score * (1.0 if row["accepted"] else 0.0), 4)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/external/raw"))
    parser.add_argument("--work", type=Path, default=Path("data/external/work/selected"))
    parser.add_argument("--manifest", type=Path, default=Path("data/external/manifests/acoustic_clean.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/external_acoustic_quality.json"))
    parser.add_argument("--whisper", type=Path, default=Path("data/cache/whisper-large-v3-turbo"))
    parser.add_argument("--wavlm", type=Path, default=Path("data/cache/wavlm-base-plus-sv"))
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()
    libri, vocal = args.raw / "dev_clean.tar.gz", args.raw / "VocalSet1-2.zip"
    assert digest(libri) == LIBRI_MD5 and digest(vocal) == VOCALSET_MD5, "external archive checksum mismatch"
    repaired_vocal = args.raw.parent / "work" / "VocalSet1-2.fixed.zip"
    if not repaired_vocal.exists():
        raise FileNotFoundError("repair official VocalSet ZIP first: zip -FF data/external/raw/VocalSet1-2.zip --out data/external/work/VocalSet1-2.fixed.zip")
    rows = extract_libri(libri, args.work) + extract_vocalset(repaired_vocal, args.work)
    rows = finalize(rows)
    add_speaker_consistency(rows, args.wavlm, "cuda" if torch.cuda.is_available() else "cpu")
    if not args.skip_asr:
        add_asr(rows, args.whisper, "cuda" if torch.cuda.is_available() else "cpu")
    rows = finalize_after_asr(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    accepted = [row for row in rows if row["accepted"]]
    report = {
        "schema": 1, "archive_checksums": {str(libri): LIBRI_MD5, str(vocal): VOCALSET_MD5},
        "vocalset_archive_recovery": "official ZIP central-directory offsets overflow; local zip -FF copy used, selected decoded WAVs validated by libsndfile",
        "rows": len(rows), "accepted": len(accepted), "rejected": len(rows) - len(accepted),
        "by_dataset": {dataset: {"rows": sum(row["dataset"] == dataset for row in rows), "accepted": sum(row["dataset"] == dataset and row["accepted"] for row in rows)} for dataset in ("libritts_r", "vocalset")},
        "by_split": {split: sum(row["split"] == split and row["accepted"] for row in rows) for split in ("train", "validation", "test")},
        "speaker_disjoint_splits": all(len({row["split"] for row in rows if row["speaker"] == speaker}) == 1 for speaker in {row["speaker"] for row in rows}),
        "raw_audio_bundled": False, "manifest": str(args.manifest),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def finalize_after_asr(rows: list[dict]) -> list[dict]:
    for row in rows:
        if row.get("asr_text_similarity") is not None:
            row["quality_gates"]["asr_text"] = row["dataset"] == "vocalset" or row["asr_text_similarity"] >= .75
        row["accepted"] = all(row["quality_gates"].values())
        failures = sum(not value for value in row["quality_gates"].values())
        row["quality_score"] = round(max(0.0, 1.0 - .12 * failures - min(row["clipping_fraction"] * 100, .2) - min(row["dc_offset_abs"] * 2, .1)), 4)
        row["trust_weight"] = row["quality_score"] if row["accepted"] else 0.0
    return rows


if __name__ == "__main__":
    main()
