#!/usr/bin/env python3
"""Build the rapid DiffSinger probe from an independent local UST score."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TICKS_PER_BEAT = 480
TIMESTEP = 0.02
PATTERN = ("い", "き", "が", "つ", "ま", "る")
PHONES = {
    "い": ("i_ja",),
    "き": ("k_ja", "i_ja"),
    "が": ("ɡ_ja", "a_ja"),
    "つ": ("ts_ja", "ɨ_ja"),
    "ま": ("m_ja", "a_ja"),
    "る": ("ɾ_ja", "ɯ_ja"),
}
PJS_PHONES = {
    "い": ("ja_i",),
    "き": ("ja_k", "ja_i"),
    "が": ("ja_g", "ja_a"),
    "つ": ("ja_ts", "ja_u"),
    "ま": ("ja_m", "ja_a"),
    "る": ("ja_r", "ja_u"),
}
ONSET_SECONDS = {"き": 0.04, "が": 0.04, "つ": 0.055, "ま": 0.04, "る": 0.02}
UNVOICED = {"k_ja", "ts_ja"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_ust(path: Path) -> tuple[dict, list[dict]]:
    text = path.read_text(encoding="cp932")
    settings: dict[str, str] = {}
    notes: list[dict] = []
    position = 0
    for block in re.split(r"(?=\[#\w+\])", text):
        header = re.match(r"\[#([^]]+)\]", block)
        if not header:
            continue
        values = dict(line.split("=", 1) for line in block.splitlines()[1:] if "=" in line)
        if header.group(1) == "SETTING":
            settings = values
        elif header.group(1).isdigit():
            length = int(float(values["Length"]))
            notes.append({
                "position": position,
                "length": length,
                "lyric": values["Lyric"],
                "tone": int(float(values["NoteNum"])),
            })
            position += length
    return settings, notes


def select_phrase(notes: list[dict], *, repeats: int = 4) -> list[dict]:
    expected = PATTERN * repeats
    lyrics = tuple(note["lyric"] for note in notes)
    for index in range(len(notes) - len(expected) + 1):
        if lyrics[index:index + len(expected)] == expected:
            return notes[index:index + len(expected)]
    raise ValueError(f"independent UST does not contain {repeats} adjacent target repetitions")


def build_ds_row(notes: list[dict], tempo: float, *, onset_scale: float = 1.0,
                 phone_scheme: str = "gtsinger", semitone_shift: int = 0) -> tuple[dict, dict]:
    tick_seconds = 60 / (tempo * TICKS_PER_BEAT)
    start_tick = notes[0]["position"]
    phones: list[str] = []
    durations: list[float] = []
    phone_intervals: list[tuple[float, float, str, int]] = []
    elapsed = 0.0
    phone_map = PHONES if phone_scheme == "gtsinger" else PJS_PHONES
    unvoiced = UNVOICED if phone_scheme == "gtsinger" else {"ja_k", "ja_ts"}
    for note in notes:
        note_duration = note["length"] * tick_seconds
        symbols = phone_map[note["lyric"]]
        if len(symbols) == 1:
            splits = (note_duration,)
        else:
            onset = min(ONSET_SECONDS[note["lyric"]] * onset_scale, note_duration * 0.4)
            splits = (onset, note_duration - onset)
        for symbol, duration in zip(symbols, splits):
            phones.append(symbol)
            durations.append(duration)
            phone_intervals.append((elapsed, elapsed + duration, symbol, note["tone"]))
            elapsed += duration

    frames = round(elapsed / TIMESTEP)
    f0 = []
    for frame in range(frames):
        center = min((frame + 0.5) * TIMESTEP, elapsed - 1e-9)
        _, _, symbol, tone = next(
            interval for interval in phone_intervals if interval[0] <= center < interval[1]
        )
        hz = 440 * 2 ** ((tone + semitone_shift - 69) / 12)
        f0.append(0.0 if symbol in unvoiced else hz)

    row = {
        "offset": 0,
        "text": "local independent rapid score",
        "ph_seq": " ".join(phones),
        "ph_dur": " ".join(f"{value:.7f}" for value in durations),
        "f0_seq": " ".join(f"{value:.3f}" for value in f0),
        "f0_timestep": str(TIMESTEP),
        "spk_mix": {"gts_ja_soprano" if phone_scheme == "gtsinger" else "pjs": 1.0},
    }
    metadata = {
        "start_tick": start_tick,
        "start_seconds": round(start_tick * tick_seconds, 4),
        "duration_seconds": round(elapsed, 4),
        "notes": len(notes),
        "phonemes": len(phones),
        "frames": frames,
        "source_ust_midi_min": min(note["tone"] for note in notes),
        "source_ust_midi_max": max(note["tone"] for note in notes),
        "render_midi_min": min(note["tone"] + semitone_shift for note in notes),
        "render_midi_max": max(note["tone"] + semitone_shift for note in notes),
        "semitone_shift": semitone_shift,
        "nominal_f0_hz": round(
            440 * 2 ** ((notes[0]["tone"] + semitone_shift - 69) / 12), 2
        ),
        "voiced_ratio": round(float(np.mean(np.asarray(f0) > 1)), 4),
        "onset_scale": onset_scale,
        "phone_scheme": phone_scheme,
    }
    return row, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ust", type=Path, required=True,
                        help="Local main-vocal UST; never copied into the repository/package")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/external/work/gtsinger/independent_rapid.ds")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "artifacts/reports/diffsinger_independent_rapid_score.json")
    parser.add_argument("--onset-scale", type=float, default=1.0)
    parser.add_argument("--semitone-shift", type=int, default=0,
                        help="Audited correction applied to UST note numbers")
    parser.add_argument("--phone-scheme", choices=("gtsinger", "pjs"), default="gtsinger")
    args = parser.parse_args()
    args.ust = args.ust.resolve()
    args.output = args.output.resolve()
    args.report = args.report.resolve()

    settings, notes = parse_ust(args.ust)
    tempo = float(settings["Tempo"])
    phrase = select_phrase(notes)
    if not 0 < args.onset_scale <= 2:
        raise ValueError("--onset-scale must be in (0, 2]")
    row, metadata = build_ds_row(
        phrase, tempo, onset_scale=args.onset_scale, phone_scheme=args.phone_scheme,
        semitone_shift=args.semitone_shift,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([row], ensure_ascii=False, indent=2) + "\n")
    report = {
        "status": "independent_score_ready_audio_alignment_pending",
        "source": {
            "kind": "third_party_ust_local_evaluation_only",
            "sha256": sha256(args.ust),
            "public_page": "https://bowlroll.net/file/255852",
            "redistribution": "prohibited_by_source_readme",
        },
        "score_independent_from_target_f0": True,
        "score_method": "independent community transcription",
        "phoneme_timing": "inferred onset split within UST note timing",
        "tempo_bpm": tempo,
        "output": str(args.output.relative_to(ROOT)),
        "metadata": metadata,
        "release_allowed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
