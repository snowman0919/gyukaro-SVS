"""Deterministic phrase-level phoneme/note frame alignment."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from gyu_singer.frontend import FrontendOutput, phonemize


@dataclass
class PhraseFrames:
    phoneme_ids: torch.Tensor
    language_ids: torch.Tensor
    features: torch.Tensor
    midi: torch.Tensor
    f0_hz: torch.Tensor
    voiced: torch.Tensor
    residual: torch.Tensor
    note_index: torch.Tensor
    note_onset: torch.Tensor
    note_duration: torch.Tensor
    boundary: torch.Tensor
    phoneme_note_mapping: torch.Tensor
    phoneme_durations: list[dict]
    note_sequence: list[dict]
    boundary_types: list[str]
    voicing_classes: list[str]
    timeline: list[dict]


_UNVOICED = {
    "en_ch", "en_f", "en_hh", "en_k", "en_p", "en_s", "en_sh", "en_t", "en_th",
    "ja_ch", "ja_f", "ja_h", "ja_hy", "ja_k", "ja_ky", "ja_p", "ja_s", "ja_sh", "ja_t", "ja_ts",
}
_VOICED_CONSONANTS = {
    "en_b", "en_d", "en_dh", "en_g", "en_jh", "en_l", "en_m", "en_n", "en_ng", "en_r", "en_v", "en_w", "en_y", "en_z", "en_zh",
    "ja_b", "ja_d", "ja_g", "ja_gy", "ja_j", "ja_m", "ja_my", "ja_n", "ja_ny", "ja_r", "ja_ry", "ja_w", "ja_y", "ja_z",
}
_KO_VOICED_ONSETS = {2, 5, 6, 11}
# Hangul jongseong indices whose sustained realization is sonorant. Obstruent
# codas (including ㅂ/ㅅ/ㅍ) are unvoiced even when written in a final cluster.
_KO_VOICED_CODAS = {4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21}


def phoneme_voicing(symbol: str, features: list[float]) -> str:
    """Classify a frontend phone without pretending inferred timing is observed."""
    if symbol in {"sil", "sp", "pau", "-"}:
        return "silence"
    if features[1] or symbol in {"ja_long"}:
        return "vowel"
    if symbol.startswith("ko_onset_"):
        return "voiced_consonant" if int(symbol.rsplit("_", 1)[1]) in _KO_VOICED_ONSETS else "unvoiced_consonant"
    if symbol.startswith("ko_coda_"):
        return "voiced_consonant" if int(symbol.rsplit("_", 1)[1]) in _KO_VOICED_CODAS else "unvoiced_consonant"
    if symbol in _UNVOICED:
        return "unvoiced_consonant"
    if symbol in _VOICED_CONSONANTS or features[7]:
        return "voiced_consonant"
    if symbol.startswith("en_"):
        return "vowel" if symbol[3:] in {"aa", "ae", "ah", "ao", "aw", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"} else "unvoiced_consonant"
    if symbol.startswith("ja_"):
        return "vowel" if symbol[3:] in {"a", "i", "u", "e", "o"} else "unvoiced_consonant"
    return "voiced_consonant"


def _unit_ranges(units: FrontendOutput, start: int, end: int, frame_hz: float) -> list[tuple[int, int]]:
    """Give consonants short physical windows and the vowel the sustained remainder."""
    count = len(units.symbols)
    if count == 1:
        return [(start, end)]
    classes = [phoneme_voicing(symbol, feature) for symbol, feature in zip(units.symbols, units.features)]
    consonant = max(1, round(.06 * frame_hz))
    weights = np.array([consonant if kind.endswith("consonant") else max(consonant, end - start) for kind in classes], dtype=np.float64)
    vowel_count = sum(kind == "vowel" for kind in classes)
    if vowel_count:
        fixed = consonant * (count - vowel_count)
        vowel_frames = max(vowel_count, end - start - fixed)
        weights = np.array([vowel_frames / vowel_count if kind == "vowel" else consonant for kind in classes])
    boundaries = np.rint(start + np.cumsum(np.r_[0.0, weights / weights.sum() * (end - start)])).astype(int)
    boundaries[0], boundaries[-1] = start, end
    return [(int(boundaries[i]), max(int(boundaries[i]) + 1, int(boundaries[i + 1]))) for i in range(count)]


def build_phrase_frames(frontend: FrontendOutput, notes: list[dict], pitch_curve: list[dict] | None = None, frame_hz: float = 12.5, phoneme_alignment: list[dict] | None = None, legacy_all_voiced: bool = False) -> PhraseFrames:
    if not notes:
        raise ValueError("notes must not be empty")
    ordered = sorted(notes, key=lambda note: float(note.get("start", note.get("start_sec", 0))))
    total = max(float(note.get("start", note.get("start_sec", 0))) + float(note.get("duration", note.get("duration_sec", 0))) for note in ordered)
    frames = max(1, math.ceil(total * frame_hz))
    phoneme_ids = torch.zeros(frames, dtype=torch.long)
    language_ids = torch.zeros(frames, dtype=torch.long)
    features = torch.zeros(frames, len(frontend.features[0]))
    midi = torch.zeros(frames)
    note_index = torch.zeros(frames, dtype=torch.long)
    note_onset = torch.zeros(frames)
    note_duration = torch.zeros(frames)
    boundary = torch.zeros(frames)
    voiced = torch.zeros(frames)
    voicing_classes = ["silence"] * frames
    phoneme_durations, note_sequence, boundary_types = [], [], []
    indexed_alignment = phoneme_alignment and all("phoneme_index" in phone for phone in phoneme_alignment)
    if indexed_alignment:
        for phone in phoneme_alignment:
            index = int(phone["phoneme_index"])
            if index >= len(frontend.phoneme_ids): continue
            start = max(0, int(round(float(phone["start"]) * frame_hz)))
            end = min(frames, max(start + 1, int(round((float(phone["start"]) + float(phone["duration"])) * frame_hz))))
            phoneme_ids[start:end] = frontend.phoneme_ids[index]
            language_ids[start:end] = frontend.language_ids[index]
            features[start:end] = torch.tensor(frontend.features[index], dtype=torch.float32)
            kind = phoneme_voicing(frontend.symbols[index], frontend.features[index])
            voiced[start:end] = float(kind in {"vowel", "voiced_consonant"})
            voicing_classes[start:end] = [kind] * (end - start)
            phoneme_durations.append({"symbol": frontend.symbols[index], "note_index": -1, "start_frame": start, "duration_frames": end - start, "boundary_type": "ctc_forced"})
    for index, note in enumerate(ordered):
        start = int(round(float(note.get("start", note.get("start_sec", 0))) * frame_hz))
        end = max(start + 1, int(round((float(note.get("start", note.get("start_sec", 0))) + float(note.get("duration", note.get("duration_sec", 0)))) * frame_hz)))
        end = min(frames, end)
        midi[start:end] = float(note["pitch"])
        note_index[start:end] = index
        note_onset[start:end] = float(note.get("start", note.get("start_sec", 0))) / max(total, 1e-3)
        note_duration[start:end] = float(note.get("duration", note.get("duration_sec", 0))) / max(total, 1e-3)
        # A slur carries articulation across note onset instead of creating a hard boundary.
        boundary[start] = 0.0 if note.get("slur", False) else 1.0
        lyric = str(note.get("lyric", ""))
        if not indexed_alignment:
            # Score lyric owns its note frames. OpenUtau phone spans constrain the
            # window; frontend phones split that window with singing-aware weights.
            units = phonemize(frontend.language, lyric) if lyric.strip() else frontend
            phone_window = [] if not phoneme_alignment else [phone for phone in phoneme_alignment if start / frame_hz <= float(phone["start"]) + float(phone["duration"]) / 2 < end / frame_hz]
            owned_start = max(start, int(round(min((float(phone["start"]) for phone in phone_window), default=start / frame_hz) * frame_hz)))
            owned_end = min(end, int(round(max((float(phone["start"]) + float(phone["duration"]) for phone in phone_window), default=end / frame_hz) * frame_hz)))
            owned_end = max(owned_start + 1, owned_end)
            for phoneme_index, (phone_start, phone_end) in enumerate(_unit_ranges(units, owned_start, owned_end, frame_hz)):
                phone_end = min(frames, phone_end)
                phoneme_ids[phone_start:phone_end] = units.phoneme_ids[phoneme_index]
                language_ids[phone_start:phone_end] = units.language_ids[phoneme_index]
                features[phone_start:phone_end] = torch.tensor(units.features[phoneme_index], dtype=torch.float32)
                kind = phoneme_voicing(units.symbols[phoneme_index], units.features[phoneme_index])
                voiced[phone_start:phone_end] = float(kind in {"vowel", "voiced_consonant"})
                voicing_classes[phone_start:phone_end] = [kind] * (phone_end - phone_start)
                phoneme_durations.append({"symbol": units.symbols[phoneme_index], "note_index": index, "start_frame": phone_start, "duration_frames": phone_end - phone_start, "boundary_type": "openutau_timed_inferred_split" if phone_window else "score_timed_inferred_split", "voicing": kind, "inferred": True})
        boundary_type = "slur" if note.get("slur", False) else "hard"
        boundary_types.append(boundary_type)
        note_sequence.append({"id": str(note.get("id", f"n{index + 1}")), "index": index, "pitch": float(note["pitch"]), "start_frame": start, "end_frame": end, "lyric": lyric, "boundary_type": boundary_type})
    for duration in phoneme_durations:
        if duration["note_index"] < 0:
            duration["note_index"] = int(note_index[min(frames - 1, duration["start_frame"] + duration["duration_frames"] // 2)])
    # Silence gaps inherit nearest preceding content but remain unvoiced.
    for index in range(1, frames):
        if phoneme_ids[index] == 0:
            phoneme_ids[index] = phoneme_ids[index - 1]
            language_ids[index] = language_ids[index - 1]
            features[index] = features[index - 1]
    residual = torch.zeros(frames)
    if pitch_curve:
        times = torch.arange(frames) / frame_hz
        points_t = torch.tensor([point["time"] for point in pitch_curve])
        points_v = torch.tensor([point["value"] for point in pitch_curve])
        residual = torch.from_numpy(np.interp(times.numpy(), points_t.numpy(), points_v.numpy()).astype("float32"))
    if legacy_all_voiced:
        voiced[:] = 1
        voicing_classes = ["legacy_forced_voiced"] * frames
    f0_hz = 440.0 * torch.pow(torch.tensor(2.0), (midi + residual - 69.0) / 12.0) * voiced
    timeline = [{"frame": index, "time": index / frame_hz, "phoneme": next((row["symbol"] for row in phoneme_durations if row["start_frame"] <= index < row["start_frame"] + row["duration_frames"]), "sil"), "note_index": int(note_index[index]), "voicing": voicing_classes[index], "voiced": bool(voiced[index]), "f0_hz": float(f0_hz[index]), "user_pitch_semitones": float(residual[index])} for index in range(frames)]
    return PhraseFrames(
        phoneme_ids=phoneme_ids,
        language_ids=language_ids,
        features=features,
        midi=midi,
        f0_hz=f0_hz,
        voiced=voiced,
        residual=residual,
        note_index=note_index,
        note_onset=note_onset,
        note_duration=note_duration,
        boundary=boundary,
        phoneme_note_mapping=note_index,
        phoneme_durations=phoneme_durations,
        note_sequence=note_sequence,
        boundary_types=boundary_types,
        voicing_classes=voicing_classes,
        timeline=timeline,
    )
