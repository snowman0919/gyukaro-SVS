"""RC9 OpenUtau song candidate with independently controlled non-KO pitch."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gyu_singer.score import normalize_score

from .rc8 import GyuSingerRC8Renderer
from .v09 import GyuSingerV09Renderer


class GyuSingerRC9Renderer(GyuSingerRC8Renderer):
    """Keep RC8 audio fixes while making EN/JA score and PITD authoritative."""

    @staticmethod
    def _contextual_subscores(score: dict) -> list[dict]:
        """Keep long OpenUtau phrases continuous without overlong OmniVoice calls."""
        notes = score["notes"]
        duration = max(note["start"] + note["duration"] for note in notes)
        if duration <= 12:
            return []
        boundaries = [0.0]
        while duration - boundaries[-1] > 12:
            boundary = next((note["start"] for note in notes if note["start"] >= boundaries[-1] + 8), None)
            if boundary is None:
                break
            boundaries.append(boundary)
        boundaries.append(duration)
        if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < 4:
            boundaries.pop(-2)
        chunks = []
        for core_start, core_end in zip(boundaries, boundaries[1:]):
            selected = [note for note in notes
                        if note["start"] + note["duration"] > max(0, core_start - 1.2)
                        and note["start"] < min(duration, core_end + 1.2)]
            context_start = selected[0]["start"]
            context_end = selected[-1]["start"] + selected[-1]["duration"]
            subscores = score | {
                "notes": [note | {"start": note["start"] - context_start} for note in selected],
                "curves": {
                    name: [point | {"time": point["time"] - context_start}
                           for point in points if context_start <= point["time"] <= context_end]
                    for name, points in score["curves"].items()
                },
            }
            if score.get("phonemes"):
                subscores["phonemes"] = [
                    phone | {"start": max(0.0, phone["start"] - context_start),
                             "duration": min(context_end, phone["start"] + phone["duration"])
                             - max(context_start, phone["start"])}
                    for phone in score["phonemes"]
                    if phone["start"] + phone["duration"] > context_start and phone["start"] < context_end
                ]
            chunks.append({"score": subscores, "core_start": core_start, "core_end": core_end,
                           "context_start": context_start, "context_end": context_end})
        return chunks

    @staticmethod
    def _stitch_contextual(chunks: list[dict], audio: list[np.ndarray], sample_rate: int = 48_000) -> np.ndarray:
        overlap = round(.1 * sample_rate)
        pieces = []
        for index, (chunk, values) in enumerate(zip(chunks, audio)):
            start = chunk["core_start"] - chunk["context_start"] - (.05 if index else 0)
            end = chunk["core_end"] - chunk["context_start"] + (.05 if index + 1 < len(chunks) else 0)
            pieces.append(values[max(0, round(start * sample_rate)):min(len(values), round(end * sample_rate))])
        output = pieces[0]
        for piece in pieces[1:]:
            count = min(overlap, len(output), len(piece))
            weight = np.linspace(0, 1, count, dtype="float32")
            output = np.concatenate((output[:-count], output[-count:] * (1 - weight) + piece[:count] * weight,
                                     piece[count:]))
        expected = round(chunks[-1]["core_end"] * sample_rate)
        output = np.pad(output[:expected], (0, max(0, expected - len(output))))
        return output * min(1.0, .97 / max(float(np.max(np.abs(output))), 1e-8))

    @classmethod
    def _semantic_content_chunks(cls, score: dict) -> list[tuple[str, float]]:
        """Split only long, jump-heavy exact repetitions at word boundaries."""
        if score["language"] != "ja" or not cls._large_interval(score):
            return []
        text = "".join(str(note["lyric"]) for note in score["notes"])
        characters: list[tuple[float, float]] = []
        for note in score["notes"]:
            lyric = str(note["lyric"])
            if not lyric:
                continue
            start, note_duration = float(note["start"]), float(note["duration"])
            characters.extend((start + note_duration * index / len(lyric),
                               start + note_duration * (index + 1) / len(lyric))
                              for index in range(len(lyric)))
        if len(characters) != len(text):
            return []
        best = None
        for start in range(len(text)):
            for size in range(2, min(16, (len(text) - start) // 5) + 1):
                unit, count = text[start:start + size], 1
                while text.startswith(unit, start + count * size):
                    count += 1
                candidate = (count * size, -size, start, size, count)
                if count >= 5 and (best is None or candidate > best):
                    best = candidate
        if best is None:
            return []
        _, _, start, size, count = best
        stop = start + size * count
        if characters[stop - 1][1] - characters[start][0] < 6:
            return []
        spans = []
        if start:
            spans.append((text[:start], 0, start))
        spans.extend((text[index:index + size], index, index + size) for index in range(start, stop, size))
        if stop < len(text):
            spans.append((text[stop:], stop, len(text)))
        chunks = [(lyrics, characters[first][0], characters[last - 1][1]) for lyrics, first, last in spans]
        merged = []
        pending = None
        for chunk in chunks:
            pending = chunk if pending is None else (pending[0] + chunk[0], pending[1], chunk[2])
            if pending[2] - pending[1] >= .8:
                merged.append(pending)
                pending = None
        if pending is not None:
            if merged:
                previous = merged[-1]
                merged[-1] = (previous[0] + pending[0], previous[1], pending[2])
            else:
                merged.append(pending)
        return [(lyrics, end - begin) for lyrics, begin, end in merged]

    def _generate_content(self, score: dict, duration: float, content: Path, temp: Path) -> None:
        chunks = self._semantic_content_chunks(score)
        if not chunks:
            return super()._generate_content(score, duration, content, temp)
        audio = []
        sample_rate = None
        for index, (lyrics, chunk_duration) in enumerate(chunks):
            path = temp / f"content-{index}.wav"
            self.omnivoice.request({"language": "ja", "lyrics": lyrics,
                                    "duration": chunk_duration, "output": str(path)})
            values, rate = sf.read(path, dtype="float32")
            sample_rate = rate if sample_rate is None else sample_rate
            target = round(chunk_duration * rate)
            audio.append(np.pad(values[:target], (0, max(0, target - len(values)))))
        sf.write(content, np.concatenate(audio), sample_rate, subtype="PCM_16")

    @classmethod
    def _score_for_voicing(cls, score: dict) -> dict:
        if not cls._semantic_content_chunks(score):
            return score
        readings = {"息": "いき", "詰": "つ", "止": "と"}
        notes = []
        for note in score["notes"]:
            lyric = str(note["lyric"])
            for source, reading in readings.items():
                lyric = lyric.replace(source, reading)
            notes.append(note | {"lyric": lyric})
        return score | {"notes": notes}

    @classmethod
    def _needs_high_rapid_onset_relief(cls, score: dict) -> bool:
        pitches = [float(note["pitch"]) for note in score["notes"]]
        duration = max(float(note["start"]) + float(note["duration"]) for note in score["notes"])
        return (
            score["language"] == "ja" and duration >= 2.5 and cls._rapid(score)
            and min(pitches) >= 70
            and max((abs(after - before) for before, after in zip(pitches, pitches[1:])), default=0) <= 3
        )

    def _target_f0(self, score: dict, duration: float, expressive: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        f0, timeline = super()._target_f0(self._score_for_voicing(score), duration, expressive)
        if self._needs_high_rapid_onset_relief(score):
            phrase_duration = max(float(note["start"]) + float(note["duration"]) for note in score["notes"])
            for note in score["notes"]:
                onset = round(float(note["start"]) / phrase_duration * duration * 50)
                voiced = np.flatnonzero(f0[onset:onset + 5] > 1)
                if len(voiced):
                    frame = onset + int(voiced[0])
                    f0[frame] = np.exp(.05 * np.log(120.0) + .95 * np.log(f0[frame]))
            timeline = [row | {"f0_hz": float(f0[index])} for index, row in enumerate(timeline)]
        return f0, timeline

    @staticmethod
    def _bypass_post_refiners(score: dict) -> bool:
        # The Korean-trained waveform refiners destroyed rapid Japanese diction
        # in the RC9 song isolation while the latent adapters were neutral.
        pitches = [float(note["pitch"]) for note in score["notes"]]
        return score["language"] == "ja" and (
            max(pitches) >= 80 or any(abs(after - before) >= 12 for before, after in zip(pitches, pitches[1:]))
        )

    def _render_once(self, score: dict) -> np.ndarray:
        if self._bypass_post_refiners(score):
            return GyuSingerV09Renderer.render(self, score)
        return super().render(score)

    def render(self, score: dict) -> np.ndarray:
        score = normalize_score(score)
        chunks = self._contextual_subscores(score)
        if not chunks:
            return self._render_once(score)
        return self._stitch_contextual(chunks, [self._render_once(chunk["score"]) for chunk in chunks], self.sample_rate)

    def _predict_pitch(self, score: dict) -> torch.Tensor:
        predicted = self.pitch_controller.predict(score, canonical_timing=True)[0]
        # The personalized residual has real GYU singing supervision only in
        # Korean. EN/JA retain score/PITD F0 and SoulX's generic singing prior.
        return predicted if score["language"] == "ko" else torch.zeros_like(predicted)

    @classmethod
    def _content_warp_strength(cls, score: dict) -> float:
        # Japanese MMS alignment is not reliable enough for dense song lyrics.
        if cls._rapid(score) and score["language"] == "ja":
            return 0.0
        return super()._content_warp_strength(score)

    def model_info(self) -> dict:
        return super().model_info() | {
            "backend": "gyu-singer-rc9",
            "model_version": "1.0.0-rc.9-candidate",
            "rc8_baseline_backend": "gyu-singer-rc8",
            "personalized_prosody": "Korean only; EN/JA use nominal score plus user PITD",
            "rapid_japanese_content_warp": "disabled after local full-song isolation",
            "japanese_content_timing": "score-timed word chunks for long jump-heavy exact repetitions",
            "long_phrase_continuity": "8 s cores with 1.2 s score/phone/PITD context and 100 ms interior overlap",
            "high_rapid_japanese_onsets": "single-frame 5% conditioning relief toward 120 Hz for >=2.5 s stepwise high phrases",
            "japanese_post_refiners": "disabled only for >=MIDI80 or >=12-semitone jumps after causal diction isolation; latent identity/style retained",
            "release_state": "OpenUtau song candidate; human listening pending",
            "final_v1_tagged": False,
        }
