"""RC9 OpenUtau song candidate with independently controlled non-KO pitch."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .rc8 import GyuSingerRC8Renderer
from .v09 import GyuSingerV09Renderer


class GyuSingerRC9Renderer(GyuSingerRC8Renderer):
    """Keep RC8 audio fixes while making EN/JA score and PITD authoritative."""

    @classmethod
    def _semantic_content_chunks(cls, score: dict) -> list[tuple[str, float]]:
        """Split only long, jump-heavy exact repetitions at word boundaries."""
        if score["language"] != "ja" or not cls._large_interval(score):
            return []
        text = "".join(str(note["lyric"]) for note in score["notes"])
        unit = next((text[:size] for size in range(1, len(text) // 5 + 1)
                     if len(text) % size == 0 and text == text[:size] * (len(text) // size)), None)
        if unit is None or len(text) // len(unit) < 5:
            return []
        duration = max(float(note["start"]) + float(note["duration"]) for note in score["notes"])
        if duration < 6 or max(abs(float(after["pitch"]) - float(before["pitch"]))
                               for before, after in zip(score["notes"], score["notes"][1:])) < 14:
            return []
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
        chunks = [(unit, characters[index][0], characters[index + len(unit) - 1][1])
                  for index in range(0, len(text), len(unit))]
        if chunks[-1][2] - chunks[-1][1] < .8 and len(chunks) > 1:
            previous = chunks[-2]
            chunks[-2:] = [(previous[0] + chunks[-1][0], previous[1], chunks[-1][2])]
        return [(lyrics, end - start) for lyrics, start, end in chunks]

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

    def render(self, score: dict) -> np.ndarray:
        if self._bypass_post_refiners(score):
            return GyuSingerV09Renderer.render(self, score)
        return super().render(score)

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
            "high_rapid_japanese_onsets": "single-frame 5% conditioning relief toward 120 Hz for >=2.5 s stepwise high phrases",
            "japanese_post_refiners": "disabled only for >=MIDI80 or >=12-semitone jumps after causal diction isolation; latent identity/style retained",
            "release_state": "OpenUtau song candidate; human listening pending",
            "final_v1_tagged": False,
        }
