"""SoulX probe processor with one word boundary per Korean syllable."""
from __future__ import annotations

import numpy as np
import torch

from soulxsinger.utils.data_processor import DataProcessor


class ExactKoreanDataProcessor(DataProcessor):
    """Place Korean phones at exact durations without per-phone BOW/EOW pairs."""

    def preprocess(self, note_duration, phonemes, note_pitch, note_type):
        if not any(phone.startswith("koexact:") for phone in phonemes):
            return super().preprocess(note_duration, phonemes, note_pitch, note_type)
        frame_hz = self.sample_rate / self.hop_size
        frame_count = int(sum(note_duration) * frame_hz)
        mel2note = torch.zeros(frame_count, dtype=torch.long)
        tokens, pitches, types = ["<PAD>"], [0], [1]
        cursor = 0.0
        for group, duration, pitch, kind in zip(
            phonemes, note_duration, note_pitch, note_type
        ):
            start = min(frame_count, int(round(cursor * frame_hz)))
            cursor += duration
            end = min(frame_count, max(start + 1, int(round(cursor * frame_hz))))
            if group == "<SP>":
                token = len(tokens)
                tokens.append("<SP>")
                pitches.append(pitch)
                types.append(kind)
                mel2note[start:end] = token
                continue
            entries = group.removeprefix("koexact:").split("|")
            phones, durations = zip(*(entry.rsplit("@", 1) for entry in entries))
            durations = np.array([float(value) for value in durations], dtype=np.float64)
            bow = len(tokens)
            tokens.append("<BOW>")
            pitches.append(pitch)
            types.append(kind)
            phone_tokens = []
            for phone in phones:
                phone_tokens.append(len(tokens))
                tokens.append(phone)
                pitches.append(pitch)
                types.append(kind)
            eow = len(tokens)
            tokens.append("<EOW>")
            pitches.append(pitch)
            types.append(kind)
            mel2note[start] = bow
            if end - start > 1:
                mel2note[end - 1] = eow
            inner_start, inner_end = start + 1, max(start + 1, end - 1)
            boundaries = np.rint(
                inner_start
                + np.cumsum(np.r_[0.0, durations / durations.sum()])
                * (inner_end - inner_start)
            ).astype(int)
            boundaries[0], boundaries[-1] = inner_start, inner_end
            for index, token in enumerate(phone_tokens):
                left = min(inner_end, boundaries[index])
                right = min(inner_end, max(left + 1, boundaries[index + 1]))
                mel2note[left:right] = token
        return {
            "phoneme": torch.tensor(
                [self.phone2idx[token] for token in tokens], device=self.device
            ).unsqueeze(0),
            "note_pitch": torch.tensor(pitches, device=self.device).unsqueeze(0),
            "note_type": torch.tensor(types, device=self.device).unsqueeze(0),
            "mel2note": mel2note.to(self.device).unsqueeze(0),
        }
