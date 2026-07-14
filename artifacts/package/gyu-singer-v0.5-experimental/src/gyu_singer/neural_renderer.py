from __future__ import annotations

import math
import json
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForCausalLM

from .baseline_renderer import midi_hz


def estimate_f0(audio: np.ndarray, rate: int) -> float:
    frame = audio[: min(len(audio), rate * 2)]
    frame = frame[np.abs(frame) > 0.005]
    if len(frame) < 2048:
        return 136.0
    frame = frame[:8192] * np.hanning(min(len(frame), 8192))
    corr = np.correlate(frame, frame, "full")[len(frame) - 1 :]
    lo, hi = int(rate / 450), int(rate / 70)
    lag = lo + int(np.argmax(corr[lo:hi]))
    return rate / lag


class NeuralRenderer:
    """Score-controlled vocalizer: GYU-reference multilingual TTS plus note transform."""

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path, reference_path: str | Path, sample_rate: int = 48000):
        self.model_path, self.tokenizer_path, self.reference_path = map(str, (model_path, tokenizer_path, reference_path))
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ponytail: one shared CUDA model is serialized; add worker processes only when concurrent renders matter.
        self._inference_lock = threading.Lock()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        self.model._set_attention_implementation("sdpa")
        self.model = self.model.to(self.device, dtype=dtype).eval()

    def _vocalize_note(self, lyric: str, pitch: float, duration: float) -> np.ndarray:
        frames = max(80, math.ceil(max(duration, 1.2) * 12.5))
        # Nano refuses some one-character Japanese prompts; repetition is vowel sustain, not new content.
        text = lyric * 3 if len(lyric.replace(" ", "")) == 1 else lyric
        raw_path = Path("/tmp/gyu_note.wav")
        try:
            with self._inference_lock, torch.inference_mode():
                try:
                    self.model.inference(text=text or "아", output_audio_path=raw_path, mode="voice_clone", prompt_audio_path=self.reference_path, audio_tokenizer_type="moss-audio-tokenizer-nano", audio_tokenizer_pretrained_name_or_path=self.tokenizer_path, device=self.device, max_new_frames=frames, do_sample=False)
                except RuntimeError:
                    self.model.inference(text=(text or "아") * 3, output_audio_path=raw_path, mode="voice_clone", prompt_audio_path=self.reference_path, audio_tokenizer_type="moss-audio-tokenizer-nano", audio_tokenizer_pretrained_name_or_path=self.tokenizer_path, device=self.device, max_new_frames=160, do_sample=False)
                audio, rate = sf.read(raw_path, dtype="float32", always_2d=True)
            audio = torch.from_numpy(audio.mean(1))
        except RuntimeError:
            # ponytail: Nano may emit an empty token sequence for isolated phonemes; replace with real GYU reference until acoustic finetuning fixes this.
            audio, rate = torchaudio.load(self.reference_path)
            audio = audio.mean(0)
        if rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, rate, self.sample_rate)
        base = estimate_f0(audio.numpy(), self.sample_rate)
        steps = float(np.clip(12 * math.log2(midi_hz(pitch) / base), -18, 18))
        shifted = torchaudio.functional.pitch_shift(audio, self.sample_rate, steps, n_fft=1024)
        target = max(1, int(duration * self.sample_rate))
        spec = torch.stft(shifted, n_fft=1024, hop_length=256, window=torch.hann_window(1024), return_complex=True)
        rate = max(0.1, len(shifted) / target)
        stretched = torchaudio.functional.phase_vocoder(spec, rate, torch.linspace(0, math.pi * 256, spec.shape[-2])[..., None])
        output = torch.istft(stretched, n_fft=1024, hop_length=256, window=torch.hann_window(1024), length=target)
        return output.numpy()

    def render(self, score: dict) -> np.ndarray:
        rate = int(score.get("sample_rate", self.sample_rate))
        if rate != self.sample_rate:
            raise ValueError("neural renderer currently outputs 48000 Hz")
        notes = score.get("notes", [])
        if not notes:
            raise ValueError("score.notes must not be empty")
        end = max(float(note["start"]) + float(note["duration"]) for note in notes)
        output = np.zeros(int((end + .05) * rate), np.float32)
        for note in notes:
            start, duration = float(note["start"]), float(note["duration"])
            rendered = self._vocalize_note(str(note.get("lyric", "아")), float(note["pitch"]), duration)
            fade = min(int(rate * .015), len(rendered) // 3)
            if fade:
                rendered[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
                rendered[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            offset = int(start * rate)
            output[offset:offset + len(rendered)] += rendered * float(note.get("dynamics", .8))
        peak = float(np.max(np.abs(output)))
        return output / max(1, peak / .92)

    def render_file(self, input_path: str | Path, output_path: str | Path) -> None:
        score = json.loads(Path(input_path).read_text())
        sf.write(output_path, self.render(score), self.sample_rate, subtype="PCM_24")
