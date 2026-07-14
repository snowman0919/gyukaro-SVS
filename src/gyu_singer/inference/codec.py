"""Frozen pretrained MOSS audio decoder for acoustic latents."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModel


class MossCodecDecoder:
    def __init__(self, path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(str(path), trust_remote_code=True).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode [batch, 12.5-Hz frames, 768] with no waveform decoder training."""
        if latent.ndim != 3 or latent.shape[-1] != 768:
            raise ValueError("codec latent must be [batch, frames, 768]")
        values = latent.to(self.device, dtype=torch.float32).transpose(1, 2)
        lengths = torch.full((values.shape[0],), values.shape[-1], device=self.device, dtype=torch.long)
        # `latent` is MOSS encoder space. Decoder consumes quantizer-decoded space;
        # bypassing RVQ produces invalid waveform activations.
        _, codes, lengths = self.model.quantizer(values, lengths, None)
        values = self.model.quantizer.decode_codes(codes).float()
        for block in self.model.decoder:
            values, lengths = block(values, lengths)
        values, lengths = self.model._restore_channels_from_codec(values, lengths)
        return values[:, 0, : int(lengths.min())].float().cpu()
