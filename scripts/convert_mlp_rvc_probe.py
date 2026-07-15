#!/usr/bin/env python3
"""Convert the stable Korean score-native source with the bounded GYU RVC model."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch
from transformers import HubertModel


ROOT = Path(__file__).resolve().parents[1]
RVC = ROOT / "data/cache/rvc"
sys.path.insert(0, str(RVC))

from infer.lib.audio import load_audio  # noqa: E402
from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid  # noqa: E402
from infer.modules.vc.pipeline import Pipeline  # noqa: E402


class TransformersHubert(torch.nn.Module):
    """Expose the fairseq call surface expected by the RVC pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.model = HubertModel.from_pretrained(
            ROOT / "data/cache/hubert-base-ls960", torch_dtype=torch.float16
        ).cuda().eval()

    def extract_features(self, source, padding_mask=None, output_layer=12):
        source = source.float()
        source = (source - source.mean(-1, keepdim=True)) / torch.sqrt(
            source.var(-1, keepdim=True, unbiased=False) + 1e-7
        )
        with torch.inference_mode():
            hidden = self.model(source.half()).last_hidden_state
        return (hidden,)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="gyu_rvc_e5.pth")
    parser.add_argument("--label", default="rvc_e5")
    args = parser.parse_args()
    os.environ["rmvpe_root"] = str(RVC / "assets/rmvpe")
    checkpoint = torch.load(
        RVC / "assets/weights" / args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["config"][-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
    net = SynthesizerTrnMs768NSFsid(*checkpoint["config"], is_half=True)
    del net.enc_q
    net.load_state_dict(checkpoint["weight"], strict=False)
    net.half().cuda().eval()
    config = SimpleNamespace(
        device="cuda:0",
        is_half=True,
        x_pad=3,
        x_query=10,
        x_center=60,
        x_max=65,
    )
    pipeline = Pipeline(48000, config)
    model = TransformersHubert()
    output = ROOT / "artifacts/reports/mlp_singer_korean_probe/listening" / args.label
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "rapid_ko": (
            ROOT / "artifacts/reports/mlp_singer_korean_probe/listening/rapid_ko_c6_generated_e2e.wav",
            0,
        ),
        "large_interval_ko": (
            ROOT / "artifacts/reports/mlp_singer_korean_probe/listening/large_interval_ko_c6_generated_e2e.wav",
            2,
        ),
    }
    for case, (source_path, semitones) in inputs.items():
        audio = load_audio(str(source_path), 16000)
        peak = abs(audio).max() / 0.95
        if peak > 1:
            audio /= peak
        rendered = pipeline.pipeline(
            model,
            net,
            0,
            audio,
            str(source_path),
            [0, 0, 0],
            semitones,
            "rmvpe",
            "",
            0.0,
            1,
            3,
            48000,
            48000,
            1.0,
            "v2",
            0.5,
            None,
        )
        sf.write(output / f"{case}.wav", rendered, 48000, subtype="PCM_16")
        print(case, len(rendered) / 48000, flush=True)


if __name__ == "__main__":
    main()
