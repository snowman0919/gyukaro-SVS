#!/usr/bin/env python3
"""Convert ACE-Step vocal candidates with Apache-2.0 SoulX SVC in one resident process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from preprocess.tools.f0_extraction import F0Extractor
from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
from soulxsinger.utils.audio_utils import load_wav
from soulxsinger.utils.file_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/ace_step_candidates.jsonl")
    parser.add_argument("--output", default="data/pseudo_singing/ace_step_soulx")
    parser.add_argument("--reference", default="data/processed/master/216.wav")
    parser.add_argument("--model", default="data/cache/soulx-singer/pretrained_models/SoulX-Singer/model-svc.pt")
    parser.add_argument("--config", default="data/cache/soulx-singer/soulxsinger/config/soulxsinger.yaml")
    parser.add_argument("--rmvpe", default="data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    model = SoulXSingerSVC(config).cuda(); model.load_state_dict(torch.load(args.model, weights_only=False, map_location="cpu")["state_dict"]); model.half(); model.mel.float(); model.eval()
    f0 = F0Extractor(args.rmvpe, device="cuda", target_sr=24000, hop_size=480, verbose=False)
    reference = load_wav(args.reference, config.audio.sample_rate).cuda()
    ref_f0_path = output / "gyu_reference_f0.npy"
    if not ref_f0_path.exists(): f0.process(args.reference, f0_path=str(ref_f0_path), verbose=False)
    ref_f0 = torch.from_numpy(np.load(ref_f0_path)).unsqueeze(0).cuda()
    for row in rows:
        target = output / f"{row['id']}.wav"; target_f0_path = output / f"{row['id']}_source_f0.npy"
        if not target_f0_path.exists(): f0.process(row["source_output_path"], f0_path=str(target_f0_path), verbose=False)
        if not target.exists():
            source = load_wav(row["source_output_path"], config.audio.sample_rate).cuda()
            source_f0 = torch.from_numpy(np.load(target_f0_path)).unsqueeze(0).cuda()
            with torch.inference_mode(): audio, _ = model.infer(reference, source, ref_f0, source_f0, auto_shift=True, pitch_shift=0, n_steps=16, cfg=2.5, use_fp16=True)
            sf.write(target, audio.squeeze().float().cpu().numpy(), config.audio.sample_rate)
        audio, rate = sf.read(target, dtype="float32")
        if rate != 48000:
            sf.write(target, resample_poly(audio, 48000, rate), 48000)
        row["output_path"] = str(target); row["generator"] = "ACE-Step lyric vocal -> SoulX SVC"; row["generator_license"] = "Apache-2.0"; row["quality_status"] = "pending_gate"
        print(row["id"], flush=True)
    Path(args.input).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    main()
