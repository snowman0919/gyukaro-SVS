#!/usr/bin/env python3
"""Report singing-aware alignment metrics and save auditable timelines."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main() -> None:
    alignments = read("data/manifests/real_phoneme_alignment.jsonl"); scores = {row["id"]: row for row in read("data/manifests/real_score_accepted.jsonl")}
    fractions = []
    for row in alignments:
        phones = row["phones"]; total = sum(phone["duration"] for phone in phones); nucleus = sum(phone["duration"] for phone in phones if "nucleus" in phone["symbol"]); fractions.append(nucleus / max(total, 1e-6))
    out = Path("artifacts/reports/alignment_timelines"); out.mkdir(parents=True, exist_ok=True)
    for row in alignments[:3]:
        score = scores[row["id"]]; audio, rate = sf.read(score["source_audio_path"], dtype="float32", always_2d=True); audio = audio.mean(1); times = np.arange(len(audio)) / rate
        fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True); axes[0].plot(times, audio, linewidth=.4); axes[0].set_ylabel("wave")
        f0 = np.load(score["f0_path"]); axes[1].plot(np.arange(len(f0)) / score.get("f0_frame_hz", 12.5), f0, linewidth=.8); axes[1].set_ylabel("RMVPE Hz")
        for note in score["notes"]: axes[2].plot([note["start"], note["start"] + note["duration"]], [note["pitch"]] * 2, linewidth=3)
        for phone in row["phones"]: axes[2].axvspan(phone["start"], phone["start"] + phone["duration"], alpha=.15, color="tab:orange" if "nucleus" in phone["symbol"] else "tab:blue")
        axes[2].set_ylabel("notes/phones"); axes[2].set_xlabel("seconds"); fig.tight_layout(); fig.savefig(out / f"{row['id']}.png", dpi=120); plt.close(fig)
    report = ["# Singing-aware phoneme alignment (v0.5)", "", f"- rows: {len(alignments)}", f"- mean vowel-nucleus share: {np.mean(fractions):.3f}", f"- p10 vowel-nucleus share: {np.percentile(fractions, 10):.3f}", f"- rows above 50% nucleus share: {sum(value > .5 for value in fractions)}/{len(fractions)}", "- alignment: MMS multilingual CTC + Korean onset/nucleus/coda singing prior", "- uniform note splitting: rejected by `uniform_split_guard`", "- timelines: `artifacts/reports/alignment_timelines/`", ""]
    Path("docs/phoneme_alignment_report.md").write_text("\n".join(report))
    print({"rows": len(alignments), "mean_nucleus_share": round(float(np.mean(fractions)), 4), "p10": round(float(np.percentile(fractions, 10)), 4)})


if __name__ == "__main__": main()
