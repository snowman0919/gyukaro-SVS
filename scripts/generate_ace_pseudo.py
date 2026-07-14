#!/usr/bin/env python3
"""Generate legal multilingual singing-teacher candidates with Apache-2.0 ACE-Step."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from acestep.pipeline_ace_step import ACEStepPipeline


PHRASES = {
    "ko": ["하늘을 향해 노래해", "작은 빛을 따라가", "바람 위로 걸어가", "새로운 꿈을 불러"],
    "en": ["Sing into the open sky", "Follow the golden light", "Carry this dream with me", "Let the morning rise"],
    "ja": ["空へ向かい歌おう", "小さな光を追う", "風の上を歩こう", "新しい夢を歌う"],
}
TARGETED_PHRASES = {
    "ko": ["낮은 숨으로 시작해", "긴 모음을 이어가", "같은 음을 다시 불러", "천천히 위로 올라가", "부드럽게 아래로 내려와", "멀리 크게 뛰어올라", "짧은 리듬을 반복해", "밝은 소리로 노래해", "어두운 빛을 품어", "바람처럼 가볍게 불러", "깊은 밤을 지나가", "새로운 아침을 맞아"],
    "en": ["Begin with a low breath", "Hold the open vowel", "Sing the same note again", "Move slowly higher now", "Drift gently downward", "Leap across the distant sky", "Repeat the short rhythm", "Sing with a brighter tone", "Carry the darker color", "Float lightly in the wind", "Travel through the deep night", "Welcome the waking morning"],
    "ja": ["ひくいこきゅうではじめよう", "ながいぼいんをのばそう", "おなじおとをくりかえす", "ゆっくりうえへすすもう", "やさしくしたへおりよう", "とおくへおおきくとぼう", "みじかいりずむをくりかえす", "あかるいこえでうたおう", "くらいいろをだいてうたう", "かぜのようにかるくうたう", "ふかいよるをこえてゆく", "あたらしいあさをむかえる"],
}
LANGUAGE_NAMES = {"ko": "Korean", "en": "English", "ja": "Japanese"}
SHAPES = ["gentle legato, mid register ascending phrase", "low register sustained vowel", "high register descending phrase", "repeated notes short rhythm", "gentle vibrato", "bright energetic pop", "soft breathy ballad"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--checkpoint", default="data/cache/ace-step-checkpoint")
    parser.add_argument("--output", default="data/pseudo_singing/ace_step_teacher")
    parser.add_argument("--manifest", default="data/manifests/ace_step_candidates.jsonl")
    parser.add_argument("--targeted", action="store_true", help="use v0.6 gap-driven text/shape coverage")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    phrases = TARGETED_PHRASES if args.targeted else PHRASES
    shapes = SHAPES + ["wide interval leap", "slow sustained final vowel"] if args.targeted else SHAPES
    rows = []
    model = ACEStepPipeline(checkpoint_dir=args.checkpoint, dtype="bfloat16")
    for index in range(args.count):
        language = ("ko", "en", "ja")[index % 3]
        phrase_index = (index // 3) % len(phrases[language])
        text = "\n".join((phrases[language][phrase_index], phrases[language][(phrase_index + 1) % len(phrases[language])]))
        shape = shapes[(index // 3 // len(phrases[language])) % len(shapes)]
        identifier = f"ace_step_{language}_{index + 1:03d}"
        wav = output / f"{identifier}.wav"
        if not wav.exists():
            model(audio_duration=10, prompt=f"a cappella {LANGUAGE_NAMES[language]} female singer, clean dry solo vocal, no instruments, {shape}", lyrics=f"[verse]\n{text}", infer_step=20, guidance_scale=7, manual_seeds=[101 + index], save_path=str(wav))
        rows.append({"id": identifier, "language": language, "text": text, "shape": shape, "coverage_target": "v0.6 pitch_range_interval_duration_transition_language" if args.targeted else "v0.5_general", "teacher": "ACE-Step-v1-3.5B", "teacher_license": "Apache-2.0", "teacher_commit": "1bee4c9f5b43e30995f8d4d33b3919197ce1bd68", "source_output_path": str(wav), "quality_status": "pending_svc"})
        print(identifier, flush=True)
    Path(args.manifest).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    main()
