#!/usr/bin/env python3
"""Build the fixed 100×3 teacher-evaluation text set."""
from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path


REFERENCES = [
    (216, "low_register"),
    (220, "neutral_mid"),
    (215, "high_register"),
    (219, "expressive_phrase"),
    (212, "long_natural_phrase"),
]
STYLES = ["neutral", "soft", "breathy", "energetic", "bright"]
TEXTS = {
    "ko": [
        "가까운 강가에서 작은 새가 노래한다.", "맑은 아침 공기가 창문으로 들어온다.",
        "까만 구름 뒤로 따뜻한 빛이 번진다.", "꽃길 끝에서 친구와 천천히 웃는다.",
        "달빛 아래 긴 그림자가 조용히 흔들린다.", "빨간 풍선이 파란 하늘로 높이 오른다.",
        "책상 위 연필이 또각또각 리듬을 만든다.", "차가운 바람 끝에 하얀 눈이 내려온다.",
        "작은 문을 열면 새로운 길이 시작된다.", "밤의 별빛이 마음속 깊이 반짝인다.",
        "햇살을 따라 한 걸음씩 앞으로 걸어간다.", "깊은 숨을 쉬고 다시 밝게 노래한다.",
        "빛나는 꿈을 품고 오늘을 힘차게 달린다.", "여름 바다에는 파도가 부드럽게 춤춘다.",
        "가을 숲길에서 낙엽이 사각사각 울린다.", "겨울 새벽의 공기는 맑고 투명하다.",
        "천천히 피어난 꽃향기가 방 안을 채운다.", "새로운 약속을 지키며 서로를 기다린다.",
        "끝없는 하늘 아래 우리의 목소리가 만난다.", "괜찮아, 오늘도 너는 충분히 빛난다.",
    ],
    "en": [
        "A bright breeze moves across the quiet bridge.", "Small clouds drift above a silver river.",
        "The black train stops beside the green field.", "Fresh flowers bloom after a gentle rain.",
        "Three blue stars shine through the clear night.", "A strong drum keeps a steady rhythm.",
        "The warm light falls softly on the window.", "Quick steps cross the wooden floor.",
        "The children laugh beside the old stone gate.", "A calm voice carries over the distant hill.",
        "We breathe slowly and begin the song again.", "Bright dreams rise beyond the morning sky.",
        "The brave traveler follows a winding road.", "Golden leaves whisper under careful feet.",
        "A long vowel floats over the open sea.", "The clock ticks while the garden grows.",
        "Please bring the bright red scarf tomorrow.", "Soft thunder rolls across the autumn valley.",
        "Our final note fades into the moonlight.", "Even in winter, hope can stay warm.",
    ],
    "ja": [
        "あさのひかりがまどからそっと入る。", "きれいなそらにしろいくもがながれる。",
        "がっこうのまえでちいさなとりがうたう。", "きってをはっててがみをおくる。",
        "ゆっくりあるけばかぜのねがきこえる。", "しんぶんをよんであたらしいことをしる。",
        "きょうはきっといいひになる。", "おんがくにあわせてこころがはずむ。",
        "ながいみちのさきにうみがみえる。", "まっかなりんごをふたつわけあう。",
        "あんしんしてふかくいきをする。", "きらきらしたほしがよるをてらす。",
        "しゃしんのなかでみんながわらう。", "せっけんのかおりがへやにひろがる。",
        "でんしゃのまどからまちをながめる。", "おおきなゆめをむねにだきしめる。",
        "しずかなもりでこだまがかえる。", "ちょっとまって、いまいくよ。",
        "あしたのためにやさしくうたう。", "さいごのもらがきれいにのびる。",
    ],
}


def main() -> None:
    output = Path("configs/teachers/trilingual_pilot.jsonl")
    segment_text = {row["source_index"]: row["text"] for row in _segments()}
    rows = []
    for language, texts in TEXTS.items():
        index = 1
        for style in STYLES:
            for text, reference in zip(texts, cycle(REFERENCES)):
                source_index, reference_role = reference
                rows.append({
                    "id": f"teacher_{language}_{index:03d}",
                    "language": language,
                    "text": text,
                    "style": style,
                    "style_prompt": style,
                    "reference_ids": [f"gyu_real_{source_index:06d}"],
                    "reference_role": reference_role,
                    "reference_audio_path": f"data/source/Korea Digital Media High School {source_index}.m4a",
                    "reference_text": segment_text[source_index],
                })
                index += 1
    assert len(rows) == 300 and {r["language"] for r in rows} == set(TEXTS)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(rows)} rows to {output}")


def _segments() -> list[dict]:
    return [json.loads(line) for line in Path("data/manifests/real_segments.jsonl").read_text().splitlines()]


if __name__ == "__main__":
    main()
