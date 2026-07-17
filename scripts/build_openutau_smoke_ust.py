#!/usr/bin/env python3
"""Create a redistribution-safe Japanese OpenUtau score-native smoke UST."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/external/work/openutau_native_candidate/rapid_c4_smoke.ust"
LYRICS = tuple("いきがつまる") * 4


def main() -> None:
    lines = [
        "[#VERSION]", "UST Version1.2", "[#SETTING]", "Tempo=148.00",
        "Tracks=1", "ProjectName=rapid-c4-smoke", "VoiceDir=%VOICE%", "Mode2=True",
    ]
    for index, lyric in enumerate(LYRICS):
        lines.extend((f"[#{index:04d}]", "Length=120", f"Lyric={lyric}", "NoteNum=60"))
    lines.append("[#TRACKEND]")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="cp932")
    print(OUTPUT)


if __name__ == "__main__":
    main()
