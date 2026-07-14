#!/usr/bin/env python3
"""Freeze CMU pronunciations used by English manifests for dependency-free runtime."""
from __future__ import annotations

import json
from pathlib import Path

from nltk.corpus import cmudict


def main() -> None:
    words = set()
    for path in Path("data/manifests").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("language") == "en":
                words.update("".join(char for char in word.lower() if char.isalpha()) for word in row.get("text", "").split())
    words.update({"soft", "voice"})
    pronunciations = cmudict.dict()
    lexicon = {word: pronunciations[word][0] for word in sorted(words) if word in pronunciations}
    output = Path("src/gyu_singer/frontend/english_lexicon.json")
    output.write_text(json.dumps(lexicon, indent=2, sort_keys=True) + "\n")
    print(f"{output}: {len(lexicon)} pronunciations")


if __name__ == "__main__":
    main()
