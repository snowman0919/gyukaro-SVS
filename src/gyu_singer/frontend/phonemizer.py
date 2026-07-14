"""Small deterministic trilingual phonetic frontend used by the hybrid model."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


LANGUAGE_IDS = {"ko": 0, "en": 1, "ja": 2}
FEATURE_SIZE = 10  # onset, nucleus, coda, stress, mora, long, geminate, nasal, tense, aspirated


@dataclass(frozen=True)
class FrontendOutput:
    language: str
    symbols: list[str]
    phoneme_ids: list[int]
    language_ids: list[int]
    features: list[list[float]]
    syllable_boundaries: list[bool]
    word_boundaries: list[bool]


def _id(symbol: str) -> int:
    return 1 + int.from_bytes(hashlib.blake2s(symbol.encode(), digest_size=2).digest(), "big") % 2047


def _unit(symbol: str, language: str, feature: int | tuple[int, ...], syllable_end: bool, word_end: bool) -> tuple[str, int, int, list[float], bool, bool]:
    flags = [0.0] * FEATURE_SIZE
    for index in (feature,) if isinstance(feature, int) else feature:
        flags[index] = 1.0
    return symbol, _id(symbol), LANGUAGE_IDS[language], flags, syllable_end, word_end


def _korean(text: str) -> list[tuple]:
    units = []
    for char in text:
        if char.isspace():
            if units:
                units[-1] = (*units[-1][:-1], True)
            continue
        code = ord(char) - 0xAC00
        if not 0 <= code < 11172:
            units.append(_unit(f"ko_{char}", "ko", 0, True, False)); continue
        onset, rest = divmod(code, 21 * 28)
        nucleus, coda = divmod(rest, 28)
        onset_features = (0,) + ((8,) if onset in {1, 4, 8, 10, 13} else ()) + ((9,) if onset in {14, 15, 16, 17} else ())
        units.append(_unit(f"ko_onset_{onset}", "ko", onset_features, False, False))
        units.append(_unit(f"ko_nucleus_{nucleus}", "ko", 1, coda == 0, False))
        if coda:
            units.append(_unit(f"ko_coda_{coda}", "ko", 2, True, False))
    return units


def _english(text: str) -> list[tuple]:
    units = []
    vowels = set("aeiouy")
    words = [word for word in text.lower().split() if word]
    for word_index, word in enumerate(words):
        letters = [char for char in word if char.isalpha()]
        for index, char in enumerate(letters):
            feature = 3 if char in vowels else 0
            units.append(_unit(f"en_{char}", "en", feature, index == len(letters) - 1, index == len(letters) - 1))
    return units


def _japanese(text: str) -> list[tuple]:
    units = []
    for char in text:
        if char.isspace():
            if units:
                units[-1] = (*units[-1][:-1], True)
            continue
        if char == "ー": feature, symbol = 5, "ja_long"
        elif char == "っ": feature, symbol = 6, "ja_geminate"
        elif char == "ん": feature, symbol = 7, "ja_nasal"
        else: feature, symbol = 4, f"ja_{char}"
        units.append(_unit(symbol, "ja", feature, True, False))
    return units


def phonemize(language: str, text: str) -> FrontendOutput:
    if language not in LANGUAGE_IDS:
        raise ValueError(f"unsupported language: {language}")
    units = {"ko": _korean, "en": _english, "ja": _japanese}[language](text)
    if not units:
        raise ValueError("text produced no phonemes")
    # Explicit terminal boundary also for scripts without whitespace.
    units[-1] = (*units[-1][:-1], True)
    symbols, phoneme_ids, language_ids, features, syllable_boundaries, word_boundaries = map(list, zip(*units))
    return FrontendOutput(language, symbols, phoneme_ids, language_ids, features, syllable_boundaries, word_boundaries)
