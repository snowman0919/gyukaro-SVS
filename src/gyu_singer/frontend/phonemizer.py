"""Small deterministic trilingual phonetic frontend used by the hybrid model."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


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
    inferred: list[bool]


def _id(symbol: str) -> int:
    return 1 + int.from_bytes(hashlib.blake2s(symbol.encode(), digest_size=2).digest(), "big") % 2047


def _unit(symbol: str, language: str, feature: int | tuple[int, ...] | dict[int, float], syllable_end: bool, word_end: bool, inferred: bool = False) -> tuple[str, int, int, list[float], bool, bool, bool]:
    flags = [0.0] * FEATURE_SIZE
    if isinstance(feature, dict):
        for index, value in feature.items(): flags[index] = value
    else:
        for index in (feature,) if isinstance(feature, int) else feature: flags[index] = 1.0
    return symbol, _id(symbol), LANGUAGE_IDS[language], flags, syllable_end, word_end, inferred


def _korean(text: str) -> list[tuple]:
    units = []
    for char in text:
        if char.isspace():
            if units:
                units[-1] = (*units[-1][:-2], True, units[-1][-1])
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
    lexicon = json.loads(Path(__file__).with_name("english_lexicon.json").read_text())
    units = []
    for word in ("".join(char for char in raw.lower() if char.isalpha()) for raw in text.split()):
        if not word: continue
        pronunciation = lexicon.get(word)
        if pronunciation:
            for index, phone in enumerate(pronunciation):
                base, stress = phone.rstrip("012"), phone[-1] if phone[-1:].isdigit() else ""
                units.append(_unit(f"en_{base.lower()}", "en", {3: float(stress)} if stress else 0, bool(stress), index == len(pronunciation) - 1))
            continue
        # ponytail: compact grapheme-to-phoneme fallback; replace with full lexicon only when open-vocabulary pronunciation matters.
        phones, index = [], 0
        rules = {"tion": ("sh", "ah", "n"), "tch": ("ch",), "dge": ("jh",), "sh": ("sh",), "ch": ("ch",), "th": ("th",), "ph": ("f",), "ng": ("ng",), "qu": ("k", "w"), "ck": ("k",), "ee": ("iy",), "ea": ("iy",), "oo": ("uw",), "ai": ("ey",), "ay": ("ey",), "oa": ("ow",), "ow": ("aw",), "oi": ("oy",), "oy": ("oy",), "er": ("er",), "ir": ("er",), "ur": ("er",)}
        singles = {"a": "ae", "e": "eh", "i": "ih", "o": "ao", "u": "ah", "y": "iy", "c": "k", "x": "k", "q": "k", "j": "jh"}
        while index < len(word):
            matched = next((key for key in rules if word.startswith(key, index)), None)
            phones.extend(rules[matched] if matched else (singles.get(word[index], word[index]),))
            index += len(matched) if matched else 1
        stressed = False
        for index, phone in enumerate(phones):
            is_vowel = phone in {"aa", "ae", "ah", "ao", "aw", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}
            units.append(_unit(f"en_{phone}", "en", {3: 1.0} if is_vowel and not stressed else 0, is_vowel, index == len(phones) - 1, True))
            stressed |= is_vowel
    return units


def _japanese(text: str) -> list[tuple]:
    units = []
    for char in text:
        if char.isspace():
            if units:
                units[-1] = (*units[-1][:-2], True, units[-1][-1])
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
    units[-1] = (*units[-1][:-2], True, units[-1][-1])
    symbols, phoneme_ids, language_ids, features, syllable_boundaries, word_boundaries, inferred = map(list, zip(*units))
    return FrontendOutput(language, symbols, phoneme_ids, language_ids, features, syllable_boundaries, word_boundaries, inferred)
