"""Small deterministic trilingual phonetic frontend used by the hybrid model."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


LANGUAGE_IDS = {"ko": 0, "en": 1, "ja": 2}
FEATURE_SIZE = 10  # onset, nucleus, coda, stress, mora, long, geminate, nasal, tense, aspirated
_JA_WORDS = {
    "空から静かな光が降りてくる": "そらからしずかなひかりがふりてくる", "空へ向かい歌おう": "そらへむかいうたおう",
    "風の上を歩こう": "かぜのうえをあるこう", "小さな光を追う": "ちいさなひかりをおう",
    "新しい夢を歌う": "あたらしいゆめをうたう", "光の向こうへ": "ひかりのむこうへ",
}
_JA_KANA = {
    "あ": ("a",), "い": ("i",), "う": ("u",), "え": ("e",), "お": ("o",), "か": ("k", "a"), "き": ("k", "i"), "く": ("k", "u"), "け": ("k", "e"), "こ": ("k", "o"),
    "が": ("g", "a"), "ぎ": ("g", "i"), "ぐ": ("g", "u"), "げ": ("g", "e"), "ご": ("g", "o"), "さ": ("s", "a"), "し": ("sh", "i"), "す": ("s", "u"), "せ": ("s", "e"), "そ": ("s", "o"),
    "ざ": ("z", "a"), "じ": ("j", "i"), "ず": ("z", "u"), "ぜ": ("z", "e"), "ぞ": ("z", "o"), "た": ("t", "a"), "ち": ("ch", "i"), "つ": ("ts", "u"), "て": ("t", "e"), "と": ("t", "o"),
    "だ": ("d", "a"), "ぢ": ("j", "i"), "づ": ("z", "u"), "で": ("d", "e"), "ど": ("d", "o"), "な": ("n", "a"), "に": ("n", "i"), "ぬ": ("n", "u"), "ね": ("n", "e"), "の": ("n", "o"),
    "は": ("h", "a"), "ひ": ("h", "i"), "ふ": ("f", "u"), "へ": ("h", "e"), "ほ": ("h", "o"), "ば": ("b", "a"), "び": ("b", "i"), "ぶ": ("b", "u"), "べ": ("b", "e"), "ぼ": ("b", "o"),
    "ぱ": ("p", "a"), "ぴ": ("p", "i"), "ぷ": ("p", "u"), "ぺ": ("p", "e"), "ぽ": ("p", "o"), "ま": ("m", "a"), "み": ("m", "i"), "む": ("m", "u"), "め": ("m", "e"), "も": ("m", "o"),
    "や": ("y", "a"), "ゆ": ("y", "u"), "よ": ("y", "o"), "ら": ("r", "a"), "り": ("r", "i"), "る": ("r", "u"), "れ": ("r", "e"), "ろ": ("r", "o"), "わ": ("w", "a"), "を": ("o",),
}
_JA_DIGRAPHS = {"きゃ": ("ky", "a"), "きゅ": ("ky", "u"), "きょ": ("ky", "o"), "ぎゃ": ("gy", "a"), "ぎゅ": ("gy", "u"), "ぎょ": ("gy", "o"), "しゃ": ("sh", "a"), "しゅ": ("sh", "u"), "しょ": ("sh", "o"), "じゃ": ("j", "a"), "じゅ": ("j", "u"), "じょ": ("j", "o"), "ちゃ": ("ch", "a"), "ちゅ": ("ch", "u"), "ちょ": ("ch", "o"), "にゃ": ("ny", "a"), "にゅ": ("ny", "u"), "にょ": ("ny", "o"), "ひゃ": ("hy", "a"), "ひゅ": ("hy", "u"), "ひょ": ("hy", "o"), "みゃ": ("my", "a"), "みゅ": ("my", "u"), "みょ": ("my", "o"), "りゃ": ("ry", "a"), "りゅ": ("ry", "u"), "りょ": ("ry", "o")}


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
    for source, reading in _JA_WORDS.items(): text = text.replace(source, reading)
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace() or char in "。、，,.!?！？":
            if units:
                units[-1] = (*units[-1][:-2], True, units[-1][-1])
            index += 1; continue
        if char == "ー":
            units.append(_unit("ja_long", "ja", (4, 5), True, False)); index += 1; continue
        if char == "っ":
            units.append(_unit("ja_geminate", "ja", (4, 6), True, False)); index += 1; continue
        if char == "ん":
            units.append(_unit("ja_n", "ja", (4, 7), True, False)); index += 1; continue
        mora = _JA_DIGRAPHS.get(text[index:index + 2]) or _JA_KANA.get(char)
        if not mora:
            units.append(_unit(f"ja_unknown_{char}", "ja", 4, True, False, True)); index += 1; continue
        for phone_index, phone in enumerate(mora):
            units.append(_unit(f"ja_{phone}", "ja", 4 if phone_index == len(mora) - 1 else 0, phone_index == len(mora) - 1, False))
        index += 2 if text[index:index + 2] in _JA_DIGRAPHS else 1
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
