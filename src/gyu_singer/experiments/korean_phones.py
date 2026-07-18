from __future__ import annotations

from dataclasses import dataclass


REPRESENTATIONS = ("ko_components_v1", "ko_canonical_v1", "ko_onset_rhyme_v1")
_ONSETS = (
    "k", "k_tense", "n", "t", "t_tense", "r", "m", "p", "p_tense", "s",
    "s_tense", "silent", "c", "c_tense", "c_aspirated", "k_aspirated",
    "t_aspirated", "p_aspirated", "h",
)
_VOWELS = (
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
    "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
)
_CODAS = (
    "none", "k", "k_tense", "ks", "n", "nc", "nh", "t", "l", "lk", "lm",
    "lp", "ls", "lt", "lph", "lh", "m", "p", "ps", "s", "s_tense", "ng",
    "c", "ch", "kh", "th", "ph", "h",
)
_MMS_ONSETS = ("g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h")
_MMS_VOWELS = ("a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i")
_MMS_CODAS = ("", "g", "kk", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h")


@dataclass(frozen=True)
class EncodedKorean:
    representation: str
    symbols: tuple[str, ...]
    audit_text: str
    audit: tuple[dict, ...]
    unknown_characters: tuple[str, ...]


def encode_korean(text: str, representation: str) -> EncodedKorean:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown Korean representation: {representation}")
    symbols, audit, unknown = [], [], []
    for char in text:
        if char.isspace():
            continue
        code = ord(char) - 0xAC00
        if not 0 <= code < 11172:
            unknown.append(char)
            symbols.append(f"ko_unknown_{ord(char):x}")
            continue
        onset, rest = divmod(code, 21 * 28)
        nucleus, coda = divmod(rest, 28)
        if representation == "ko_components_v1":
            encoded = [f"ko_onset_{onset}", f"ko_nucleus_{nucleus}"]
            if coda:
                encoded.append(f"ko_coda_{coda}")
        elif representation == "ko_canonical_v1":
            encoded = [f"ko_canonical_{_ONSETS[onset]}", f"ko_canonical_vowel_{_VOWELS[nucleus]}"]
            if coda:
                encoded.append(f"ko_canonical_coda_{_CODAS[coda]}")
        else:
            encoded = [f"ko_onset_{onset}", f"ko_rhyme_{nucleus}_{coda}"]
        symbols.extend(encoded)
        audit.append({"character": char, "onset": onset, "nucleus": nucleus,
                      "coda": coda, "symbols": encoded})
    if not symbols:
        raise ValueError("text produced no Korean phones")
    return EncodedKorean(representation, tuple(symbols), "".join(row["character"] for row in audit),
                         tuple(audit), tuple(unknown))


def representation_coverage(texts: list[str], representation: str) -> dict:
    encoded = [encode_korean(text, representation) for text in texts]
    maximum = {"ko_components_v1": 68, "ko_canonical_v1": 68, "ko_onset_rhyme_v1": 607}[representation]
    return {
        "representation": representation,
        "observed_symbols": len({symbol for row in encoded for symbol in row.symbols}),
        "maximum_symbols": maximum,
        "unknown_characters": sorted({char for row in encoded for char in row.unknown_characters}),
    }


def mms_alignment_target(text: str) -> str:
    output = []
    for char in text:
        code = ord(char) - 0xAC00
        if not 0 <= code < 11172:
            continue
        onset, rest = divmod(code, 21 * 28)
        nucleus, coda = divmod(rest, 28)
        output.extend((_MMS_ONSETS[onset], _MMS_VOWELS[nucleus], _MMS_CODAS[coda]))
    if not output:
        raise ValueError("text produced no MMS alignment target")
    return "".join(output)
