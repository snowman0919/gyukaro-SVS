#!/usr/bin/env python3
"""Build the native OpenUtau DiffSinger evaluation singer deterministically.

This is a non-release candidate.  Its acoustic and duration models are derived
from GTSinger and therefore remain CC BY-NC-SA 4.0.  The script deliberately
does not copy any evaluation UST or user-provided reference audio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import yaml

try:
    from scripts.embed_diffsinger_voicing_mask import embed_voicing_mask
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from embed_diffsinger_voicing_mask import embed_voicing_mask


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/work/openutau_gyu_diffsinger_candidate"
DEFAULT_OUTPUT = ROOT / "data/external/work/openutau_native_candidate/GYU-DiffSinger-JA-eval"

VOWELS = {
    "a_ja", "aː_ja", "e_ja", "eː_ja", "i_ja", "i̥_ja", "o_ja", "oː_ja",
    "ɨ_ja", "ɨː_ja", "ɨ̥_ja", "ɯ_ja", "ɯ̥_ja",
}
NASALS = {"m_ja", "mʲ_ja", "n_ja", "nː_ja", "ɲ_ja", "ɴ_ja", "ɴː_ja"}
STOPS = {"b_ja", "bʲ_ja", "d_ja", "k_ja", "p_ja", "t_ja", "ɟ_ja", "ɡ_ja", "ʔ_ja"}
AFFRICATES = {"dz_ja", "dʑ_ja", "ts_ja", "tɕ_ja"}
FRICATIVES = {"h_ja", "s_ja", "z_ja", "ç_ja", "ɕ_ja", "ʑ_ja", "ɸ_ja"}
SEMIVOWELS = {"j_ja", "w_ja"}
LIQUIDS = {"ɾ_ja", "ɾʲ_ja"}
UNVOICED_PHONEMES = {
    "AP", "SP", "c_ja", "h_ja", "k_ja", "p_ja", "s_ja", "t_ja", "ts_ja",
    "tɕ_ja", "ç_ja", "ɕ_ja", "ɸ_ja", "ʔ_ja", "i̥_ja", "ɨ̥_ja", "ɯ̥_ja",
}


def mora_entries() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {
        "a": ["a_ja"], "i": ["i_ja"], "u": ["ɯ_ja"], "e": ["e_ja"], "o": ["o_ja"],
        "n": ["ɴ_ja"], "q": ["ʔ_ja"], "xtsu": ["ʔ_ja"], "ltsu": ["ʔ_ja"],
        "vu": ["b_ja", "ɯ_ja"], "ye": ["j_ja", "e_ja"], "br_inA3": ["AP"],
    }
    rows.update({
        "ka": ["k_ja", "a_ja"], "ki": ["k_ja", "i_ja"], "ku": ["k_ja", "ɯ_ja"],
        "ke": ["k_ja", "e_ja"], "ko": ["k_ja", "o_ja"],
        "ga": ["ɡ_ja", "a_ja"], "gi": ["ɡ_ja", "i_ja"], "gu": ["ɡ_ja", "ɯ_ja"],
        "ge": ["ɡ_ja", "e_ja"], "go": ["ɡ_ja", "o_ja"],
        "sa": ["s_ja", "a_ja"], "shi": ["ɕ_ja", "i_ja"], "su": ["s_ja", "ɨ_ja"],
        "se": ["s_ja", "e_ja"], "so": ["s_ja", "o_ja"],
        "za": ["dz_ja", "a_ja"], "ji": ["dʑ_ja", "i_ja"], "zu": ["dz_ja", "ɨ_ja"],
        "ze": ["dz_ja", "e_ja"], "zo": ["dz_ja", "o_ja"],
        "ta": ["t_ja", "a_ja"], "chi": ["tɕ_ja", "i_ja"], "tsu": ["ts_ja", "ɨ_ja"],
        "te": ["t_ja", "e_ja"], "to": ["t_ja", "o_ja"],
        "da": ["d_ja", "a_ja"], "di": ["d_ja", "i_ja"], "du": ["d_ja", "ɨ_ja"],
        "de": ["d_ja", "e_ja"], "do": ["d_ja", "o_ja"],
        "na": ["n_ja", "a_ja"], "ni": ["ɲ_ja", "i_ja"], "nu": ["n_ja", "ɯ_ja"],
        "ne": ["n_ja", "e_ja"], "no": ["n_ja", "o_ja"],
        "ha": ["h_ja", "a_ja"], "hi": ["ç_ja", "i_ja"], "fu": ["ɸ_ja", "ɯ_ja"],
        "he": ["h_ja", "e_ja"], "ho": ["h_ja", "o_ja"],
        "ba": ["b_ja", "a_ja"], "bi": ["bʲ_ja", "i_ja"], "bu": ["b_ja", "ɯ_ja"],
        "be": ["b_ja", "e_ja"], "bo": ["b_ja", "o_ja"],
        "pa": ["p_ja", "a_ja"], "pi": ["p_ja", "i_ja"], "pu": ["p_ja", "ɯ_ja"],
        "pe": ["p_ja", "e_ja"], "po": ["p_ja", "o_ja"],
        "ma": ["m_ja", "a_ja"], "mi": ["mʲ_ja", "i_ja"], "mu": ["m_ja", "ɯ_ja"],
        "me": ["m_ja", "e_ja"], "mo": ["m_ja", "o_ja"],
        "ya": ["j_ja", "a_ja"], "yu": ["j_ja", "ɯ_ja"], "yo": ["j_ja", "o_ja"],
        "ra": ["ɾ_ja", "a_ja"], "ri": ["ɾʲ_ja", "i_ja"], "ru": ["ɾ_ja", "ɯ_ja"],
        "re": ["ɾ_ja", "e_ja"], "ro": ["ɾ_ja", "o_ja"],
        "wa": ["w_ja", "a_ja"], "wi": ["w_ja", "i_ja"], "we": ["w_ja", "e_ja"],
        "wo": ["o_ja"],
    })
    palatals = {
        "ky": "c_ja", "gy": "ɟ_ja", "sh": "ɕ_ja", "j": "dʑ_ja", "ch": "tɕ_ja",
        "ny": "ɲ_ja", "hy": "ç_ja", "by": "bʲ_ja", "py": "p_ja",
        "my": "mʲ_ja", "ry": "ɾʲ_ja",
    }
    for prefix, phone in palatals.items():
        for suffix, vowel in (("a", "a_ja"), ("u", "ɯ_ja"), ("o", "o_ja"), ("e", "e_ja")):
            rows[prefix + suffix] = [phone, vowel]
    rows.update({"che": ["tɕ_ja", "e_ja"], "ti": ["t_ja", "i_ja"], "tu": ["t_ja", "ɨ_ja"]})
    return rows


def symbol_type(symbol: str) -> str:
    if symbol in VOWELS:
        return "vowel"
    if symbol in NASALS:
        return "nasal"
    if symbol in STOPS:
        return "stop"
    if symbol in AFFRICATES:
        return "affricate"
    if symbol in FRICATIVES:
        return "fricative"
    if symbol in SEMIVOWELS:
        return "semivowel"
    if symbol in LIQUIDS:
        return "liquid"
    if symbol == "c_ja":
        return "stop"
    raise ValueError(f"unclassified phoneme: {symbol}")


def dictionary_payload(phonemes: list[str]) -> dict:
    entries = mora_entries()
    exported = set(phonemes)
    symbols = sorted({phone for phones in entries.values() for phone in phones})
    missing = set(symbols) - exported
    if missing:
        raise ValueError(f"Japanese dictionary tokens missing from acoustic model: {sorted(missing)}")
    return {
        "symbols": [{"symbol": symbol, "type": "vowel" if symbol in {"AP", "SP"}
                     else symbol_type(symbol)} for symbol in symbols],
        "entries": [
            {"grapheme": grapheme, "phonemes": phones}
            for grapheme, phones in sorted(entries.items())
        ],
    }


def copy_tree(output: Path, vocoder: Path | None, acoustic_dir: Path = SOURCE) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    source_config = yaml.safe_load((acoustic_dir / "dsconfig.yaml").read_text())
    for name in ("dsconfig.yaml", source_config["acoustic"], source_config["phonemes"],
                 source_config["languages"]):
        shutil.copy2(acoustic_dir / name, output / name)
    shutil.copytree(SOURCE / "dsdur", output / "dsdur")
    shutil.copytree(SOURCE / "dsvocoder", output / "dsvocoder")
    if vocoder:
        shutil.copy2(vocoder, output / "dsvocoder/nsf_hifigan.onnx")
    config_path = output / "dsconfig.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["unvoiced_phonemes"] = sorted(UNVOICED_PHONEMES)
    config["onset_mode"] = "note_internal"
    # This checkpoint trained the deterministic auxiliary acoustic decoder only
    # (`train_diffusion: false`). Any positive depth injects an untrained
    # stochastic reflow path and destroys intelligibility.
    config["max_depth"] = 0.0
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    embed_voicing_mask(
        output / config["acoustic"],
        output / config["phonemes"],
        UNVOICED_PHONEMES,
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vocoder", type=Path)
    parser.add_argument("--acoustic-dir", type=Path, default=SOURCE)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    copy_tree(output, args.vocoder.resolve() if args.vocoder else None, args.acoustic_dir.resolve())
    config = yaml.safe_load((output / "dsconfig.yaml").read_text())
    phonemes = json.loads((output / config["phonemes"]).read_text())
    dictionary = dictionary_payload(phonemes)
    (output / "dsdur/dsdict-ja.yaml").write_text(
        yaml.safe_dump(dictionary, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "character.txt").write_text(
        "name=DiffSinger JA evaluation candidate (GYU identity unvalidated)\n"
        "author=GYU Singer project / GTSinger contributors\n"
        "version=unreleased-evaluation\n", encoding="utf-8")
    (output / "character.yaml").write_text(yaml.safe_dump({
        "name": "DiffSinger JA evaluation candidate (GYU identity unvalidated)",
        "singer_type": "diffsinger",
        "text_file_encoding": "utf-8",
        "default_phonemizer": "OpenUtau.Core.DiffSinger.DiffSingerJapanesePhonemizer",
        "author": "GYU Singer project / GTSinger contributors",
        "version": "unreleased-evaluation",
        "subbanks": [{"color": "", "prefix": "", "suffix": "", "tone_ranges": ["C2-C6"]}],
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "README.md").write_text(
        "# DiffSinger Japanese evaluation candidate\n\n"
        "This is an unreleased OpenUtau integration candidate and is not a validated GYU voice. "
        "The acoustic and duration foundations are derived from GTSinger. Human listening "
        "approval is required before any release claim.\n\n"
        "License: CC BY-NC-SA 4.0. Commercial use is not permitted.\n",
        encoding="utf-8")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "status": "evaluation_only",
        "release_ready": False,
        "gyu_identity_status": "unvalidated",
        "license": "CC BY-NC-SA 4.0",
        "source": str(args.acoustic_dir.resolve()),
        "diffusion_depth": 0.0,
        "diffusion_disabled_reason": "checkpoint trained auxiliary acoustic decoder only",
        "stock_openutau_compatible_voicing_mask": True,
        "voicing_mask_location": "acoustic ONNX graph",
        "files": {str(path.relative_to(output)): digest(path) for path in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if args.zip:
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    handle.write(path, Path(output.name) / path.relative_to(output))
        print(json.dumps({"output": str(output), "archive": str(archive), "sha256": digest(archive)}))
    else:
        print(json.dumps({"output": str(output), "files": len(files) + 1}))


if __name__ == "__main__":
    main()
