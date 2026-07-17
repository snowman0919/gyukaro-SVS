from pathlib import Path
import importlib.util
import json

import onnx
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "package_openutau_diffsinger_candidate",
    ROOT / "scripts/package_openutau_diffsinger_candidate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dictionary_covers_reference_song_morae():
    entries = MODULE.mora_entries()
    expected = {
        "a", "i", "u", "e", "o", "ka", "ga", "ki", "kyu", "kyo", "gyo", "ku",
        "gu", "ke", "ge", "ko", "go", "sa", "za", "shi", "sha", "sho", "ji", "ja",
        "ju", "jo", "su", "zu", "se", "so", "zo", "ta", "da", "chi", "che", "chu",
        "cho", "tsu", "te", "de", "di", "to", "tu", "do", "na", "ni", "nu", "ne",
        "no", "ha", "ba", "hi", "hyo", "bi", "pi", "fu", "bu", "pu", "be", "ho",
        "bo", "po", "ma", "mi", "mu", "me", "mo", "yu", "yo", "ra", "ri", "ryo",
        "ru", "re", "ro", "wa", "wo", "n", "vu",
    }
    assert expected <= entries.keys()


def test_dictionary_uses_only_exported_tokens():
    tokens = json.loads((
        ROOT / "data/external/work/openutau_gyu_diffsinger_candidate/"
        "gtsinger_ja_tenor.phonemes.json"
    ).read_text())
    used = {phone for phones in MODULE.mora_entries().values() for phone in phones}
    assert used <= set(tokens)
    assert all(MODULE.symbol_type(phone) for phone in tokens if phone not in {"AP", "SP"})


def test_evaluation_package_disables_untrained_diffusion_and_embeds_voicing_mask():
    package = (
        ROOT / "data/external/work/openutau_native_candidate/"
        "DiffSinger-JA-source15000-depth0-final-eval"
    )
    config = yaml.safe_load((package / "dsconfig.yaml").read_text())
    assert config["max_depth"] == 0.0
    model = onnx.load(package / config["acoustic"])
    metadata = {item.key: item.value for item in model.metadata_props}
    assert "token-derived zero-F0 mask" in metadata["gyu.voicing_mask"]
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["release_ready"] is False
    assert manifest["gyu_identity_status"] == "unvalidated"
