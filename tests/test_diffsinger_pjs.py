from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path("scripts").resolve()))
import prepare_diffsinger_gyu_segments as prepare  # noqa: E402
from evaluate_diffsinger_pjs_rapid import lyric_similarity  # noqa: E402
from gyu_singer.diffsinger_weighting import weighted_frame_l1  # noqa: E402


def test_japanese_vocabulary_remap_uses_vowel_and_onset_priors(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    old_dictionary = tmp_path / "old.txt"
    new_dictionary = tmp_path / "new.txt"
    old_dictionary.write_text(
        "a\ta\ne\te\ni\ti\no\to\nu\tu\nko_onset_g\tko_onset_g\nko_onset_k\tko_onset_k\n"
    )
    new_dictionary.write_text("ja_I\tja_I\nja_k\tja_k\n")

    old_tokens = prepare.dictionary_tokens(old_dictionary)
    embedding = torch.arange((len(old_tokens) + 1) * 3, dtype=torch.float32).reshape(-1, 3)
    source = tmp_path / "source.ckpt"
    target = tmp_path / "target.ckpt"
    torch.save({"state_dict": {"model.fs2.txt_embed.weight": embedding}}, source)

    report = prepare.remap_vocabulary(source, old_dictionary, new_dictionary, target)
    remapped = torch.load(target, map_location="cpu", weights_only=False)["state_dict"][
        "model.fs2.txt_embed.weight"
    ]
    new_tokens = prepare.dictionary_tokens(new_dictionary)
    old_ids = {token: index + 1 for index, token in enumerate(old_tokens)}
    new_ids = {token: index + 1 for index, token in enumerate(new_tokens)}
    vowel_prior = embedding[[old_ids[token] for token in ("a", "e", "i", "o", "u")]].mean(0)
    onset_prior = embedding[[old_ids[token] for token in ("ko_onset_g", "ko_onset_k")]].mean(0)

    assert torch.equal(remapped[new_ids["ja_I"]], vowel_prior)
    assert torch.equal(remapped[new_ids["ja_k"]], onset_prior)
    assert report["new_token_initialization"] == {
        "ja_I": "mean:nonlexical_vowels",
        "ja_k": "mean:ko_onset",
    }
    assert report["shared_embedding_max_abs_error"] == 0


def test_rapid_lyric_gate_accepts_kanji_or_katakana_alias():
    expected = ["息が詰まる" * 4, "イキガツマル" * 4]

    assert lyric_similarity(expected, "イキガツマル" * 4) == 1


def test_consonant_weighted_loss_emphasizes_short_consonant_error():
    prediction = torch.tensor([[[0.0], [1.0]]])
    target = torch.zeros_like(prediction)
    tokens = torch.tensor([[1, 2]])

    loss = weighted_frame_l1(prediction, target, tokens, low_weight_ids={1}, consonant_weight=5)

    assert torch.isclose(loss, torch.tensor(5 / 6))
