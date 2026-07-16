from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path("scripts").resolve()))
import prepare_diffsinger_gyu_segments as prepare  # noqa: E402
from evaluate_diffsinger_pjs_rapid import lyric_similarity, passes_gate, pitch_errors  # noqa: E402
from prepare_diffsinger_common_voice_ja import (  # noqa: E402
    ctc_symbols,
    phone_durations,
    resize_speaker_embedding,
)
from build_diffsinger_rapid_chunks import split_row  # noqa: E402
from diffsinger_neutral_augmentation_binarizer import (  # noqa: E402
    apply_phoneme_voicing,
    neutralize_controls,
)
from prepare_diffsinger_pjs_paired_speech import source_phones  # noqa: E402
from build_diffsinger_rapid_voicing import mask_f0  # noqa: E402
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


def test_japanese_vocabulary_remap_falls_back_to_existing_japanese_consonants(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    old_dictionary = tmp_path / "old.txt"
    new_dictionary = tmp_path / "new.txt"
    old_dictionary.write_text("ja_a\tja_a\nja_k\tja_k\n")
    new_dictionary.write_text("ja_a\tja_a\nja_k\tja_k\nja_v\tja_v\n")
    old_tokens = prepare.dictionary_tokens(old_dictionary)
    embedding = torch.arange((len(old_tokens) + 1) * 2, dtype=torch.float32).reshape(-1, 2)
    source, target = tmp_path / "source.ckpt", tmp_path / "target.ckpt"
    torch.save({"state_dict": {"model.fs2.txt_embed.weight": embedding}}, source)

    report = prepare.remap_vocabulary(source, old_dictionary, new_dictionary, target)

    assert report["new_token_initialization"]["ja_v"] == "mean:ja_consonant"
    assert report["shared_embedding_max_abs_error"] == 0


def test_rapid_lyric_gate_accepts_kanji_or_katakana_alias():
    expected = ["息が詰まる" * 4, "イキガツマル" * 4]

    assert lyric_similarity(expected, "イキガツマル" * 4) == 1


def test_invalid_free_asr_gate_uses_reference_calibrated_nll():
    row = {"asr_lyric_similarity": 0.1, "teacher_forced_lyric_nll": 2.0,
           "pitch_p90_abs_cents": 50, "gross_error_over_600_cents": 0,
           "observed_voiced_ratio": 0.9, "clip_fraction": 0}

    assert passes_gate(row, asr_gate_valid=False, reference_nll=2.1)
    assert not passes_gate(row | {"teacher_forced_lyric_nll": 2.3},
                           asr_gate_valid=False, reference_nll=2.1)


def test_equal_hop_pitch_gate_crops_one_tail_frame_without_interpolating_unvoiced_boundaries():
    target = torch.tensor([440.0, 0.0, 660.0]).numpy()
    observed = torch.tensor([440.0, 0.0, 660.0, 0.0]).numpy()

    assert pitch_errors(target, observed).tolist() == [0.0, 0.0]


def test_equal_hop_pitch_gate_rejects_material_timing_drift():
    target = torch.ones(3).numpy()
    observed = torch.ones(5).numpy()

    try:
        pitch_errors(target, observed)
    except ValueError as error:
        assert "length mismatch" in str(error)
    else:
        raise AssertionError("material timing drift must fail instead of being time-warped")


def test_consonant_weighted_loss_emphasizes_short_consonant_error():
    prediction = torch.tensor([[[0.0], [1.0]]])
    target = torch.zeros_like(prediction)
    tokens = torch.tensor([[1, 2]])

    loss = weighted_frame_l1(prediction, target, tokens, low_weight_ids={1}, consonant_weight=5)

    assert torch.isclose(loss, torch.tensor(5 / 6))


def test_common_voice_ctc_spans_preserve_phone_order_and_duration():
    phones = ["ky", "a", "cl", "t", "e"]
    spans = [(1, 2), (2, 3), (4, 6), (7, 8), (9, 10), (11, 13)]

    symbols, durations = phone_durations(phones, spans, frames=20, duration=2.0)

    assert ctc_symbols("cl") == "q"
    assert symbols == ["ja_ky", "ja_a", "ja_cl", "ja_t", "ja_e"]
    assert all(value > 0 for value in durations)
    assert abs(sum(durations) - 2.0) < 1e-9


def test_common_voice_uses_a_distinct_initialized_speaker_row(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    original = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    torch.save({"state_dict": {"model.fs2.spk_embed.weight": original}}, checkpoint)

    report = resize_speaker_embedding(checkpoint, 2)
    resized = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"][
        "model.fs2.spk_embed.weight"
    ]

    assert resized.shape == (2, 6)
    assert torch.equal(resized[0], original[0])
    assert torch.equal(resized[1], original[0])
    assert report["preserved_row_max_abs_error"] == 0


def test_rapid_chunks_preserve_all_phones_and_condition_frames():
    row = {"ph_seq": " ".join(f"p{i}" for i in range(44)),
           "ph_dur": " ".join(["0.04"] * 44),
           "f0_seq": " ".join(map(str, range(88))),
           "velocity": " ".join(["1"] * 88)}

    chunks = split_row(row, repeats_per_chunk=2)

    assert [len(chunk["ph_seq"].split()) for chunk in chunks] == [22, 22]
    assert sum(len(chunk["f0_seq"].split()) for chunk in chunks) == 88


def test_pitch_and_rate_augmentation_do_not_select_a_different_voice():
    item = {"key_shift": 6.0, "speed": 3.5, "mel": torch.ones(2, 3)}

    result = neutralize_controls(item)

    assert result["key_shift"] == 0
    assert result["speed"] == 1
    assert torch.equal(result["mel"], torch.ones(2, 3))


def test_training_f0_uses_the_same_phoneme_voicing_mask_as_inference():
    item = {"ph_text": "ja_i ja_k ja_a", "mel2ph": [1, 2, 3],
            "f0": torch.tensor([440.0, 440.0, 440.0])}

    result = apply_phoneme_voicing(item)

    assert torch.equal(result["f0"], torch.tensor([440.0, 0.0, 440.0]))

    ipa = {"ph_text": "i_ja i̥_ja a_ja", "mel2ph": [1, 2, 3],
           "f0": torch.tensor([440.0, 440.0, 440.0])}
    assert torch.equal(apply_phoneme_voicing(ipa)["f0"], torch.tensor([440.0, 0.0, 440.0]))


def test_pjs_speech_reuses_only_the_official_phone_sequence():
    phones = source_phones(["0 1000000 pau", "1000000 2000000 k",
                            "2000000 3000000 a", "3000000 4000000 xx"])

    assert phones == ["k", "a"]


def test_rapid_score_zeros_f0_only_in_unvoiced_phone_frames():
    values = mask_f0(["ja_i", "ja_k", "ja_a"], [0.02, 0.02, 0.02],
                     [440, 440, 440], 0.02)

    assert values == [440, 0, 440]


def test_gtsinger_japanese_selection_and_phone_normalization():
    from prepare_diffsinger_gtsinger_ja import heldout_names, normalized_phones, selected_rows

    rows = [
        {"language": "Japanese", "singer": "JA-Soprano-1", "item_name": "Japanese#JA-Soprano-1#x#song#Control_Group#0", "ph": ["<SP>", "i_ja", "k_ja", "<AP>"]},
        {"language": "Japanese", "singer": "JA-Tenor-1", "item_name": "Japanese#JA-Tenor-1#x#song#Control_Group#0", "ph": ["a_ja"]},
        {"language": "Korean", "singer": "JA-Soprano-1", "item_name": "Korean#JA-Soprano-1#x#song#Control_Group#0", "ph": ["a_ko"]},
        {"language": "Japanese", "singer": "JA-Soprano-1", "item_name": "Japanese#JA-Soprano-1#x#song#Breathy_Group#0", "ph": ["a_ja"]},
    ]
    assert selected_rows(rows) == rows[:1]
    assert normalized_phones(rows[0]) == ["SP", "i_ja", "k_ja", "AP"]
    split_rows = [
        {"item_name": "Japanese#JA-Soprano-1#Breathy#song-a#Control#0000"},
        {"item_name": "Japanese#JA-Soprano-1#Breathy#song-b#Control#0000"},
        {"item_name": "Japanese#JA-Soprano-1#Breathy#song-b#Breathy#0000"},
    ]
    assert heldout_names(split_rows, song_count=1) == ["gtsja0001", "gtsja0002"]


def test_gtsinger_rapid_translation_uses_native_ipa():
    from build_diffsinger_gtsinger_rapid import translate

    assert translate("ja_i ja_k ja_i ja_g ja_a ja_ts ja_u ja_m ja_a ja_r ja_u") == (
        "i_ja k_ja i_ja ɡ_ja a_ja ts_ja ɯ_ja m_ja a_ja ɾ_ja ɯ_ja"
    )
