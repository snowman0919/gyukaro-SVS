from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path("scripts").resolve()))
from build_diffsinger_gtsinger_heldout_set import build_ds_row  # noqa: E402
from evaluate_diffsinger_pjs_rapid import similarity_summary, voicing_accuracy  # noqa: E402
from summarize_diffsinger_gtsinger_heldout_gate import (  # noqa: E402
    aggregate_candidate_gate,
    seed_stability,
)
from prepare_diffsinger_gtsinger_gyu_identity import (  # noqa: E402
    build_strict_identity_checkpoint,
    restore_shared_token_rows,
)


def test_heldout_ds_row_preserves_manual_timing_and_rmvpe_grid():
    row = {"ph": ["i_ja", "<AP>"], "ph_durs": [.08, .02], "txt": ["い", "<AP>"]}
    result = build_ds_row(row, .10, np.array([261.63] * 5, dtype=np.float32))

    assert result["ph_seq"] == "i_ja AP"
    assert result["ph_dur"] == "0.0800000 0.0200000"
    assert result["text"] == "い"
    assert result["f0_timestep"] == 0.02
    assert len(result["f0_seq"].split()) == 5


def test_identity_summary_keeps_every_reference_and_distribution():
    references = {
        "a.wav": (np.array([1., 0.]), np.array([1., 0.])),
        "b.wav": (np.array([0., 1.]), np.array([0., 1.])),
    }

    result = similarity_summary(references, (np.array([1., 0.]), np.array([1., 0.])))

    assert result["reference_count"] == 2
    assert result["wavlm"]["values"] == {"a.wav": 1.0, "b.wav": 0.0}
    assert result["wavlm"]["mean"] == .5
    assert result["ecapa"]["min"] == 0.0


def test_voicing_accuracy_uses_equal_hop_grid():
    target = np.array([440.0, 0.0, 660.0], dtype=np.float32)
    observed = np.array([440.0, 440.0, 660.0, 0.0], dtype=np.float32)

    assert voicing_accuracy(target, observed) == 2 / 3


def _gate_report(identifier: str, *, failed_label: str | None = None) -> dict:
    reference = {
        "free_asr_similarity": 1.0,
        "waveform_analysis": {"hf_spike_p99_over_median": 100.0},
    }
    rows = []
    for label in ("soprano", "tenor", "mix20"):
        rows.append({
            "label": label,
            "asr_lyric_similarity": 0.7 if label == failed_label else 0.9,
            "pitch_p90_abs_cents": 40.0,
            "gross_error_over_600_cents": 0.0,
            "voicing_accuracy": 0.9,
            "clip_fraction": 0.0,
            "hf_spike_p99_over_median": 100.0,
            "sample_jump_p999": 0.1,
            "identity_similarity": {
                "wavlm": {"mean": 0.6, "values": {"a": 0.6}},
                "ecapa": {"mean": 0.2, "values": {"a": 0.2}},
            },
        })
    return {"id": identifier, "reference_calibration": reference, "rows": rows}


def test_aggregate_gate_rejects_candidate_when_one_of_five_phrases_fails():
    reports = [_gate_report(f"p{index}", failed_label="mix20" if index == 4 else None)
               for index in range(5)]

    result = aggregate_candidate_gate(reports, "mix20", baseline_label="tenor")

    assert result["status"] == "reject"
    assert result["passed_phrases"] == 4
    assert result["failed_phrases"] == ["p4"]


def test_seed_stability_uses_quality_gates_not_identical_sha(tmp_path):
    files = {}
    for phrase in ("a", "b"):
        files[phrase] = {}
        for seed in (7, 21, 42):
            path = tmp_path / f"{phrase}_{seed}.wav"
            path.write_bytes(f"{phrase}-{seed}".encode())
            files[phrase][seed] = path
    reports = {
        phrase: {
            "reference_calibration": {"free_asr_similarity": 1.0},
            "rows": [{"label": f"seed{seed}", "pass": True} for seed in (7, 21, 42)],
        }
        for phrase in files
    }

    result = seed_stability(files, reports)

    assert result["stable"] is True
    assert result["byte_identical"] is False
    assert all(row["all_quality_gates_pass"] for row in result["phrases"].values())


def test_shared_silence_tokens_are_restored_after_identity_training(tmp_path):
    source_dictionary = tmp_path / "source.txt"
    adapted_dictionary = tmp_path / "adapted.txt"
    source_dictionary.write_text("a_ja\ta_ja\n")
    adapted_dictionary.write_text("a_ja\ta_ja\nko_nucleus_0\tko_nucleus_0\n")
    source = tmp_path / "source.ckpt"
    adapted = tmp_path / "adapted.ckpt"
    target = tmp_path / "target.ckpt"
    torch.save({"state_dict": {"model.fs2.txt_embed.weight": torch.tensor(
        [[0.], [1.], [2.], [3.]]
    )}}, source)
    torch.save({"state_dict": {"model.fs2.txt_embed.weight": torch.tensor(
        [[0.], [10.], [20.], [3.], [40.]]
    )}}, adapted)

    report = restore_shared_token_rows(
        source, source_dictionary, adapted, adapted_dictionary, target
    )
    restored = torch.load(target, map_location="cpu", weights_only=False)["state_dict"][
        "model.fs2.txt_embed.weight"
    ]

    assert restored[:, 0].tolist() == [0., 1., 2., 3., 40.]
    assert report["restored_tokens"] == ["AP", "SP"]
    assert report["max_abs_error_after"] == 0.0


def test_strict_identity_checkpoint_copies_only_gyu_rows(tmp_path):
    dictionary = tmp_path / "dictionary.txt"
    dictionary.write_text("a_ja\ta_ja\nko_nucleus_0\tko_nucleus_0\n")
    initial = tmp_path / "initial.ckpt"
    adapted = tmp_path / "adapted.ckpt"
    target = tmp_path / "target.ckpt"
    torch.save({"state_dict": {
        "model.fs2.txt_embed.weight": torch.tensor([[0.], [1.], [2.], [3.], [4.]]),
        "model.fs2.spk_embed.weight": torch.tensor([[5.], [6.]]),
        "model.fs2.stretch_embed.weight": torch.tensor([7.]),
    }}, initial)
    torch.save({"state_dict": {
        "model.fs2.txt_embed.weight": torch.tensor([[9.], [10.], [20.], [30.], [40.]]),
        "model.fs2.spk_embed.weight": torch.tensor([[50.], [60.]]),
        "model.fs2.stretch_embed.weight": torch.tensor([70.]),
    }}, adapted)

    report = build_strict_identity_checkpoint(initial, adapted, dictionary, target)
    state = torch.load(target, map_location="cpu", weights_only=False)["state_dict"]

    assert state["model.fs2.txt_embed.weight"][:, 0].tolist() == [0., 1., 2., 3., 40.]
    assert state["model.fs2.spk_embed.weight"][:, 0].tolist() == [5., 60.]
    assert state["model.fs2.stretch_embed.weight"].tolist() == [7.]
    assert report["copied_text_tokens"] == ["ko_nucleus_0"]
    assert report["unexpected_changed_tensors_in_adapted"] == [
        "model.fs2.stretch_embed.weight"
    ]
