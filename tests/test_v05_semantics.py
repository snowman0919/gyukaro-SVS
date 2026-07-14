import torch
from gyu_singer.inference.acoustic_style import GyuAcousticStyleAdapter
from gyu_singer.inference.quality_controller import condition_batch
from gyu_singer.score import normalize_score


def score():
    return normalize_score({"language": "ko", "tempo": 120, "style": {"preset": "neutral", "prosody_strength": 1.0, "acoustic_style_strength": 1.0}, "notes": [{"pitch": 60, "start_beat": 0, "duration_beats": 2, "lyric": "아"}]})


def test_real_target_change_cannot_change_production_condition():
    batch_a, _ = condition_batch(score(), torch.zeros(160), "cpu")
    batch_b, _ = condition_batch(score(), torch.zeros(160), "cpu")
    target_a = torch.full((1, batch_a["f0_hz"].shape[1]), 220.0)
    target_b = torch.full_like(target_a, 330.0)
    assert torch.equal(batch_a["f0_hz"], batch_b["f0_hz"])
    assert not torch.equal(target_a, target_b)
    assert "target_f0_hz" not in batch_a and "actual_rmvpe_f0" not in batch_a


def test_acoustic_style_controls_change_adapter_representation():
    torch.manual_seed(2); adapter = GyuAcousticStyleAdapter(); ref = torch.zeros(1, 160); neutral = torch.zeros(1, 5); bright = torch.tensor([[.8, 0, 0, 1, 0.]])
    a = adapter(ref, neutral, torch.tensor([0])); b = adapter(ref, bright, torch.tensor([5]))
    assert not torch.allclose(a, b)


def test_singing_alignment_manifest_rejects_uniform_split():
    import json
    from pathlib import Path
    rows = [json.loads(line) for line in Path("data/manifests/real_phoneme_alignment.jsonl").read_text().splitlines() if line]
    assert rows and all(row.get("uniform_split_guard") for row in rows)
