import urllib.request
import importlib.util

import numpy as np
import torch

from gyu_singer.alignment import build_phrase_frames
from gyu_singer.frontend import FEATURE_SIZE, phonemize
from gyu_singer.losses import flow_matching_loss, log_pitch_loss, weighted_distillation_loss
from gyu_singer.model import TriSingerModel, grad_norm
from gyu_singer.renderer import build_server
from gyu_singer.score import normalize_score


def test_trilingual_frontend_structural_features():
    ko, en, ja = phonemize("ko", "한글"), phonemize("en", "soft voice"), phonemize("ja", "がっこうん")
    tense, aspirated = phonemize("ko", "까카"), phonemize("ko", "파타")
    assert any(row[2] for row in ko.features)  # Korean coda
    assert any(row[8] for row in tense.features) and any(row[9] for row in aspirated.features)
    assert any(row[3] for row in en.features)  # rule-based stress proxy, inferred
    assert any(row[6] for row in ja.features) and any(row[7] for row in ja.features)
    assert all(ko.word_boundaries[-1:]) and all(en.word_boundaries[-1:])


def test_alignment_assigns_each_note_its_lyric():
    frames = build_phrase_frames(phonemize("ko", "하늘"), [{"pitch": 60, "start": 0, "duration": .5, "lyric": "하"}, {"pitch": 64, "start": .5, "duration": .5, "lyric": "늘"}])
    assert frames.note_index.max() == 1 and frames.boundary.sum() == 2
    assert frames.phoneme_ids[0] != frames.phoneme_ids[7]


def test_blurred_boundary_and_pitch_conditions_change_phrase_condition():
    model, batch = TriSingerModel(dim=32), _batch()
    base, _, _ = model.condition(batch)
    batch["boundary"][:, 3] = 1
    batch["residual"][:, 4:] = 1.5
    shifted, _, _ = model.condition(batch)
    assert not torch.allclose(base, shifted)


def _batch():
    return {"phoneme_ids": torch.ones(1, 6, dtype=torch.long), "language_ids": torch.zeros(1, 6, dtype=torch.long),
            "features": torch.zeros(1, 6, FEATURE_SIZE), "midi": torch.full((1, 6), 60.0), "note_index": torch.zeros(1, 6, dtype=torch.long),
            "note_onset": torch.zeros(1, 6), "note_duration": torch.ones(1, 6), "boundary": torch.zeros(1, 6), "reference_features": torch.zeros(1, 160), "style_preset": torch.zeros(1, dtype=torch.long),
            "style_controls": torch.zeros(1, 5), "f0_hz": torch.full((1, 6), 261.0), "voiced": torch.ones(1, 6), "residual": torch.zeros(1, 6)}


def test_all_hybrid_modules_receive_gradient():
    model, batch = TriSingerModel(dim=32), _batch()
    output = model(torch.randn(1, 6, 768), torch.tensor([.4]), batch)
    loss = output["velocity"].square().mean() + output["acoustic_bias"].square().mean() + output["pitch_log_f0"].mean() + model.distillation_prediction(batch).square().mean()
    loss.backward()
    for name in ("phoneme_encoder", "language_encoder", "score_encoder", "blurred_boundary_encoder", "timbre_encoder", "style_encoder", "pitch_encoder", "conditional_flow_transformer", "singing_decoder"):
        assert grad_norm(getattr(model, name)) > 0, name


def test_teacher_distillation_reaches_timbre_language_and_style_encoders():
    model, batch = TriSingerModel(dim=32), _batch()
    batch["style_preset"][:] = 4
    loss = weighted_distillation_loss(model.distillation_prediction(batch), torch.ones(1, 160), torch.tensor([0.2]))
    loss.backward()
    assert grad_norm(model.timbre_encoder) > 0
    assert grad_norm(model.language_encoder) > 0
    assert grad_norm(model.style_encoder) > 0


def test_losses_use_pitch_mask_and_teacher_trust():
    assert log_pitch_loss(torch.zeros(1, 2), torch.tensor([[1.0, 100.0]]), torch.tensor([[0.0, 1.0]])) > 0
    assert weighted_distillation_loss(torch.tensor([[0.0], [10.0]]), torch.zeros(2, 1), torch.tensor([1.0, 0.0])) == 0
    assert flow_matching_loss(torch.ones(1, 2, 3), torch.zeros(1, 2, 3)) == 1


def test_score_protocol_and_resident_http():
    score = normalize_score({"language": "ko", "tempo": 120, "style": {"preset": "bright"}, "curves": {"pitch": [[0, 0], [1, 2]]}, "notes": [{"id": "n1", "pitch": 60, "start_beat": 0, "duration_beats": 1, "lyric": "아"}]})
    assert score["sample_rate"] == 48000
    assert score["notes"][0]["duration"] == .5 and score["curves"]["pitch"][1]["time"] == .5
    class Fake:
        sample_rate = 48000
        def render(self, incoming):
            assert incoming["language"] == "ko"; return np.zeros(9600, np.float32)
    server = build_server(Fake(), port=0)
    import threading
    thread = threading.Thread(target=server.serve_forever); thread.start()
    try:
        host, port = server.server_address
        assert b'"status": "ok"' in urllib.request.urlopen(f"http://{host}:{port}/health").read()
        request = urllib.request.Request(f"http://{host}:{port}/render", data=b'{"language":"ko","notes":[{"pitch":60,"start":0,"duration":0.2,"lyric":"\xec\x95\x84"}]}', method="POST")
        assert urllib.request.urlopen(request).read()[:4] == b"RIFF"
    finally:
        server.shutdown(); thread.join()


def test_hybrid_path_has_no_baseline_dsp_calls():
    source = open("src/gyu_singer/inference/hybrid.py").read()
    assert "pitch_shift" not in source and "phase_vocoder" not in source and "NeuralRenderer" not in source


def test_phrase_flow_uses_all_notes_in_one_tensor():
    model, batch = TriSingerModel(dim=32), _batch()
    batch["note_index"] = torch.tensor([[0, 0, 0, 1, 1, 1]])
    latent = model.sample(batch, steps=2)
    assert latent.shape == (1, 6, 768)


def test_sample_integrates_decoder_bias_with_training_scale():
    model, batch = TriSingerModel(dim=32), _batch()
    model.forward = lambda latent, time, batch: {"velocity": torch.zeros_like(latent), "acoustic_bias": torch.ones_like(latent)}
    torch.manual_seed(7); expected = torch.randn(1, 6, 768) + .1
    torch.manual_seed(7); assert torch.allclose(model.sample(batch, steps=2), expected, atol=1e-5)


def test_openutau_ustx_bridge_converts_ticks(tmp_path):
    bridge_path = "integrations/openutau/bridge.py"
    spec = importlib.util.spec_from_file_location("ustx_bridge", bridge_path)
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    source = tmp_path / "song.ustx"
    source.write_text("resolution: 480\ntempos:\n  - bpm: 120\nvoice_parts:\n  - position: 480\n    notes:\n      - position: 0\n        duration: 480\n        tone: 60\n        lyric: 아\n")
    score = bridge.ustx_score(source, "ko")
    assert score["notes"] == [{"pitch": 60.0, "start": 0.5, "duration": 0.5, "lyric": "아"}]
