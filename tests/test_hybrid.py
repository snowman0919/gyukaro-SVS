import json
import urllib.error
import urllib.request
import importlib.util

import numpy as np
import pytest
import torch

from gyu_singer.alignment import build_phrase_frames
from gyu_singer.frontend import FEATURE_SIZE, phonemize
from gyu_singer.inference.soulx import SoulXPhraseRenderer, _Worker
from gyu_singer.inference.content_timing import CTCAlignmentUnavailable
from gyu_singer.inference.quality_controller import condition_batch
from gyu_singer.inference.v08 import GyuSingerV08Renderer
from gyu_singer.inference.v09 import GyuSingerV09Renderer, soften_large_jumps
from gyu_singer.inference.rc9 import GyuSingerRC9Renderer
from gyu_singer.losses import flow_matching_loss, log_pitch_loss, weighted_distillation_loss
from gyu_singer.model import TriSingerModel, grad_norm
from gyu_singer.renderer import build_server
from gyu_singer.score import normalize_score


def test_trilingual_frontend_structural_features():
    ko, en, ja = phonemize("ko", "한글"), phonemize("en", "soft voice"), phonemize("ja", "がっこうん")
    tense, aspirated = phonemize("ko", "까카"), phonemize("ko", "파타")
    assert any(row[2] for row in ko.features)  # Korean coda
    assert any(row[8] for row in tense.features) and any(row[9] for row in aspirated.features)
    assert any(row[3] == 1 for row in en.features) and not any(en.inferred)  # CMU lexical stress
    assert "en_aa" in en.symbols and "en_oy" in en.symbols
    unknown = phonemize("en", "thing")
    assert all(unknown.inferred) and unknown.symbols == ["en_th", "en_ih", "en_ng"]
    assert phonemize("ja", "光の向こうへ").symbols == ["ja_h", "ja_i", "ja_k", "ja_a", "ja_r", "ja_i", "ja_n", "ja_o", "ja_m", "ja_u", "ja_k", "ja_o", "ja_u", "ja_h", "ja_e"]
    assert not any("unknown" in symbol for text in ("空へ向かい", "歌おう", "小さな光を", "追う") for symbol in phonemize("ja", text).symbols)
    katakana = phonemize("ja", "ノンブレス・オブリージュ")
    assert not any("unknown" in symbol for symbol in katakana.symbols)
    mixed = phonemize("ja", "I love you息")
    assert any(symbol.startswith("en_") for symbol in mixed.symbols)
    unknown = mixed.symbols.index("ja_unknown_息")
    assert mixed.inferred[unknown] and mixed.features[unknown][1] == 1
    assert any(row[6] for row in ja.features) and any(row[7] for row in ja.features)
    assert all(ko.word_boundaries[-1:]) and all(en.word_boundaries[-1:])


def test_alignment_assigns_each_note_its_lyric():
    frames = build_phrase_frames(phonemize("ko", "하늘"), [{"pitch": 60, "start": 0, "duration": .5, "lyric": "하"}, {"pitch": 64, "start": .5, "duration": .5, "lyric": "늘"}])
    assert frames.note_index.max() == 1 and frames.boundary.sum() == 2
    assert frames.phoneme_ids[0] != frames.phoneme_ids[7]
    assert frames.phoneme_note_mapping.tolist() == frames.note_index.tolist()
    assert len(frames.phoneme_durations) == 5 and [note["boundary_type"] for note in frames.note_sequence] == ["hard", "hard"]


def test_alignment_uses_slur_to_soften_note_boundary():
    frames = build_phrase_frames(phonemize("ko", "하늘"), [{"pitch": 60, "start": 0, "duration": .5, "lyric": "하"}, {"pitch": 64, "start": .5, "duration": .5, "lyric": "늘", "slur": True}])
    assert frames.boundary[0] == 1 and frames.boundary[6] == 0


def test_ctc_alignment_overrides_even_phoneme_division():
    front = phonemize("ko", "하늘")
    aligned = [{"phoneme_index": 0, "start": 0.0, "duration": .08}, {"phoneme_index": 1, "start": .08, "duration": .56}, {"phoneme_index": 2, "start": .64, "duration": .08}, {"phoneme_index": 3, "start": .72, "duration": .08}, {"phoneme_index": 4, "start": .80, "duration": .56}]
    frames = build_phrase_frames(front, [{"pitch": 60, "start": 0, "duration": .8, "lyric": "하"}, {"pitch": 64, "start": .8, "duration": .8, "lyric": "늘"}], phoneme_alignment=aligned)
    assert frames.phoneme_durations[1]["duration_frames"] > frames.phoneme_durations[0]["duration_frames"]
    assert frames.phoneme_durations[0]["boundary_type"] == "ctc_forced"


def test_canonical_timeline_zeros_unvoiced_phones_and_silence():
    notes = [{"pitch": 60, "start": .1, "duration": .5, "lyric": "파"}]
    frames = build_phrase_frames(phonemize("ko", "파"), notes, frame_hz=50)
    assert frames.voicing_classes[0] == "silence" and frames.f0_hz[0] == 0
    assert "unvoiced_consonant" in frames.voicing_classes and "vowel" in frames.voicing_classes
    unvoiced = torch.tensor([kind in {"silence", "unvoiced_consonant"} for kind in frames.voicing_classes])
    assert torch.all(frames.f0_hz[unvoiced] == 0) and torch.all(frames.f0_hz[frames.voiced.bool()] > 0)


def test_english_ay_diphthong_remains_voiced():
    frames = build_phrase_frames(
        phonemize("en", "light"),
        [{"pitch": 60, "start": 0, "duration": 1, "lyric": "light"}],
        frame_hz=50,
    )
    ay = next(row for row in frames.phoneme_durations if row["symbol"] == "en_ay")
    region = slice(ay["start_frame"], ay["start_frame"] + ay["duration_frames"])
    assert set(frames.voicing_classes[region]) == {"vowel"}
    assert torch.all(frames.f0_hz[region] > 0)


def test_openutau_phone_window_drives_inferred_frontend_split():
    notes = [{"pitch": 60, "start": 0, "duration": .6, "lyric": "하"}]
    frames = build_phrase_frames(phonemize("ko", "하"), notes, frame_hz=50, phoneme_alignment=[{"phoneme": "ha", "start": .04, "duration": .5}])
    assert frames.voicing_classes[0] == "silence"
    assert any(row["boundary_type"] == "openutau_timed_inferred_split" for row in frames.phoneme_durations)


def test_korean_obstruent_coda_is_unvoiced():
    frames = build_phrase_frames(
        phonemize("ko", "높"),
        [{"pitch": 60, "start": 0, "duration": 1.0, "lyric": "높"}],
        frame_hz=50,
    )
    assert frames.voicing_classes[-1] == "unvoiced_consonant"
    assert frames.f0_hz[-1] == 0


def test_rc4_legacy_condition_remains_explicitly_callable():
    score = {"language": "ko", "notes": [{"pitch": 60, "start": 0, "duration": .5, "lyric": "파"}], "curves": {name: [] for name in ("pitch", "dynamics", "breathiness", "tension", "brightness", "vibrato")}, "style": {"preset": "neutral"}}
    legacy, _ = condition_batch(score, torch.zeros(160), "cpu")
    canonical, _ = condition_batch(score, torch.zeros(160), "cpu", canonical_timing=True)
    assert torch.all(legacy["voiced"] == 1) and torch.any(canonical["voiced"] == 0)


def test_rc5_decode_policy_matches_human_reviewed_stress_set():
    renderer = GyuSingerV09Renderer.__new__(GyuSingerV09Renderer)
    base = {"language": "ko", "style": {"preset": "neutral"}, "notes": [{"pitch": 60, "start": 0, "duration": 1}]}
    assert renderer._decoder_options(base) == {"n_steps": 32, "cfg": 1.5, "seed": 21}
    assert renderer._decoder_options(base | {"notes": [{"pitch": 60, "start": 0, "duration": .25}]})["n_steps"] == 64
    assert renderer._decoder_options(base | {"notes": [{"pitch": 60, "start": 0, "duration": 1}, {"pitch": 72, "start": 1, "duration": 1}]}) == {"n_steps": 50, "cfg": 2.0, "seed": 21}
    assert renderer._content_warp_strength(base) == .05
    assert renderer._content_warp_strength(base | {"language": "en"}) == .25
    assert renderer._content_warp_strength(base | {"notes": [{"pitch": 60, "start": 0, "duration": .25}]}) == 1
    assert renderer._content_warp_strength(base | {"notes": [{"pitch": 60, "start": 0, "duration": 1}, {"pitch": 72, "start": 1, "duration": 1}]}) == 0
    assert renderer._content_warp_strength(base | {"style": {"preset": "breathy"}}) == 0
    assert renderer._content_warp_strength(base | {"notes": [{"pitch": 60, "start": 0, "duration": 1}, {"pitch": 62, "start": 2, "duration": 1}]}) == 0


def test_rc9_personalized_pitch_residual_is_korean_only():
    renderer = GyuSingerRC9Renderer.__new__(GyuSingerRC9Renderer)
    renderer.pitch_controller = type("Controller", (), {"predict": lambda self, score, canonical_timing: (torch.ones(4),)})()
    assert torch.equal(renderer._predict_pitch({"language": "ko"}), torch.ones(4))
    assert torch.equal(renderer._predict_pitch({"language": "ja"}), torch.zeros(4))
    assert torch.equal(renderer._predict_pitch({"language": "en"}), torch.zeros(4))
    rapid_ja = {"language": "ja", "notes": [{"pitch": 60, "start": 0, "duration": .25}]}
    assert renderer._content_warp_strength(rapid_ja) == 0


def test_rc9_chunks_only_long_jump_heavy_japanese_repetitions():
    score = {
        "language": "ja",
        "notes": [
            {"pitch": 74 if index % 2 == 0 else 60, "start": index * 1.2, "duration": 1.2, "lyric": "息が詰まる"}
            for index in range(5)
        ],
    }
    chunks = GyuSingerRC9Renderer._semantic_content_chunks(score)
    assert [lyrics for lyrics, _ in chunks] == ["息が詰まる"] * 5
    assert [duration for _, duration in chunks] == pytest.approx([1.2] * 5)
    embedded = score | {"notes": [
        {"pitch": 60, "start": 0, "duration": 1, "lyric": "前"},
        *[note | {"start": note["start"] + 1} for note in score["notes"]],
        {"pitch": 74, "start": 7, "duration": 1, "lyric": "後"},
    ]}
    assert "".join(lyrics for lyrics, _ in GyuSingerRC9Renderer._semantic_content_chunks(embedded)) == "前" + "息が詰まる" * 5 + "後"
    corrected = GyuSingerRC9Renderer._score_for_voicing(score)
    assert corrected["notes"][0]["lyric"] == "いきがつまる"
    assert GyuSingerRC9Renderer._semantic_content_chunks(score | {"language": "ko"}) == []
    assert GyuSingerRC9Renderer._bypass_post_refiners(score)
    assert not GyuSingerRC9Renderer._bypass_post_refiners(score | {"language": "ko"})
    assert not GyuSingerRC9Renderer._bypass_post_refiners({
        "language": "ja", "notes": [{"pitch": 64}, {"pitch": 69}],
    })
    high_stepwise = {
        "language": "ja",
        "notes": [{"pitch": 72 + index % 3, "start": index * .25, "duration": .25} for index in range(12)],
    }
    assert GyuSingerRC9Renderer._needs_high_rapid_onset_relief(high_stepwise)
    assert not GyuSingerRC9Renderer._needs_high_rapid_onset_relief(high_stepwise | {
        "notes": high_stepwise["notes"][:6],
    })


def test_rc9_long_phrase_context_chunks_preserve_duration_and_no_zero_join():
    score = normalize_score({
        "language": "ja", "tempo": 120, "sample_rate": 48000,
        "notes": [{"pitch": 72, "start": index * .5, "duration": .5, "lyric": "あ"} for index in range(48)],
        "curves": {"pitch": [{"time": 0, "value": 0}, {"time": 24, "value": 0}]},
        "style": {"preset": "neutral"},
    })
    chunks = GyuSingerRC9Renderer._contextual_subscores(score)
    audio = [np.ones(round((chunk["context_end"] - chunk["context_start"]) * 48000), dtype="float32") for chunk in chunks]
    stitched = GyuSingerRC9Renderer._stitch_contextual(chunks, audio)
    assert len(chunks) == 3 and len(stitched) == 24 * 48000
    assert np.min(stitched) > .9


def test_rc5_skips_only_infeasible_optional_ctc_warp(monkeypatch, tmp_path):
    renderer = GyuSingerV09Renderer.__new__(GyuSingerV09Renderer)
    renderer._ctc = (object(), ("-",))
    content = tmp_path / "content.wav"
    import soundfile as sf
    sf.write(content, np.zeros(1600, np.float32), 16_000)
    monkeypatch.setattr(
        "gyu_singer.inference.v09.ctc_phone_alignment",
        lambda *args: (_ for _ in ()).throw(CTCAlignmentUnavailable("too dense")),
    )
    score = {
        "language": "ja", "style": {"preset": "neutral"},
        "notes": [{"pitch": 60, "start": 0, "duration": .2, "lyric": "あ"}],
    }
    assert renderer._content_options(score, content, np.ones(10), tmp_path) == {}


def test_large_jump_transition_is_local():
    f0 = np.array([200, 200, 0, 400, 400, 300, 300], dtype="float32")
    score = {"notes": [
        {"pitch": 55, "start": 0, "duration": .06},
        {"pitch": 67, "start": .06, "duration": .04},
        {"pitch": 62, "start": .1, "duration": .04},
    ]}
    softened = soften_large_jumps(f0, score, .04)
    assert 200 < softened[3] < 400 and np.isclose(softened[4], 400)
    assert np.array_equal(softened[5:], f0[5:])


def test_large_jump_transition_preserves_authoritative_user_pitch(monkeypatch):
    renderer = GyuSingerV09Renderer.__new__(GyuSingerV09Renderer)
    f0 = np.array([200, 200, 400, 400], dtype="float32")
    timeline = [{"f0_hz": float(value)} for value in f0]
    monkeypatch.setattr(renderer, "_canonical_f0", lambda score, duration, expressive: (f0.copy(), timeline))
    score = {
        "notes": [
            {"pitch": 55, "start": 0, "duration": .04},
            {"pitch": 67, "start": .04, "duration": .04},
        ],
        "curves": {"pitch": [{"time": 0, "value": 0}]},
    }
    target, returned_timeline = renderer._target_f0(score, .08, np.zeros(4, dtype="float32"))
    assert np.array_equal(target, f0)
    assert returned_timeline == timeline


def test_rc5_safety_gain_is_only_applied_above_point_97(monkeypatch):
    renderer = GyuSingerV09Renderer.__new__(GyuSingerV09Renderer)
    monkeypatch.setattr(GyuSingerV08Renderer, "render", lambda self, score: np.array([-.5, 1.0], np.float32))
    assert np.isclose(np.max(np.abs(renderer.render({}))), .97)


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


def test_resident_http_serializes_and_recovers_after_failure():
    import concurrent.futures
    import threading
    import time
    class Fake:
        sample_rate = 48000
        active = peak = 0
        lock = threading.Lock()
        def render(self, incoming):
            if incoming.get("fail"): raise RuntimeError("expected failure")
            with self.lock:
                self.active += 1; self.peak = max(self.peak, self.active)
            time.sleep(.02)
            with self.lock: self.active -= 1
            return np.zeros(960, np.float32)
    fake = Fake(); server = build_server(fake, port=0)
    thread = threading.Thread(target=server.serve_forever); thread.start()
    host, port = server.server_address
    def request(body):
        incoming = urllib.request.Request(f"http://{host}:{port}/render", data=json.dumps(body).encode(), method="POST")
        return urllib.request.urlopen(incoming).read()
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            assert all(body[:4] == b"RIFF" for body in pool.map(request, [{}, {}]))
        assert fake.peak == 1
        with pytest.raises(urllib.error.HTTPError) as error: request({"fail": True})
        assert error.value.code == 500 and request({})[:4] == b"RIFF"
    finally:
        server.shutdown(); thread.join()


def test_hybrid_path_has_no_baseline_dsp_calls():
    source = open("src/gyu_singer/inference/hybrid.py").read()
    assert "pitch_shift" not in source and "phase_vocoder" not in source and "NeuralRenderer" not in source


def test_soulx_phrase_backend_builds_one_score_contour():
    score = normalize_score({"language": "ja", "tempo": 120, "notes": [{"pitch": 60, "start": 0, "duration": 1, "lyric": "あ"}, {"pitch": 67, "start": 1, "duration": 1, "lyric": "い"}], "curves": {"pitch": [{"time": 0, "value": 0}, {"time": 1, "value": 0}]}})
    contour = SoulXPhraseRenderer._f0(score, 2)
    expressive = SoulXPhraseRenderer._f0(score, 2, np.full(25, .1))
    assert contour.shape == (100,) and np.isclose(np.median(contour[:50]), 261.6256, atol=1) and np.isclose(np.median(contour[50:]), 391.9954, atol=1)
    assert np.median(expressive[:50]) > np.median(contour[:50])


def test_quality_worker_waits_for_tagged_response():
    class Sink:
        value = ""
        def write(self, value): self.value += value
        def flush(self): pass
    class Process:
        stdin, stdout = Sink(), iter(["startup log\n", '__GYU_RESULT__ {"output":"x"}\n'])
    worker = _Worker.__new__(_Worker); worker.process = Process()
    worker.request({"output": "x"})
    assert '"output": "x"' in worker.process.stdin.value


def test_phrase_flow_uses_all_notes_in_one_tensor():
    model, batch = TriSingerModel(dim=32), _batch()
    batch["note_index"] = torch.tensor([[0, 0, 0, 1, 1, 1]])
    latent = model.sample(batch, steps=2)
    assert latent.shape == (1, 6, 768)


def test_sample_starts_from_decoder_source_then_integrates_flow():
    model, batch = TriSingerModel(dim=32), _batch()
    model.acoustic_source = lambda batch: torch.ones(1, 6, 768)
    model.forward = lambda latent, time, batch: {"velocity": torch.full_like(latent, 2.0)}
    assert torch.allclose(model.sample(batch, steps=2), torch.full((1, 6, 768), 3.0))


def test_openutau_ustx_bridge_converts_ticks(tmp_path):
    bridge_path = "integrations/openutau/bridge.py"
    spec = importlib.util.spec_from_file_location("ustx_bridge", bridge_path)
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    source = tmp_path / "song.ustx"
    source.write_text("resolution: 480\ntempos:\n  - bpm: 120\nvoice_parts:\n  - position: 480\n    notes:\n      - position: 0\n        duration: 480\n        tone: 60\n        lyric: 아\n")
    score = bridge.ustx_score(source, "ko")
    assert score["notes"] == [{"pitch": 60.0, "start": 0.0, "duration": 0.5, "lyric": "아"}]


def test_openutau_bridge_maps_tempo_pitch_and_style(tmp_path):
    bridge_path = "integrations/openutau/bridge.py"
    spec = importlib.util.spec_from_file_location("ustx_bridge_controls", bridge_path)
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    source = tmp_path / "controls.ustx"
    source.write_text("""resolution: 480
tempos: [{position: 0, bpm: 120}, {position: 480, bpm: 60}]
voice_parts:
- position: 0
  notes:
  - {position: 0, duration: 960, tone: 60, tuning: 25, lyric: sing}
  curves:
  - {abbr: pitd, xs: [0, 960], ys: [0, 100]}
  - {abbr: gyus, xs: [0, 960], ys: [3, 3]}
""")
    score = bridge.ustx_score(source, "en")
    assert score["notes"][0] == {"pitch": 60.25, "start": 0.0, "duration": 1.5, "lyric": "sing"}
    assert score["curves"]["pitch"][-1] == {"time": 1.5, "value": 1.0}
    assert score["style"]["preset"] == "energetic"
