#!/usr/bin/env python3
"""Bounded JA content-source replacement diagnostic; never changes RC8."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT if (ROOT / ".venv-soulx").is_dir() else ROOT.parents[1]
CACHE = RUNTIME_ROOT / "data/cache"
TIMESTEP = 0.02
UNVOICED = {
    "c_ja", "h_ja", "k_ja", "p_ja", "s_ja", "t_ja", "ts_ja",
    "tɕ_ja", "ç_ja", "ɕ_ja", "ɸ_ja", "ʔ_ja", "i̥_ja", "ɨ̥_ja", "ɯ̥_ja",
}
PHONE_MAP = {
    "a": "a_ja", "i": "i_ja", "I": "i̥_ja", "u": "ɯ_ja", "U": "ɯ̥_ja",
    "e": "e_ja", "o": "o_ja", "N": "ɴ_ja", "cl": "ʔ_ja", "pau": "SP",
    "b": "b_ja", "by": "bʲ_ja", "ch": "tɕ_ja", "d": "d_ja", "f": "ɸ_ja",
    "g": "ɡ_ja", "gy": "ɟ_ja", "h": "h_ja", "hy": "ç_ja", "j": "dʑ_ja",
    "k": "k_ja", "ky": "c_ja", "m": "m_ja", "my": "mʲ_ja", "n": "n_ja",
    "ny": "ɲ_ja", "p": "p_ja", "py": "p_ja", "r": "ɾ_ja", "ry": "ɾʲ_ja",
    "s": "s_ja", "sh": "ɕ_ja", "t": "t_ja", "ts": "ts_ja", "w": "w_ja",
    "y": "j_ja", "z": "z_ja",
}
VOWELS = {"a_ja", "i_ja", "i̥_ja", "ɨ_ja", "ɨ̥_ja", "ɯ_ja", "ɯ̥_ja", "e_ja", "o_ja"}


def _midi_hz(pitch: int) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def build_source_row(
    score: dict,
    g2p: Callable[[str], list[str]],
) -> tuple[dict, list[dict]]:
    """Force a JA phoneme sequence into contiguous score timing for diagnosis."""
    notes = score["notes"]
    if any(abs(after["start"] - (before["start"] + before["duration"])) > 1e-5
           for before, after in zip(notes, notes[1:])):
        raise ValueError("bounded content-source probe requires a contiguous score")
    phones: list[str] = []
    durations: list[float] = []
    intervals: list[tuple[float, float, str, int]] = []
    evidence: list[dict] = []
    for note in notes:
        raw = g2p(note["lyric"])
        try:
            symbols = [PHONE_MAP[value] for value in raw]
        except KeyError as error:
            raise ValueError(f"unsupported OpenJTalk phone: {error.args[0]}") from error
        if not symbols or not any(symbol in VOWELS for symbol in symbols):
            raise ValueError(f"no vowel nucleus for lyric: {note['lyric']}")
        consonants = sum(symbol not in VOWELS for symbol in symbols)
        onset = min(0.065, note["duration"] * 0.12)
        consonant_total = min(consonants * onset, note["duration"] * 0.35)
        consonant_duration = consonant_total / consonants if consonants else 0.0
        vowel_duration = (note["duration"] - consonant_total) / sum(
            symbol in VOWELS for symbol in symbols
        )
        cursor = float(note["start"])
        local = []
        for symbol in symbols:
            duration = vowel_duration if symbol in VOWELS else consonant_duration
            phones.append(symbol)
            durations.append(duration)
            intervals.append((cursor, cursor + duration, symbol, int(note["pitch"])))
            local.append({"phoneme": symbol, "duration_seconds": round(duration, 7)})
            cursor += duration
        durations[-1] += note["start"] + note["duration"] - cursor
        intervals[-1] = (intervals[-1][0], note["start"] + note["duration"],
                         intervals[-1][2], intervals[-1][3])
        evidence.append({
            "lyric": note["lyric"], "start_seconds": note["start"],
            "duration_seconds": note["duration"], "openjtalk_phones": raw,
            "diffsinger_phones": local, "timing_status": "score_inferred",
        })
    phrase_duration = notes[-1]["start"] + notes[-1]["duration"]
    frame_count = round(phrase_duration / TIMESTEP)
    f0 = []
    interval_index = 0
    for frame in range(frame_count):
        center = min((frame + .5) * TIMESTEP, phrase_duration - 1e-9)
        while interval_index + 1 < len(intervals) and center >= intervals[interval_index][1]:
            interval_index += 1
        _, _, symbol, pitch = intervals[interval_index]
        f0.append(0.0 if symbol in UNVOICED else _midi_hz(pitch))
    row = {
        "offset": 0,
        "text": "".join(note["lyric"] for note in notes),
        "ph_seq": " ".join(phones),
        "ph_dur": " ".join(f"{value:.7f}" for value in durations),
        "f0_seq": " ".join(f"{value:.3f}" for value in f0),
        "f0_timestep": TIMESTEP,
        "spk_mix": {"gts_ja_soprano": 1.0},
    }
    return row, evidence


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import librosa
    import matplotlib.pyplot as plt
    import pyopenjtalk
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import (
        AutoFeatureExtractor, AutoModelForAudioXVector, AutoModelForSpeechSeq2Seq,
        AutoProcessor,
    )

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(CACHE / "soulx-singer")]
    from analyze_rc8_defects import FFT, metrics as multires_metrics
    from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch
    from gyu_singer.inference.acoustic_style import adapt_waveform
    from gyu_singer.inference.quality_controller import STYLE
    from gyu_singer.inference.rc8 import GyuSingerRC8Renderer
    from gyu_singer.score import normalize_score
    from preprocess.tools.f0_extraction import F0Extractor

    cases = {"quality_ja": "examples/quality_ja.json", "heldout_ja": "examples/heldout_ja.json"}
    output = ROOT / "artifacts/reports/diffsinger_ja_content_source"
    output.mkdir(parents=True, exist_ok=True)
    ds_dir = output / "ds"
    ds_dir.mkdir(exist_ok=True)
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    scores: dict[str, dict] = {}
    timing_evidence = {}
    for case, score_name in cases.items():
        score = normalize_score(json.loads((ROOT / score_name).read_text()))
        row, evidence = build_source_row(score, lambda text: pyopenjtalk.g2p(text).split())
        ds_path = ds_dir / f"{case}.ds"
        ds_path.write_text(json.dumps([row], ensure_ascii=False, indent=2) + "\n")
        timing_evidence[case] = evidence
        scores[case] = score
        command = [
            str(RUNTIME_ROOT / ".venv-diffsinger/bin/python"), "scripts/infer.py", "acoustic",
            str(ds_path), "--exp", "gtsinger_ja_source", "--ckpt", "15000",
            "--spk", "gts_ja_soprano", "--out", str(source_dir), "--title", f"{case}_diffsinger",
            "--depth", "0", "--seed", "20260718",
        ]
        subprocess.run(command, cwd=CACHE / "diffsinger", check=True,
                       env=os.environ | {"PYTHONPATH": str(CACHE / "diffsinger")})

    rows, source_paths, targets = [], {}, {}
    renderer = GyuSingerRC8Renderer(RUNTIME_ROOT / "data/processed/master/216.wav", root=RUNTIME_ROOT)
    try:
        identity = renderer._identity_vector()
        identity_ref = renderer.reference_features + .05 * identity.repeat(
            (renderer.reference_features.shape[0] + identity.shape[0] - 1) // identity.shape[0]
        )[:renderer.reference_features.shape[0]]
        for case, score in scores.items():
            case_dir = output / case
            raw_dir, listening = case_dir / "raw_soulx", case_dir / "listening"
            raw_dir.mkdir(parents=True, exist_ok=True)
            listening.mkdir(parents=True, exist_ok=True)
            duration = score["notes"][-1]["start"] + score["notes"][-1]["duration"]
            lyrics = "".join(note["lyric"] for note in score["notes"])
            omni = source_dir / f"{case}_omnivoice.wav"
            renderer.omnivoice.request({"language": "ja", "lyrics": lyrics, "duration": duration,
                                        "output": str(omni)})
            diff = source_dir / f"{case}_diffsinger.wav"
            source_paths[case] = {"omnivoice": omni, "diffsinger": diff}

            controls = np.array([.8, 0, 0, 0, 0], dtype="float32")
            for index, name in enumerate(("dynamics", "breathiness", "tension", "brightness", "vibrato")):
                if score["curves"][name]:
                    controls[index] = float(np.mean([p["value"] for p in score["curves"][name]]))
            preset = torch.tensor(STYLE[renderer._content_style_preset(score["style"])],
                                  device=renderer.pitch_controller.device)
            adapted = {}
            for variant, source in source_paths[case].items():
                audio, rate = sf.read(source, dtype="float32", always_2d=True)
                audio = adapt_waveform(
                    audio.mean(1), rate, renderer.acoustic_adapter, identity_ref,
                    torch.from_numpy(controls).to(renderer.pitch_controller.device), preset,
                    score["style"]["acoustic_style_strength"],
                )
                adapted[variant] = case_dir / f"content_{variant}.wav"
                sf.write(adapted[variant], audio, rate, subtype="PCM_16")
            style = renderer._style_vector(score["style"], renderer.pitch_controller.device)
            identity_path, style_path = case_dir / "identity.npy", case_dir / "style.npy"
            np.save(identity_path, identity.detach().cpu().numpy())
            np.save(style_path, style.detach().cpu().numpy())
            expressive = renderer._predict_pitch(score) * score["style"]["prosody_strength"]
            target_f0, _ = renderer._target_f0(score, duration, expressive.cpu().numpy())
            targets[case] = target_f0
            f0_path = case_dir / "target_f0.npy"
            np.save(f0_path, target_f0)
            common = {
                "f0_npy": str(f0_path), "identity_npy": str(identity_path),
                "style_npy": str(style_path), **renderer._decoder_options(score),
            }
            for variant in ("omnivoice", "diffsinger"):
                raw = raw_dir / f"{variant}.wav"
                final = listening / f"{variant}.wav"
                renderer.soulx.request(common | {"source": str(adapted[variant]), "output": str(raw)})
                audio, rate = sf.read(raw, dtype="float32", always_2d=True)
                audio = audio.mean(1)
                if rate != 48_000:
                    audio = resample_poly(audio, 48_000, rate).astype("float32")
                refined = renderer.acoustic_refiner.process(audio)
                audio += .25 * (refined - audio)
                refined = renderer.spectral_refiner.process(audio)
                audio += .5 * (refined - audio)
                audio *= min(1.0, .97 / max(float(np.max(np.abs(audio))), 1e-8))
                sf.write(final, audio, 48_000, subtype="PCM_24")
                rows.append({
                    "case": case, "variant": variant,
                    "path": str(final.relative_to(ROOT)), "sha256": sha256(final),
                    "score": cases[case],
                })
    finally:
        renderer.close()
        del renderer
        torch.cuda.empty_cache()

    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24_000, hop_size=480, verbose=False,
    )
    for row in rows:
        path = ROOT / row["path"]
        row.update(acoustics(path) | pitch(path, targets[row["case"]], extractor)
                   | multires_metrics(path, ROOT / row["score"]))
    del extractor
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", dtype=torch.float16,
    ).cuda().eval()
    wavlm_processor = AutoFeatureExtractor.from_pretrained(CACHE / "wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained(CACHE / "wavlm-base-plus-sv").cuda().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(CACHE / "spkrec-ecapa-voxceleb"), savedir=str(CACHE / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cuda:0"},
    )

    def transcribe(path: Path) -> str:
        values = processor(audio16(path), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = whisper.generate(values.input_features.cuda().half(), language="ja", task="transcribe",
                                   max_new_tokens=64)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    def speaker(path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = audio16(path)
        values = wavlm_processor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            first = wavlm(**{key: value.cuda() for key, value in values.items()}).embeddings
            second = ecapa.encode_batch(torch.from_numpy(audio)[None].cuda())
        first = torch.nn.functional.normalize(first, dim=-1).squeeze().cpu().numpy()
        second = second.squeeze().cpu().numpy(); second /= max(np.linalg.norm(second), 1e-8)
        return first, second

    reference = speaker(RUNTIME_ROOT / "data/processed/master/216.wav")
    source_rows = []
    for case, paths in source_paths.items():
        expected = normalized("".join(note["lyric"] for note in scores[case]["notes"]))
        for variant, path in paths.items():
            transcript = transcribe(path)
            source_rows.append({
                "case": case, "variant": variant, "path": str(path.relative_to(ROOT)),
                "whisper_transcript": transcript,
                "lyric_similarity": round(SequenceMatcher(None, expected, normalized(transcript)).ratio(), 4),
            })
    for row in rows:
        expected = normalized("".join(note["lyric"] for note in scores[row["case"]]["notes"]))
        transcript = transcribe(ROOT / row["path"])
        current = speaker(ROOT / row["path"])
        row |= {
            "asr_transcript": transcript,
            "asr_lyric_similarity": round(SequenceMatcher(None, expected, normalized(transcript)).ratio(), 4),
            "wavlm_to_gyu": round(float(np.dot(reference[0], current[0])), 5),
            "ecapa_to_gyu": round(float(np.dot(reference[1], current[1])), 5),
        }
    del whisper, wavlm, ecapa
    torch.cuda.empty_cache()

    plots = {}
    for case in cases:
        selected = {
            "source_omnivoice": source_paths[case]["omnivoice"],
            "source_diffsinger": source_paths[case]["diffsinger"],
            **{f"final_{row['variant']}": ROOT / row["path"] for row in rows if row["case"] == case},
        }
        fig, axes = plt.subplots(len(selected), 4, figsize=(18, 3 * len(selected)), constrained_layout=True)
        for row_index, (label, path) in enumerate(selected.items()):
            audio, rate = sf.read(path, dtype="float32", always_2d=True); audio = audio.mean(1)
            axes[row_index, 0].plot(np.arange(len(audio)) / rate, audio, linewidth=.35)
            axes[row_index, 0].set_title(f"{label}: waveform")
            for column, (resolution, (n_fft, hop)) in enumerate(FFT.items(), 1):
                spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft,
                                                                         hop_length=hop)), ref=np.max)
                axes[row_index, column].imshow(
                    spectrum, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                    extent=[0, len(audio) / rate, 0, rate / 2],
                )
                axes[row_index, column].set_ylim(0, min(12_000, rate / 2))
                axes[row_index, column].set_title(f"{label}: {resolution} STFT")
        plot = output / case / "waveform_multires_stft.png"
        fig.savefig(plot, dpi=120); plt.close(fig)
        plots[case] = str(plot.relative_to(ROOT))

    by = {(row["case"], row["variant"]): row for row in rows}
    heldout = by[("heldout_ja", "diffsinger")]
    quality = by[("quality_ja", "diffsinger")]
    heldout_source = next(row for row in source_rows if row["case"] == "heldout_ja" and row["variant"] == "diffsinger")
    baseline_manifest = json.loads((ROOT / "artifacts/reports/rc8_candidate3_full/manifest.json").read_text())
    regression = {
        case: {"expected": item["sha256"], "actual": sha256(Path(item["path"])),
               "unchanged": item["sha256"] == sha256(Path(item["path"]))}
        for case, item in baseline_manifest["files"].items()
    }
    gates = {
        "heldout_source_similarity_at_least_090": heldout_source["lyric_similarity"] >= .90,
        "heldout_source_repetition_removed": heldout_source["lyric_similarity"] >= .90
            and heldout_source["whisper_transcript"].count("新しい歌") <= 1,
        "heldout_final_similarity_at_least_090": heldout["asr_lyric_similarity"] >= .90,
        "heldout_final_repetition_removed": heldout["asr_transcript"].count("新しい歌") <= 1,
        "quality_similarity_nonregression": quality["asr_lyric_similarity"] >= by[("quality_ja", "omnivoice")]["asr_lyric_similarity"],
        "pitch_nonregression": all(by[(case, "diffsinger")]["pitch_mae_cents"] <= by[(case, "omnivoice")]["pitch_mae_cents"] + 2 for case in cases),
        "voicing_nonregression": all(by[(case, "diffsinger")]["voicing_accuracy"] >= by[(case, "omnivoice")]["voicing_accuracy"] - .01 for case in cases),
        "hf_spike_nonregression": all(by[(case, "diffsinger")]["hf_spike_p99_over_median"] <= by[(case, "omnivoice")]["hf_spike_p99_over_median"] * 1.05 for case in cases),
        "sample_jump_nonregression": all(by[(case, "diffsinger")]["sample_jump_p999"] <= by[(case, "omnivoice")]["sample_jump_p999"] * 1.05 for case in cases),
        "identity_nonregression": all(by[(case, "diffsinger")][metric] >= by[(case, "omnivoice")][metric] - .02 for case in cases for metric in ("wavlm_to_gyu", "ecapa_to_gyu")),
        "existing_9_file_sha_unchanged": all(value["unchanged"] for value in regression.values()),
    }
    report = {
        "status": "replacement_probe_human_pending" if all(gates.values()) else "replacement_probe_reject",
        "runtime_integrated": False, "rc8_human_status": "pending", "rc9_started": False,
        "source_model": {
            "name": "GTSinger JA soprano DiffSinger source checkpoint",
            "revision": "gtsinger_ja_source/model_ckpt_steps_15000.ckpt",
            "decode": "depth-zero auxiliary decoder", "role": "content source only",
            "license": "CC BY-NC-SA 4.0", "release_allowed": False,
        },
        "timing_evidence": timing_evidence, "source_whisper": source_rows, "rows": rows,
        "waveform_multires_stft": plots, "existing_9_file_regression": regression, "gates": gates,
        "constraints": {"phrase_level_soulx_decode": True, "per_note_tts": False,
                        "final_wav_stitching": False, "waveform_pitch_shift": False},
    }
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "gates": gates, "source_whisper": source_rows,
                      "final": [{key: row[key] for key in ("case", "variant", "asr_transcript",
                                                            "asr_lyric_similarity", "pitch_mae_cents",
                                                            "voicing_accuracy", "hf_spike_p99_over_median",
                                                            "sample_jump_p999", "wavlm_to_gyu", "ecapa_to_gyu")}
                                for row in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
