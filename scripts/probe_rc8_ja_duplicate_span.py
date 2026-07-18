#!/usr/bin/env python3
"""Bounded JA duplicate-span diagnostic; never changes the RC8 runtime."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector, AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
RUNTIME_ROOT = ROOT if (ROOT / ".venv-soulx").is_dir() else ROOT.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(CACHE / "soulx-singer")]

from analyze_rc8_defects import FFT, metrics as multires_metrics  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
from gyu_singer.inference.acoustic_style import adapt_waveform  # noqa: E402
from gyu_singer.inference.content_timing import (  # noqa: E402
    ctc_phone_alignment,
    duplicate_span_content_warp,
    latent_content_warp,
)
from gyu_singer.inference.quality_controller import STYLE  # noqa: E402
from gyu_singer.inference.rc8 import GyuSingerRC8Renderer  # noqa: E402
from gyu_singer.score import normalize_score  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


CASES = {"quality_ja": "examples/quality_ja.json", "heldout_ja": "examples/heldout_ja.json"}
VARIANTS = ("current_rc8", "global_ctc_025", "chunked_single_decode", "duplicate_span_candidate")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def postprocess(renderer: GyuSingerRC8Renderer, source: Path, target: Path) -> None:
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if rate != 48_000:
        audio = resample_poly(audio, 48_000, rate).astype("float32")
    waveform = renderer.acoustic_refiner.process(audio)
    audio += .25 * (waveform - audio)
    spectral = renderer.spectral_refiner.process(audio)
    audio += .5 * (spectral - audio)
    audio *= min(1.0, .97 / max(float(np.max(np.abs(audio))), 1e-8))
    sf.write(target, audio, 48_000, subtype="PCM_24")


def plot_case(case_dir: Path, paths: dict[str, Path]) -> str:
    fig, axes = plt.subplots(len(paths), 4, figsize=(18, 3 * len(paths)), constrained_layout=True)
    for row, (name, path) in enumerate(paths.items()):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(1)
        axes[row, 0].plot(np.arange(len(audio)) / rate, audio, linewidth=.35)
        axes[row, 0].set_title(f"{name}: waveform")
        for column, (resolution, (n_fft, hop)) in enumerate(FFT.items(), 1):
            spectrum = librosa.amplitude_to_db(
                np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)), ref=np.max,
            )
            axes[row, column].imshow(
                spectrum, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                extent=[0, len(audio) / rate, 0, rate / 2],
            )
            axes[row, column].set_ylim(0, min(12_000, rate / 2))
            axes[row, column].set_title(f"{name}: {resolution} STFT")
    target = case_dir / "waveform_multires_stft.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    return str(target.relative_to(ROOT))


def main() -> None:
    output = ROOT / "artifacts/reports/rc8_ja_duplicate_span"
    output.mkdir(parents=True, exist_ok=True)
    renderer = GyuSingerRC8Renderer(RUNTIME_ROOT / "data/processed/master/216.wav", root=RUNTIME_ROOT)
    rows, sources, targets, evidence_by_case = [], {}, {}, {}
    try:
        identity = renderer._identity_vector()
        identity_ref = renderer.reference_features + .05 * identity.repeat(
            (renderer.reference_features.shape[0] + identity.shape[0] - 1) // identity.shape[0]
        )[:renderer.reference_features.shape[0]]
        for case, score_name in CASES.items():
            case_dir = output / case
            raw_dir, listening = case_dir / "raw_soulx", case_dir / "listening"
            raw_dir.mkdir(parents=True, exist_ok=True)
            listening.mkdir(parents=True, exist_ok=True)
            score = normalize_score(json.loads((ROOT / score_name).read_text()))
            duration = max(note["start"] + note["duration"] for note in score["notes"])
            lyrics = "".join(note["lyric"] for note in score["notes"])
            full_raw, grouped_raw = case_dir / "omnivoice_full.wav", case_dir / "omnivoice_grouped.wav"
            renderer.omnivoice.request({"language": "ja", "lyrics": lyrics, "duration": duration, "output": str(full_raw)})
            grouped = []
            for index, notes in enumerate((score["notes"][:2], score["notes"][2:])):
                chunk_duration = notes[-1]["start"] + notes[-1]["duration"] - notes[0]["start"]
                path = case_dir / f"omnivoice_chunk_{index}.wav"
                renderer.omnivoice.request({
                    "language": "ja", "lyrics": "".join(note["lyric"] for note in notes),
                    "duration": chunk_duration, "output": str(path),
                })
                audio, rate = sf.read(path, dtype="float32", always_2d=True)
                target = round(chunk_duration * rate)
                grouped.append(np.pad(audio.mean(1)[:target], (0, max(0, target - len(audio)))))
            sf.write(grouped_raw, np.concatenate(grouped), rate, subtype="PCM_16")

            controls = np.array([.8, 0, 0, 0, 0], dtype="float32")
            for index, name in enumerate(("dynamics", "breathiness", "tension", "brightness", "vibrato")):
                if score["curves"][name]:
                    controls[index] = float(np.mean([point["value"] for point in score["curves"][name]]))
            preset = torch.tensor(STYLE[renderer._content_style_preset(score["style"])], device=renderer.pitch_controller.device)
            adapted = {}
            for name, path in (("full", full_raw), ("grouped", grouped_raw)):
                audio, rate = sf.read(path, dtype="float32", always_2d=True)
                audio = adapt_waveform(
                    audio.mean(1), rate, renderer.acoustic_adapter, identity_ref,
                    torch.from_numpy(controls).to(renderer.pitch_controller.device), preset,
                    score["style"]["acoustic_style_strength"],
                )
                adapted[name] = case_dir / f"content_{name}.wav"
                sf.write(adapted[name], audio, rate, subtype="PCM_16")
            style = renderer._style_vector(score["style"], renderer.pitch_controller.device)
            identity_path, style_path = case_dir / "identity.npy", case_dir / "style.npy"
            np.save(identity_path, identity.detach().cpu().numpy())
            np.save(style_path, style.detach().cpu().numpy())
            expressive = renderer._predict_pitch(score) * score["style"]["prosody_strength"]
            info = sf.info(adapted["full"])
            target_f0, _ = renderer._target_f0(score, info.duration, expressive.cpu().numpy())
            f0_path = case_dir / "target_f0.npy"
            np.save(f0_path, target_f0)
            targets[case] = target_f0

            audio, rate = sf.read(adapted["full"], dtype="float32", always_2d=True)
            mono = audio.mean(1)
            aligned = resample_poly(mono, 16_000, rate).astype("float32") if rate != 16_000 else mono
            if renderer._ctc is None:
                bundle = torchaudio.pipelines.MMS_FA
                renderer._ctc = (bundle.get_model().eval(), bundle.get_labels())
            alignment = ctc_phone_alignment(torch.from_numpy(aligned), 16_000, score, *renderer._ctc)
            (case_dir / "ctc_alignment.json").write_text(json.dumps(alignment, ensure_ascii=False, indent=2) + "\n")
            global_warp = latent_content_warp(alignment, info.duration, len(target_f0) / 50, len(target_f0))
            candidate_warp, evidence = duplicate_span_content_warp(
                alignment, info.duration, len(target_f0) / 50, len(target_f0),
            )
            np.save(case_dir / "global_ctc_warp.npy", global_warp)
            if candidate_warp is not None:
                np.save(case_dir / "duplicate_span_warp.npy", candidate_warp)
            evidence_by_case[case] = evidence
            (case_dir / "duplicate_span_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")

            common = {
                "f0_npy": str(f0_path), "identity_npy": str(identity_path), "style_npy": str(style_path),
                **renderer._decoder_options(score),
            }
            requests = {
                "current_rc8": {"source": str(adapted["full"])},
                "global_ctc_025": {
                    "source": str(adapted["full"]), "content_warp_npy": str(case_dir / "global_ctc_warp.npy"),
                    "content_warp_strength": .25,
                },
                "chunked_single_decode": {"source": str(adapted["grouped"])},
            }
            for variant, options in requests.items():
                raw_path, path = raw_dir / f"{variant}.wav", listening / f"{variant}.wav"
                renderer.soulx.request(common | options | {"output": str(raw_path)})
                postprocess(renderer, raw_path, path)
            current = listening / "current_rc8.wav"
            candidate = listening / "duplicate_span_candidate.wav"
            if candidate_warp is None:
                shutil.copyfile(current, candidate)
            else:
                raw_path = raw_dir / "duplicate_span_candidate.wav"
                renderer.soulx.request(common | {
                    "source": str(adapted["full"]), "content_warp_npy": str(case_dir / "duplicate_span_warp.npy"),
                    "content_warp_strength": 1.0, "output": str(raw_path),
                })
                postprocess(renderer, raw_path, candidate)
            sources[case] = {"omnivoice_full": full_raw, "omnivoice_grouped": grouped_raw}
            for variant in VARIANTS:
                path = listening / f"{variant}.wav"
                rows.append({"case": case, "variant": variant, "path": str(path.relative_to(ROOT)), "sha256": sha(path), "score": score_name})
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
        row.update(acoustics(path) | pitch(path, targets[row["case"]], extractor) | multires_metrics(path, ROOT / row["score"]))
    del extractor
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).cuda().eval()
    wavlm_processor = AutoFeatureExtractor.from_pretrained(CACHE / "wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained(CACHE / "wavlm-base-plus-sv").cuda().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(CACHE / "spkrec-ecapa-voxceleb"), savedir=str(CACHE / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cuda:0"},
    )

    def transcribe(path: Path, language: str = "ja") -> str:
        values = processor(audio16(path), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = whisper.generate(values.input_features.cuda().half(), language=language, task="transcribe", max_new_tokens=64)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    def speaker(path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = audio16(path)
        values = wavlm_processor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            a = wavlm(**{key: value.cuda() for key, value in values.items()}).embeddings
            b = ecapa.encode_batch(torch.from_numpy(audio)[None].cuda())
        a = torch.nn.functional.normalize(a, dim=-1).squeeze().cpu().numpy()
        b = b.squeeze().cpu().numpy(); b /= max(np.linalg.norm(b), 1e-8)
        return a, b

    reference = speaker(ROOT / "data/processed/master/216.wav")
    source_rows = []
    for case, paths in sources.items():
        expected = normalized("".join(note["lyric"] for note in json.loads((ROOT / CASES[case]).read_text())["notes"]))
        for name, path in paths.items():
            transcript = transcribe(path)
            source_rows.append({
                "case": case, "variant": name, "path": str(path.relative_to(ROOT)),
                "whisper_transcript": transcript,
                "lyric_similarity": round(SequenceMatcher(None, expected, normalized(transcript)).ratio(), 4),
            })
    for row in rows:
        expected = normalized("".join(note["lyric"] for note in json.loads((ROOT / row["score"]).read_text())["notes"]))
        transcript = transcribe(ROOT / row["path"])
        embeddings = speaker(ROOT / row["path"])
        row |= {
            "asr_transcript": transcript,
            "asr_lyric_similarity": round(SequenceMatcher(None, expected, normalized(transcript)).ratio(), 4),
            "wavlm_to_gyu": round(float(np.dot(reference[0], embeddings[0])), 5),
            "ecapa_to_gyu": round(float(np.dot(reference[1], embeddings[1])), 5),
        }
    del whisper, wavlm, ecapa
    torch.cuda.empty_cache()

    plots = {}
    for case in CASES:
        selected = {row["variant"]: ROOT / row["path"] for row in rows if row["case"] == case}
        plots[case] = plot_case(output / case, selected)
    baseline_manifest = json.loads((ROOT / "artifacts/reports/rc8_candidate3_full/manifest.json").read_text())
    regression = {
        case: {"expected": item["sha256"], "actual": sha(Path(item["path"])), "unchanged": item["sha256"] == sha(Path(item["path"]))}
        for case, item in baseline_manifest["files"].items()
    }
    by = {(row["case"], row["variant"]): row for row in rows}
    heldout = by[("heldout_ja", "duplicate_span_candidate")]
    heldout_base = by[("heldout_ja", "current_rc8")]
    quality = by[("quality_ja", "duplicate_span_candidate")]
    quality_base = by[("quality_ja", "current_rc8")]
    gates = {
        "heldout_similarity_at_least_090": heldout["asr_lyric_similarity"] >= .90,
        "heldout_repetition_removed": heldout["asr_lyric_similarity"] >= .90 and heldout["asr_transcript"] != heldout_base["asr_transcript"],
        "quality_similarity_nonregression": quality["asr_lyric_similarity"] >= quality_base["asr_lyric_similarity"],
        "pitch_nonregression": all(by[(case, "duplicate_span_candidate")]["pitch_mae_cents"] <= by[(case, "current_rc8")]["pitch_mae_cents"] + 2 for case in CASES),
        "voicing_nonregression": all(by[(case, "duplicate_span_candidate")]["voicing_accuracy"] >= by[(case, "current_rc8")]["voicing_accuracy"] - .01 for case in CASES),
        "hf_spike_nonregression": all(by[(case, "duplicate_span_candidate")]["hf_spike_p99_over_median"] <= by[(case, "current_rc8")]["hf_spike_p99_over_median"] * 1.05 for case in CASES),
        "sample_jump_nonregression": all(by[(case, "duplicate_span_candidate")]["sample_jump_p999"] <= by[(case, "current_rc8")]["sample_jump_p999"] * 1.05 for case in CASES),
        "identity_nonregression": all(
            by[(case, "duplicate_span_candidate")][metric] >= by[(case, "current_rc8")][metric] - .02
            for case in CASES for metric in ("wavlm_to_gyu", "ecapa_to_gyu")
        ),
        "existing_9_file_sha_unchanged": all(row["unchanged"] for row in regression.values()),
    }
    report = {
        "status": "diagnostic_candidate_human_pending" if all(gates.values()) else "diagnostic_reject",
        "runtime_integrated": False, "human_listening": "pending", "variants": VARIANTS,
        "source_whisper": source_rows, "ctc_evidence": evidence_by_case, "rows": rows,
        "waveform_multires_stft": plots, "existing_9_file_regression": regression, "gates": gates,
        "constraints": {"phrase_level_soulx_decode": True, "per_note_tts": False, "final_wav_stitching": False, "waveform_pitch_shift": False},
    }
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({
        "status": report["status"], "runtime_integrated": False,
        "files": [{"case": row["case"], "variant": row["variant"], "path": row["path"], "sha256": row["sha256"]} for row in rows],
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "gates": gates, "ctc_evidence": evidence_by_case}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
