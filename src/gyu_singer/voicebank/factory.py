from __future__ import annotations

import csv
import hashlib
from importlib import resources
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import soundfile as sf

from gyu_singer.evaluation.korean_lexical import classify_alignment
from gyu_singer.experiments.korean_phones import encode_korean

from .audio import energy_vad, inspect_audio_directory, normalize_audio
from .rights import validate_rights_manifest


class FactoryError(ValueError):
    pass


PHASES = ("init", "inspect", "prepare", "train", "evaluate", "review-pack", "package")


def choose_transcript(exact_script, verified_metadata, corrected_draft, automatic_draft, language: str) -> dict:
    if exact_script:
        return {"text": exact_script, "source": "user_provided_exact_script", "training_trust": "high"}
    if verified_metadata:
        return {"text": verified_metadata, "source": "verified_metadata", "training_trust": "high"}
    if corrected_draft:
        return {"text": corrected_draft, "source": "user_corrected_draft", "training_trust": "high"}
    return {
        "text": automatic_draft or "",
        "source": "untrusted_draft_transcript" if language == "ko" else "automatic_stt_draft",
        "training_trust": "excluded_until_review",
    }


def deterministic_split(rows: list[dict]) -> list[dict]:
    trusted = [dict(row) for row in rows if row.get("training_trust") == "high"]
    hashes = [row["audio_sha256"] for row in trusted]
    if len(set(hashes)) != len(hashes):
        raise FactoryError("duplicate clips cannot cross frozen splits")
    ordered = sorted(trusted, key=lambda row: (row["audio_sha256"], row["id"]))
    count = len(ordered)
    for index, row in enumerate(ordered):
        fraction = index / max(count, 1)
        row["split"] = "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "heldout"
    return ordered


def select_checkpoint(candidates: list[dict]) -> dict:
    passing = [row for row in candidates if row.get("preservation_pass")]
    if not passing:
        raise FactoryError("no checkpoint passes preservation gates")
    return min(passing, key=lambda row: (-row["identity_score"], row["update_norm"], row["step"], row["id"]))


def _phones(text: str, language: str) -> list[str]:
    if not text.strip():
        return []
    if language == "ko":
        return list(encode_korean(text, "ko_canonical_v1").symbols)
    if language == "en":
        return [character.lower() for character in text if character.isalpha()]
    return [character for character in text if not character.isspace()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VoicebankFactory:
    def __init__(self, repository: Path, workspace: Path):
        self.repository = repository.resolve()
        self.workspace = workspace.resolve()
        config_path = self.repository / "configs/voicebank_factory.json"
        if not config_path.is_file():
            config_path = resources.files("gyu_singer").joinpath("voicebank_factory.json")
        self.config = json.loads(config_path.read_text())

    def _read_state(self) -> dict:
        path = self.workspace / "state.json"
        return json.loads(path.read_text()) if path.is_file() else {"last_completed_phase": None, "status": "new"}

    def _record(self, phase: str, status: str, **extra) -> dict:
        state = {"last_completed_phase": phase, "status": status, **extra}
        temporary = self.workspace / "state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.workspace / "state.json")
        with (self.workspace / "factory.jsonl").open("a") as handle:
            handle.write(json.dumps({"phase": phase, "status": status, **extra}, sort_keys=True) + "\n")
        return state

    def status(self) -> dict:
        return self._read_state() | {"workspace": str(self.workspace)}

    def init(self, input_directory: Path, name: str, languages: list[str], rights_path: Path, dry_run: bool = False) -> dict:
        input_directory = input_directory.resolve()
        rights = json.loads(rights_path.read_text())
        validate_rights_manifest(rights)
        if not input_directory.is_dir() or set(languages) != set(rights["languages"]):
            raise FactoryError("input directory or language list does not match the rights manifest")
        files = [path for path in input_directory.iterdir() if path.is_file()]
        estimate = {"status": "dry_run", "files": len(files), "input_bytes": sum(path.stat().st_size for path in files), "estimated_workspace_bytes": sum(path.stat().st_size for path in files) * 2}
        if dry_run:
            return estimate
        if (self.workspace / "project.json").is_file():
            current = json.loads((self.workspace / "project.json").read_text())
            if current["input_directory"] == str(input_directory) and current["name"] == name:
                return {"status": "already_complete", "phase": "init"}
            raise FactoryError("workspace already belongs to a different project")
        self.workspace.mkdir(parents=True)
        (self.workspace / "config").mkdir()
        project = {"schema_version": 1, "name": name, "languages": languages, "input_directory": str(input_directory), "rights_manifest_sha256": _sha256(rights_path)}
        (self.workspace / "project.json").write_text(json.dumps(project, indent=2, sort_keys=True) + "\n")
        (self.workspace / "config/rights.json").write_text(json.dumps(rights, indent=2, sort_keys=True) + "\n")
        self._record("init", "complete", resource_estimate=estimate)
        return {"status": "complete", "phase": "init"}

    def inspect(self) -> dict:
        project = json.loads((self.workspace / "project.json").read_text())
        report = inspect_audio_directory(Path(project["input_directory"]))
        (self.workspace / "reports").mkdir(exist_ok=True)
        (self.workspace / "reports/audio_inspection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        status = "audio_inspection_failed" if report["corrupt_count"] else "audio_inspection_complete"
        self._record("inspect", status, valid=report["valid_count"], corrupt=report["corrupt_count"])
        return report | {"status": status}

    def prepare(self, review_manifest: Path | None = None) -> dict:
        report_path = self.workspace / "reports/audio_inspection.json"
        inspection = json.loads(report_path.read_text()) if report_path.is_file() else self.inspect()
        if inspection["corrupt_count"]:
            raise FactoryError("audio inspection has corrupt inputs")
        project = json.loads((self.workspace / "project.json").read_text())
        rights = json.loads((self.workspace / "config/rights.json").read_text())
        corrections = self._read_review(review_manifest) if review_manifest else {}
        rows = []
        alignment_root = self.workspace / "reports/alignments"
        alignment_root.mkdir(parents=True, exist_ok=True)
        for index, audio_row in enumerate(inspection["rows"]):
            if audio_row["status"] != "ok":
                continue
            source = Path(audio_row["path"])
            clip_id = f"clip-{index:05d}"
            file_metadata = rights.get("file_metadata", {}).get(audio_row["file"], {})
            language = file_metadata.get("language", project["languages"][0] if len(project["languages"]) == 1 else "unknown")
            normalized = normalize_audio(source, self.workspace / "prepared/audio" / f"{clip_id}.wav", self.config["target_sample_rate"])
            vad = energy_vad(Path(normalized["path"]))
            transcript = choose_transcript(
                rights.get("known_scripts", {}).get(audio_row["file"]), None,
                corrections.get(clip_id), None, language,
            )
            phones = _phones(transcript["text"], language)
            confidence = 0.5 if phones and transcript["training_trust"] == "high" else 0.0
            classification = classify_alignment("forced" if confidence else "inferred", confidence)
            duration = audio_row["duration_seconds"]
            boundaries = [round(position * duration / max(len(phones), 1), 6) for position in range(len(phones) + 1)]
            alignment = {"source": "uniform_forced_alignment_hook" if confidence else "unavailable", "confidence": confidence, "classification": classification, "boundaries_seconds": boundaries, "manual_correction": None}
            (alignment_root / f"{clip_id}.json").write_text(json.dumps({"phones": phones, "alignment": alignment}, indent=2) + "\n")
            rows.append({
                "id": clip_id, "source_file": audio_row["file"], "language": language,
                "normalized_audio": normalized["path"], "audio_sha256": normalized["sha256"],
                "duration_seconds": duration, "pitch_hz_estimate": audio_row["pitch_hz_estimate"],
                "transcript": transcript, "training_trust": transcript["training_trust"],
                "phones": phones, "alignment": alignment,
                "vad": vad,
                "coverage_tags": sorted(set(file_metadata.get("coverage_tags", []))),
                "recording_condition": file_metadata.get("recording_condition", rights["recording_environment"]),
            })
        manifest_root = self.workspace / "manifests"
        manifest_root.mkdir(exist_ok=True)
        (manifest_root / "segments.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        with (manifest_root / "transcript_review.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "source_file", "draft_text", "corrected_text", "review_status"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"id": row["id"], "source_file": row["source_file"], "draft_text": row["transcript"]["text"], "corrected_text": "", "review_status": "accepted" if row["training_trust"] == "high" else "needs_review"})
        coverage = self._coverage(rows, project["languages"])
        (self.workspace / "reports/coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
        plan = self._recording_plan(coverage, project["languages"])
        (self.workspace / "reports/recording_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        strategy = self._adaptation_plan(project["languages"])
        (self.workspace / "reports/adaptation_plan.json").write_text(json.dumps(strategy, indent=2, sort_keys=True) + "\n")
        trusted = [row for row in rows if row["training_trust"] == "high" and row["alignment"]["classification"] != "rejected"]
        split = deterministic_split(trusted)
        (manifest_root / "frozen_dataset.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in split))
        environment = {"python": sys.version.split()[0], "platform": platform.platform(), "seeds": self.config["seeds"], "checkpoint_selection": self.config["checkpoint_selection"], "early_stop": self.config["early_stop"], "heldout_tuning": False}
        (self.workspace / "config/training_protocol.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
        status = "coverage_ready" if coverage["sufficient"] else "dataset_needs_more_recording"
        self._record("prepare", status, eligible_rows=len(split), coverage_gaps=coverage["gaps"])
        return {"status": status, "eligible_rows": len(split), "coverage": coverage}

    def _read_review(self, path: Path) -> dict[str, str]:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        return {row["id"]: row.get("corrected_text", row.get("text", "")) for row in rows if row.get("review_status") == "accepted" and row.get("corrected_text", row.get("text", ""))}

    def _coverage(self, rows: list[dict], languages: list[str]) -> dict:
        thresholds = self.config["coverage_thresholds"]
        unique_phones = sorted({phone for row in rows for phone in row["phones"]})
        duration = sum(row["duration_seconds"] for row in rows)
        pitches = [row["pitch_hz_estimate"] for row in rows if row["pitch_hz_estimate"]]
        gaps = []
        if duration < thresholds["minimum_total_duration_seconds"]:
            gaps.append(f"total_duration_{thresholds['minimum_total_duration_seconds']}s")
        if len(unique_phones) < thresholds["minimum_unique_phones"]:
            gaps.append(f"unique_phones_{thresholds['minimum_unique_phones']}")
        for language in languages:
            if sum(row["language"] == language for row in rows) < thresholds["minimum_clips_per_language"]:
                gaps.append(f"language_{language}_{thresholds['minimum_clips_per_language']}_clips")
        observed_tags = {tag for row in rows for tag in row["coverage_tags"]}
        gaps.extend(tag for tag in thresholds["required_tags"] if tag not in observed_tags)
        conditions = {row["recording_condition"] for row in rows}
        if not conditions:
            gaps.append("recording_condition_balance")
        return {"sufficient": not gaps, "gaps": gaps, "total_duration_seconds": round(duration, 3), "unique_phones": unique_phones, "pitch_hz_min": min(pitches) if pitches else None, "pitch_hz_max": max(pitches) if pitches else None, "observed_tags": sorted(observed_tags), "recording_conditions": sorted(conditions), "language_counts": {language: sum(row["language"] == language for row in rows) for language in languages}}

    def _recording_plan(self, coverage: dict, languages: list[str]) -> dict:
        return {"status": "dataset_needs_more_recording" if coverage["gaps"] else "coverage_complete", "targeted_gaps": coverage["gaps"], "scripts": [{"language": language, "prompt": f"Record reviewed {language} phrases covering: {', '.join(coverage['gaps'])}"} for language in languages]}

    def _adaptation_plan(self, languages: list[str]) -> dict:
        foundation = "gtsinger-ja-soprano" if languages == ["ja"] else None
        return {"selected_foundation": foundation, "status": "research_only" if foundation else "blocked_no_authorized_multilingual_foundation", "rejected_alternatives": ["OmniVoice-to-SoulX production route", "GTSinger tenor", "GYU mix20"], "trainable_modules": [] if not foundation else ["bounded_linguistic_or_identity_adapter_after_gates"], "frozen_modules": ["foundation", "pitch", "duration", "decoder", "vocoder"], "expected_risks": ["lexical_regression", "pitch_regression", "identity_inconsistency"], "required_gates": ["coverage", "phone_lexical", "pitch", "voicing", "artifacts", "multi_seed", "human_approval"]}

    def train(self) -> dict:
        coverage = json.loads((self.workspace / "reports/coverage.json").read_text())
        if not coverage["sufficient"]:
            raise FactoryError("coverage is insufficient; dataset_needs_more_recording")
        plan = json.loads((self.workspace / "reports/adaptation_plan.json").read_text())
        if not plan["selected_foundation"]:
            raise FactoryError("no authorized foundation; training blocked")
        raise FactoryError("bounded training runner is not activated without an approved experiment protocol")

    def evaluate(self) -> dict:
        checkpoint = self.workspace / "checkpoints/selected.json"
        status = "evaluation_blocked_no_checkpoint" if not checkpoint.is_file() else "voicebank_candidate_human_pending"
        report = {"status": status, "korean_lexical": "phone_centered_whisper_auxiliary", "ja_en_asr": "auxiliary_not_ground_truth", "metrics": ["lexical", "score_pitch", "voicing", "timing", "artifacts", "speaker_identity", "seed_stability", "long_form", "resources"]}
        (self.workspace / "reports/evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        self._record("evaluate", status)
        return report

    def review_pack(self) -> dict:
        evaluation = json.loads((self.workspace / "reports/evaluation.json").read_text())
        output = self.workspace / "review"
        output.mkdir(exist_ok=True)
        report = {"status": "review_blocked_no_candidate" if evaluation["status"].startswith("evaluation_blocked") else "voicebank_candidate_human_pending", "blind": True, "transcript_hints": False, "audio_bundled": False}
        (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        self._record("review-pack", report["status"])
        return report

    def package(self, release: bool = False, diagnostic: bool = False) -> dict:
        if release:
            approval = self.workspace / "human_approval.json"
            if not approval.is_file() or json.loads(approval.read_text()).get("approved") is not True:
                raise FactoryError("release package requires an explicit approved human approval record")
            raise FactoryError("central release gate is blocked")
        if not diagnostic:
            raise FactoryError("choose diagnostic=True or release=True")
        output = self.workspace / "packages/voicebank-diagnostic"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        (output / "README.md").write_text("# NOT A RELEASE — DIAGNOSTIC VOICEBANK PROJECT\n\nNo checkpoint, audio, or approval is bundled.\n")
        metadata = {"status": "diagnostic_package_not_a_release", "workspace_state": self._read_state(), "checkpoints_bundled": False, "audio_bundled": False}
        (output / "PACKAGE.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        sums = [f"{_sha256(path)}  {path.name}" for path in sorted(output.iterdir())]
        (output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        self._record("package", metadata["status"])
        return metadata

    def build(self) -> dict:
        state = self._read_state()
        if state["last_completed_phase"] is None:
            raise FactoryError("run init before build")
        if state["last_completed_phase"] == "init":
            inspection = self.inspect()
            if inspection["status"] != "audio_inspection_complete":
                return inspection
        state = self._read_state()
        if state["last_completed_phase"] in {"init", "inspect"}:
            prepared = self.prepare()
            if prepared["status"] != "coverage_ready":
                return prepared
        if self._read_state()["last_completed_phase"] == "prepare":
            coverage = json.loads((self.workspace / "reports/coverage.json").read_text())
            if not coverage["sufficient"]:
                return {"status": "dataset_needs_more_recording", "coverage": coverage}
        try:
            return self.train()
        except FactoryError as error:
            return {"status": "training_blocked", "reason": str(error)}
