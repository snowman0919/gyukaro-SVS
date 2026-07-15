#!/usr/bin/env python3
"""Index a bounded CC-BY VocalSet prior without extracting or modifying audio."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/work/VocalSet1-2.fixed.zip"
ALLOWED = {
    "scales/fast_forte", "scales/fast_piano", "scales/slow_forte", "scales/slow_piano",
    "scales/straight", "scales/vibrato", "scales/breathy", "scales/belt",
    "arpeggios/fast_forte", "arpeggios/fast_piano", "arpeggios/slow_forte",
    "arpeggios/slow_piano", "arpeggios/straight", "arpeggios/vibrato",
    "arpeggios/breathy", "arpeggios/belt", "long_tones/straight",
    "long_tones/forte", "long_tones/pp", "long_tones/messa",
}


def main() -> None:
    candidates: dict[tuple[str, str, str], list] = {}
    with ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if len(path.parts) != 5 or path.parts[0] != "data_by_singer" or path.suffix != ".wav":
                continue
            speaker, category = path.parts[1], "/".join(path.parts[2:4])
            vowel = path.stem.rsplit("_", 1)[-1]
            if category in ALLOWED and vowel in {"a", "e", "i", "o", "u"}:
                candidates.setdefault((speaker, category, vowel), []).append(info)
        chosen = []
        for key, infos in sorted(candidates.items()):
            info = min(infos, key=lambda item: hashlib.sha256(item.filename.encode()).digest())
            speaker, category, vowel = key
            chosen.append({
                "id": "vocalset_" + Path(info.filename).stem,
                "dataset": "VocalSet-1.2",
                "license": "CC-BY-4.0",
                "source_member": info.filename,
                "uncompressed_bytes": info.file_size,
                "speaker": speaker,
                "language": "nonlexical_vowel",
                "phoneme": vowel,
                "phoneme_label_status": "inferred_from_official_filename",
                "technique": category,
                "split": "validation" if speaker in {"female10", "male10"} else "train",
                "training_role": "generic_score_native_singing_prior",
            })
    target = ROOT / "data/external/manifests/score_native_vocalset_prior.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in chosen))
    report = {
        "status": "index_ready_audio_not_extracted",
        "rows": len(chosen),
        "speakers": len({row["speaker"] for row in chosen}),
        "categories": len({row["technique"] for row in chosen}),
        "uncompressed_gib": round(sum(row["uncompressed_bytes"] for row in chosen) / 2**30, 3),
        "labels": "vowels inferred from official filenames; pitch/F0 must be measured after extraction",
        "lexical_excerpts_used": False,
        "raw_audio_bundled": False,
    }
    (ROOT / "artifacts/reports/score_native_prior_index.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
