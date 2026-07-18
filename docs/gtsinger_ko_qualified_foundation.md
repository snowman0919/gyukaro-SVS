NOT A RELEASE REPORT — FOUNDATION SOURCE REJECTED

# GTSinger Korean source-qualified foundation diagnostic

## Decision

- Conclusion: `foundation_source_gate_reject`
- Training allowed: false
- Candidate source rows: 328
- Accepted rows: 88
- Accepted duration: 1028.191021 seconds (17.137 minutes)
- Failed frozen minimums: `rows, duration_seconds`
- Binarization, optimizer training, checkpoint selection, rendering, multilingual adaptation, GYU identity adaptation, runtime integration, packaging, and OpenUtau work were not started.

The frozen source gate required at least 200 accepted rows and 1,800 seconds. Only 88 rows and 1028.191 seconds passed all row-level evidence. Identity or acoustic training cannot repair source labels that failed qualification, so the mandatory early stop was applied without changing a threshold.

## Frozen source and evidence

- Dataset: `GTSinger/GTSinger` at `4426c862beed558b7e1cb8a4dce7e8c0c83bb208`
- License: `CC-BY-NC-SA-4.0`; local non-commercial experiment only
- Selection: Korean / KO-Soprano-2 / Control_Group
- Label status: `dataset_metadata_plus_measured_whisper_and_target_conditioned_mms_ctc`
- Human verified: false
- Accepted compact manifest: `data/manifests/gtsinger_ko_source_qualified.jsonl`
- Compact summary: `artifacts/reports/gtsinger_ko_source_qualification/summary.json`
- Full local row evidence: `data/external/work/gtsinger_ko_source_qualification/all_rows.jsonl`
- Local source cache: `data/external/raw/gtsinger-lfs/`
- Original project recordings under `data/source/`: unchanged and uncommitted

Row rejection counts: `{"whisper": 226, "duration": 50, "clipping": 10}`. Accepted stress coverage was fast=69, high-register=40, sustained=62, and large-interval=61; these do not override the failed row-count and duration minimums.

## Environment

- Python 3.11.14; PyTorch 2.11.0+cu130; CUDA build 13.0
- GPU: NVIDIA GB10; reported total memory 128452014080 bytes
- System memory: 128452014080 bytes
- Disk at evidence freeze: 23316504576 bytes free of 982819848192
- DiffSinger checkout: `753b7cc622aadf802b3145d7bb8f7df4afa213c4`
- Tools: `{"torchaudio": "2.11.0", "transformers": "5.8.1", "numpy": "2.3.5", "scipy": "1.17.1", "soundfile": "0.13.1"}`

## Status boundary

- This is not a trained Korean foundation and not a neural GYU SVS package.
- No generated WAV is presented as a usable singer.
- Production renderer, RC7/RC8 decisions, package configuration, and OpenUtau paths remain unchanged.
- Public release remains unauthorized; GTSinger-derived work is governed by CC BY-NC-SA 4.0.

## Next valid requirement

The next valid path is a new rights-controlled, score-native GYU recording corpus with independently verified lyrics, phonemes, durations, and scores that satisfies the same frozen source minimums. Lowering the gate or training on the rejected rows is not an acceptable workaround.
