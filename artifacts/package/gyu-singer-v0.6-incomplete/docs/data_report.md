# GYU recording report

132 sequential ALAC `.m4a` recordings, indices 106–237, decode to 48 kHz mono PCM S24LE masters. Total duration: 1,740.117 s (29.002 min); source duration 3.669–41.216 s; median detected F0 136.075 Hz; median voiced ratio 0.5031; 2 peak-clipping flags. Per-file measurements are in `data/manifests/real_recordings.jsonl`.

Local Whisper large-v3-turbo transcribed every recording. Source order plus the embedded script PDF reconstructs blocks as A 106–117, B 118–139, C 140–157, D 158–211, E 212–221, F 222–236, G 237. 76 phrase segments have explicit script correspondence at confidence 0.99; 56 exercise/free-take mappings remain marked for review. Evidence: `asr_transcripts.jsonl`, `script_alignment.jsonl`, and `artifacts/reports/alignment_review.md`.

`neural_supervision.jsonl` contains 76 high-confidence C/D/E real anchors; `moss_sft_raw.jsonl` selects the 64 real singing D/E takes used by SFT. No generated teacher sample is represented as real data.
