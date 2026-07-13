# GYU Singer

Experimental personalized score renderer from authorized GYU recordings. It is real-GYU pitch-controlled audio, not yet lyric-conditioned trilingual neural SVS.

```sh
python scripts/index_recordings.py
python scripts/validate_dataset.py
python scripts/train.py
PYTHONPATH=src python -m gyu_singer.cli render examples/korean.json --output out.wav
```

See `docs/v1_report.md` for exact limits.
