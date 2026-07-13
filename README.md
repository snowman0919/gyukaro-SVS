# GYU Singer

Experimental personalized singing renderer from authorized GYU recordings. The default backend is a GYU-conditioned MOSS-Nano SFT checkpoint with explicit score pitch/time control. Korean SFT is real; English and Japanese are foundation-model cross-lingual experiments, not verified GYU-language singing.

```sh
python scripts/index_recordings.py
python scripts/validate_dataset.py
python scripts/prepare_svs_data.py
PYTHONPATH=src python -m gyu_singer.cli --backend neural render examples/smoke.json --output out.wav
```

See `docs/v1_report.md` for the packaged checkpoint, reproducible smoke test, and exact limitations.
