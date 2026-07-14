Overall status: achieved — experimental acceptance gates satisfied; quality limits retained below.
Current version: gyu-singer-v0.6
Package: gyu-singer-v0.6-experimental
Package SHA-256: 01c2d945e236396788af2ae92207da8f668151686e837f56950ac00f0a83221e
Git commit: 7a2be8c0872ce75c6678990b7b891a83580b33fe
Manual verified score phrases: 24 independently transcribed, target-RMVPE-independent KO phrases; PyIN/script/spectrogram/CTC review artifacts; not source MIDI or human-certified notation
Independent prosody evaluation: completed on those 24 phrases; v0.5 MAE 134.55c, experimental v0.6 135.18c; v0.6 improves onset/transition but is not a consistent win
Teacher internal paired rows: 249 Fish/MOSS rows from 102 unique semantic groups; train/validation/test 191/32/26; KO 128, EN 92, JA 29
Fish internal representation: Fish-S2-Pro-DAC.encoder_hidden, mean pooled
MOSS internal representation: MOSS-Audio-Tokenizer-Nano.encoder_hidden_states, mean pooled
Higgs internal representation: no stable official internal hook; 180 waveform-level auxiliary rows only
Shared identity space: 64D trust-weighted Barlow cross-view space; held-out cross-teacher cosine 0.94029, teacher leakage 0.50000, language clustering 0.42308
Identity adapter affects final audio: yes; same-score 6-phrase ablation changes WavLM/ECAPA output metrics, while quality gain remains small (WavLM mean +0.00196, 95% −0.00158..+0.00561)
Latent acoustic-style adapter: yes; gated FiLM at SoulX `gt_decoder_inp`; dark latent-vs-spectral-only WavLM L2 0.002592/0.002966 on two KO phrases; preset semantics remain weakly calibrated
Pseudo singing used: 200 targeted ACE-Step candidates, 45 quality-gated accepted low-trust rows (KO 21, EN 14, JA 10); not used as real-GYU prosody targets
Phrase-level generation: SoulX phrase decode, 48 kHz WAV
Per-note TTS used: no
Waveform pitch shifting used: no
v0.5 fallback used by v0.6 backend: no; v0.5 prosody controller is deliberately retained because experimental v0.6 prosody did not win consistently
OpenUtau readiness: protocol/USTX field mapping documented; native editor integration deferred
Korean: real-GYU prosody supervision and independent evaluation; production controller v0.5
English: generic singing prosody plus GYU identity/style adaptation; no real-GYU EN singing claim
Japanese: generic singing prosody plus GYU identity/style adaptation; no real-GYU JA singing claim

## Acceptance evidence

The independent score set is not built by quantizing RMVPE target F0. Fish and MOSS contribute actual internal hidden states over a non-duplicated 102-group corpus. The identity vector and latent style vector are injected inside the real SoulX phrase path and causal ablations change final WAV embeddings. KO/EN/JA phrase rendering uses one phrase-level path only. The packaged runtime passed a clean-unpack smoke with the pinned external cache and compatible pinned SoulX Python; it emitted a 48 kHz mono WAV.

## Limits

This is an experimental singer, not proof of a high-quality personalized commercial voice. The manual score set is independently transcribed but not human-certified notation. v0.6 prosody is retained as a measured baseline, not production default, because its aggregate independent metrics are mixed. Identity and style conditioning demonstrably reach output, but their identity/style quality gains are weak. Higgs has no internal-representation contribution. English/Japanese identity rendering does not claim personalized real-GYU prosody.

Reproduce key evidence with `python scripts/validate_dataset.py`, `PYTHONPATH=src pytest -q`, `python scripts/evaluate_independent_prosody.py`, and `PYTHONPATH=src python scripts/evaluate_v06_identity_style_ablation.py --metrics-only` after the render artifacts exist.
