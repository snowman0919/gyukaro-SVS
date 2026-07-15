# RC4 artifact isolation

RC4 remains frozen at tag `v1.0.0-rc.4`; no final `v1.0.0` tag or release was created. Human listening marked RC4 failed.

## Matrix

`artifacts/reports/rc5_isolation/` contains 4 cases × A–F plus 16/32/64-step, CFG 1.5/2.0/2.5, FP16/FP32 sweeps. Metrics are normalized to 48 kHz. The compact listening directory is `artifacts/reports/rc5_isolation/listening_matrix/`.

Findings:

- RC4 forced nonzero target F0 through every note frame. The generated OmniVoice content and score voicing agreed poorly (KO neutral 0.778, EN 0.565, rapid KO 0.680, large interval KO 0.154).
- Nominal F0 (B) was materially more stable than production F0 (C): aggregate spectral flux 0.229→0.265 and HF-spike ratio 758→1057.
- Identity/style toggles C/D/E/F were close. They are not the primary artifact source.
- 16-step/CFG2.5 was not optimal. The lowest diagnostic proxy was 32-step/CFG1.5 after fair 48 kHz normalization.
- Blind canonical F0 masking, waveform WSOLA correction, ACE score-guide source, full linear hidden warp, and CTC-voicing-only were each measured and rejected where they harmed lyrics or artifacts. Rejected reports remain under `artifacts/reports/rc5_*`.

Primary source: interaction between score/content timing mismatch and the all-frame-voiced production F0. Secondary source: aggressive low-step/high-CFG SoulX decode. Adapters are retained because ablations did not identify them as the primary cause.
