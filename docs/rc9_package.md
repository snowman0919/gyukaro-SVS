# RC9 package gate

Status: packaging is intentionally blocked until the RC9 full-song listening set receives an explicit human PASS.

`scripts/package_rc9.py` is prepared to create `artifacts/package/gyu-singer-rc9/`, `gyu-singer-rc9.zip`, and `SHA256SUMS`. It refuses to run unless the objective song gate, identity gate, and human acceptance recorded in `docs/final_rc9_report.md` all pass.

The package allowlist includes the RC9 runtime, project-trained checkpoints, native OpenUtau overlay, installer/launcher/downloader, original GYU demonstration projects, licenses, model metadata, and compact evidence. It explicitly excludes the local reference mix, off-vocal, vocal estimate, lyrics, reconstructed reference USTX, phrase requests, and every derived reference-song WAV.

After human acceptance the clean test is:

```text
unpack outside the repository
→ install with pinned cache copy
→ render KO/EN/JA install smoke
→ build the pinned official OpenUtau checkout
→ start the packaged gyu-singer-rc9 resident server
→ render/export the packaged original long-form USTX through OpenUtau
→ verify archive/checkpoint/OpenUtau hashes and WAV properties
```

No final `v1.0.0` tag or release is authorized by RC9.
