# GYU-SINGER OpenUtau Character Library (v0.9v)

- Release artifact: `artifacts/package/gyu-singer-openutau-character-library.zip`
- Alias: `artifacts/package/gyu-singer-v0.9-openutau-character.zip`
- SHA-256: `155e89891dfcfacb899e113c0b861b6f91b5ff1a24e23b51afbf8e0a96ad659c`

- Included files (within the zip root):
  - `GYU-SINGER/character.txt`
  - `GYU-SINGER/character.yaml`
  - `GYU-SINGER/README.md`
  - `GYU-SINGER/as/urls.txt` (when present)
  - `GYU-SINGER/portraits/*`
  - `metadata.json`

## Install (official OpenUtau)

1. Unzip the package.
2. Copy `GYU-SINGER` directory into OpenUtau's singer/character directory.
3. In OpenUtau, select singer **GYU-SINGER**.

## Important

This is a metadata-only singer package.

- It provides discovery/presets/portraits for the official OpenUtau GUI.
- It does **not** include a renderer runtime.
- Runtime rendering still uses the v0.9 package path or the maintained OpenUtau fork path documented in `docs/openutau_v0.9.md`.

## Validation

- Structure check: required singer files and portrait assets are present under `GYU-SINGER/`.
- `scripts/package_openutau_character_library.py` is the reproducible generator.
- `python scripts/validate_dataset.py` pass: `132` recordings, `corrupt 0`.

