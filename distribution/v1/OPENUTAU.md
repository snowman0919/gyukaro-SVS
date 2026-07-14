# OpenUtau integration

OpenUtau does not expose an external renderer-registration API at the pinned revision, so the installer builds a maintained fork from official `stakira/OpenUtau@27573ac5c888d927119d5f65a207312d79194b1f`.

The overlay adds:

- `USingerType.GyuSinger` and the `GYU-SINGER` renderer registration;
- a virtual singer whose dummy OTO objects only form phrases and never synthesize audio;
- `GyuSingerRenderer`, which maps multi-note phrases to the resident service;
- native regression tests and three-language example projects.

Mapped editor data: notes, tuning, lyrics, generated phonemes, tempo, final pitch including portamento/vibrato/PITD, dynamics, breathiness, tension, and GYUS style. Cache keys hash the complete request, so relevant edits invalidate only the edited phrase.

Unsupported as separate stable controls: brightness and dedicated vibrato-depth protocol fields. Soft/dark/bright remain experimental relative styles.
