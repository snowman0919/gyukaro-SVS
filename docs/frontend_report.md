# Frontend report

`gyu_singer.frontend.phonemize` supports `ko`, `en`, `ja`, emitting stable IDs, language IDs, syllable/word boundaries and eight language-aware features. Korean decomposes Hangul onset/nucleus/coda. Japanese marks mora, long vowel, geminate and moraic nasal. English uses a rule-based vowel stress proxy; this label is inferred, not lexicon-derived. `build_phrase_frames` assigns each score note its own lyric frames rather than stretching whole phrase text across every note.
