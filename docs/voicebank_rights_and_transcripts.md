# Voicebank Rights and Transcript Trust

The rights manifest records source type, owner or authorized user, allowed use, redistribution permission, languages, known scripts, recording environment, consent/provenance notes, and an explicit permission affirmation. Missing or false affirmation stops before workspace creation.

Transcript precedence is:

1. user-provided exact script
2. verified metadata
3. user-corrected draft with accepted review status
4. automatic STT draft

Automatic STT is never ground truth. Korean automatic output is labeled `untrusted_draft_transcript` and excluded from training until a user accepts a correction. Forced or uniform alignment remains inferred and is classified by confidence; it is never relabeled as manual verification.
