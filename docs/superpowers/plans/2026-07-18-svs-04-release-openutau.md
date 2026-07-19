# SVS-04 Release and OpenUtau Implementation Plan

1. Freeze the 11 release gates and current evidence paths.
2. Test missing gates, Whisper-only lexical evidence, human approval evidence, and current blocked decision.
3. Implement one pure central release decision engine.
4. Test RC8 and RC9 package refusal before filesystem mutation.
5. Implement reproducible singer, phonemizer, model, checkpoint-reference, language, sample, license, evaluation, README, and SHA metadata generation.
6. Make diagnostic output unmistakable and checkpoint-free.
7. Generate a compact release status report from the frozen config.
8. Run focused and full tests, dataset and evidence checks, diagnostic smoke, and diff checks.
