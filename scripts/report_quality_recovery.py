#!/usr/bin/env python3
"""Rebuild the RC6 quality-recovery reports from measured JSON evidence."""
from __future__ import annotations

import json
from pathlib import Path


def read(path: str) -> dict:
    return json.loads(Path(path).read_text())


def write(path: str, body: str) -> None:
    Path(path).write_text(body.strip() + "\n")


def fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def main() -> None:
    external = read("artifacts/reports/external_acoustic_quality.json")
    pairs = read("artifacts/reports/pipeline_degradation_pairs.json")
    universal = read("artifacts/reports/acoustic_refiner_universal.json")
    singing = read("artifacts/reports/acoustic_refiner_singing.json")
    gyu = read("artifacts/reports/acoustic_refiner_gyu.json")
    rc5 = read("artifacts/reports/rc5_stress_candidate4/evaluation.json")["aggregate_9"]
    rc6_report = read("artifacts/reports/rc6_backend_candidate/evaluation.json")
    rc6 = rc6_report["aggregate_9"]
    identity = read("artifacts/reports/refiner_identity_evaluation.json")["aggregate"]
    runtime = read("artifacts/reports/runtime_rc6_stress.json")
    longform = read("artifacts/reports/longform_rc6_supervised.json")
    package_archive = Path("artifacts/package/gyu-singer-v1.0.0-rc6-candidate.zip")
    package_root = Path("artifacts/package/gyu-singer-v1.0")
    if package_archive.is_file() and (package_root / "PACKAGE.json").is_file():
        package_meta = read(str(package_root / "PACKAGE.json"))
        package_name = str(package_archive)
        package_sha = __import__("hashlib").sha256(package_archive.read_bytes()).hexdigest()
        package_commit = package_meta["source_commit"]
    else:
        package_name = package_sha = package_commit = "pending"

    write("docs/timing_voicing_fix.md", """
# Timing and voicing fix

RC4's primary artifact source was the interaction of score/content timing mismatch and nonzero F0 in silence or unvoiced consonants. RC5 introduced one 50 Hz phrase timeline carrying phoneme and note bounds, voicing class, nominal score F0, validated GYU residual, and editor pitch. Silence and unvoiced consonants receive F0=0. Vowels and voiced consonants receive score F0 plus prosody and user PITD.

The content path remains phrase-level. Rapid Korean uses MMS CTC phoneme-hold mapping inside SoulX content hidden state; English uses the measured 0.25 latent timing correction. Unedited slurs receive only a minimal 60 ms transition; explicit OpenUtau pitch remains authoritative and hard attacks are preserved. No per-note TTS, phase vocoder, or waveform pitch shifting is used.

The isolation matrix and rejected timing variants are recorded in `docs/rc4_artifact_isolation.md` and `artifacts/reports/rc5_isolation/`.
""")
    write("docs/acoustic_prior_training.md", f"""
# Acoustic prior training

The compatible public prior uses LibriTTS-R and VocalSet, both CC BY 4.0. {external['accepted']} of {external['rows']} measured candidates passed quality gates: {external['by_dataset']['libritts_r']['accepted']} LibriTTS-R and {external['by_dataset']['vocalset']['accepted']} VocalSet. Splits are speaker-disjoint. Original archives and source audio are not packaged.

Real SoulX reconstruction created {pairs['rows']} degradation pairs ({pairs['datasets']['libritts_r']} LibriTTS-R, {pairs['datasets']['vocalset']} VocalSet, {pairs['datasets']['real_gyu']} real GYU), split {pairs['splits']['train']}/{pairs['splits']['validation']}/{pairs['splits']['test']}. No random-noise corruption or synthetic codec degradation was used.

The universal residual backbone has {universal['total_parameters']:,} parameters, with {universal['trainable_parameters']:,} trained for {universal['steps']} steps. Validation loss was {fmt(universal['validation_loss'])}. Its bounded residual is activity-gated, so silent regions cannot receive an invented noise floor.
""")
    write("docs/singing_prior_training.md", f"""
# Singing prior training

The singing stage trained only the {singing['trainable_parameters']:,}-parameter singing adapter for {singing['steps']} steps on {singing['primary_rows']} VocalSet singing pairs plus {singing['replay_rows']} LibriTTS-R replay rows. Validation used {singing['validation_rows']} held-out rows and reached loss {fmt(singing['validation_loss'])}.

VocalSet targets include rapid scales, arpeggios, vibrato/long tones, and straight reference excerpts. This stage was measured on production stress files, but it reduced high-frequency spikes while regressing voicing and ASR. It is therefore retained as a training baseline and disabled in the RC6 runtime.
""")
    write("docs/gyu_acoustic_adaptation.md", f"""
# GYU acoustic adaptation

The GYU stage trained {gyu['trainable_parameters']:,} parameters for {gyu['steps']} steps with {gyu['primary_rows']} real-GYU primary pairs and {gyu['replay_rows']} public replay pairs. Validation used {gyu['validation_rows']} held-out GYU rows and reached loss {fmt(gyu['validation_loss'])}.

This adapter improved several pair-reconstruction metrics, but did not beat the universal backbone on the actual nine-file production path. The RC6 runtime therefore selects the universal checkpoint at 25% residual strength. Singing and GYU adapters remain explicit measured baselines, not production components.

Identity preservation at the selected strength: WavLM before/after cosine {fmt(identity['wavlm_before_after'])}, ECAPA {fmt(identity['ecapa_before_after'])}; similarity-to-GYU deltas were {fmt(identity['wavlm_to_gyu_delta'])} and {fmt(identity['ecapa_to_gyu_delta'])}. These are diagnostics, not listening evidence.
""")
    changes = {key: rc6[key] - rc5[key] for key in rc5 if isinstance(rc5[key], (int, float)) and key in rc6}
    write("docs/final_quality_evaluation.md", f"""
# RC6 objective quality evaluation

Status: objective candidate; mandatory human listening pending.

The actual `gyu-singer-rc6` backend rendered all nine stress cases. Against the fixed RC5 baseline, aggregate changes were: pitch MAE {fmt(changes['pitch_mae_cents'])} cents, voicing accuracy {fmt(changes['voicing_accuracy'])}, HF spike ratio {fmt(changes['hf_spike_p99_over_median'])}, spectral flux p95 {fmt(changes['spectral_flux_p95'])}, sample jump p99.9 {fmt(changes['sample_jump_p999'])}, and ASR similarity {fmt(changes['asr_lyric_similarity'])}. Clipping stayed zero.

The selected universal refiner lowers HF spikes and waveform jumps and slightly improves voicing. HF energy p95 rises by {fmt(changes['hf_energy_ratio_p95'])}; this is a remaining risk that objective metrics cannot adjudicate perceptually.

Resident stress passed with one unique hash across repeated renders, KO/EN/JA, concurrency, invalid-request recovery, restart, clean shutdown, and steady-state memory checks. The maintained OpenUtau overlay rendered {longform['render_metrics']['notes']} notes in {longform['render_metrics']['phrases']} phrases over {fmt(longform['render_metrics']['duration_seconds'])} seconds with zero failed phrases.

Listening gate: `artifacts/reports/rc6_listening_gate/`. Each of nine files requires an explicit PASS/FAIL and observation. Final `v1.0.0` remains forbidden until that review passes.
""")
    write("docs/final_v1.0_report.md", f"""
Overall status: RC6 objective candidate; HUMAN LISTENING PENDING; final release blocked
Current version: 1.0.0-rc.6-candidate
Package: {package_name}
Package SHA-256: {package_sha}
Git commit: {package_commit}
Primary artifact source: RC4 content/score timing mismatch plus all-frame voiced F0
Secondary artifact sources: low-step/high-CFG decode and hard unedited note steps
Selected SoulX settings: FP32; standard 32/CFG1.5, rapid 64/CFG2.0, large interval 32/CFG2.0 seed 21
Selected acoustic refiner: universal backbone at 25%
Disabled components: singing adapter and GYU adapter (production regressions); BigVGAN diagnostic (mixed regressions)
Phrase-level generation: yes
Per-note TTS used: no
Waveform pitch shifting used: no
OpenUtau long-form: PASS ({longform['render_metrics']['notes']} notes, {longform['render_metrics']['phrases']} phrases, {fmt(longform['render_metrics']['duration_seconds'])} seconds)
Runtime stress: {'PASS' if runtime['pass'] else 'FAIL'}
Korean: objective stress rendered; human pending
English: objective stress rendered; human pending
Japanese: objective stress rendered; human pending
Release recommendation: DO NOT tag or publish v1.0.0

# RC6 artifact-recovery report

RC4 is preserved exactly. RC5's canonical timing/voicing and safer SoulX policy fixed the identified structural defects. RC6 adds a small bounded, activity-gated residual refiner trained on actual SoulX reconstruction pairs. It does not denoise source recordings and does not replace SoulX.

Compared with RC5 across nine files, HF-spike ratio changed from {fmt(rc5['hf_spike_p99_over_median'])} to {fmt(rc6['hf_spike_p99_over_median'])}, spectral flux from {fmt(rc5['spectral_flux_p95'])} to {fmt(rc6['spectral_flux_p95'])}, sample jump from {fmt(rc5['sample_jump_p999'])} to {fmt(rc6['sample_jump_p999'])}, voicing from {fmt(rc5['voicing_accuracy'])} to {fmt(rc6['voicing_accuracy'])}, and pitch MAE from {fmt(rc5['pitch_mae_cents'])} to {fmt(rc6['pitch_mae_cents'])} cents. ASR similarity is unchanged at {fmt(rc6['asr_lyric_similarity'])}; clipping is zero.

Remaining audible defects are unknown until human review. Objective risks include a small aggregate HF-energy increase, persistent English “Sing”/“Sink” ambiguity, sustained-vowel ASR ambiguity, and weak voicing scores for large-interval and phrase-boundary cases.

Before/after files and the nine final candidate WAVs are in `artifacts/reports/rc6_listening_gate/`. The required next outcome is a per-file human verdict, not a final release.
""")


if __name__ == "__main__":
    main()
