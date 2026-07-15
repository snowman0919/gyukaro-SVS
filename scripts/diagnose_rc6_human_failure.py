#!/usr/bin/env python3
"""Rebuild the RC6 human-failure root-cause report from recorded evidence."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def span(rows: list[dict], key: str) -> list[float]:
    values = [float(row[key]) for row in rows]
    return [round(min(values), 6), round(max(values), 6)]


def main() -> None:
    human = read("artifacts/reports/rc6_listening_gate/human_review.json")
    rapid = read("artifacts/reports/rc5_rapid_decode/evaluation.json")
    interval = read("artifacts/reports/rc5_large_interval_decode/evaluation.json")
    source = (ROOT / "src/gyu_singer/inference/v09.py").read_text()
    timing = (ROOT / "src/gyu_singer/inference/content_timing.py").read_text()
    assert "content_warp_strength\": 1.0 if self._rapid(score)" in source
    assert "latent_content_hold(alignment" in source
    assert "Hold a stable CTC phone-center hidden" in timing

    report = {
        "status": "root_cause_isolated_current_svc_path_rejected",
        "human_release_verdict": human["overall_release_suitability"],
        "final_release_allowed": False,
        "primary_source": (
            "The duration-only OmniVoice phrase and score timeline are reconciled by "
            "post-hoc SoulX content-hidden remapping; rapid mode holds one phone-center "
            "hidden vector at full strength. This destroys natural within-phone motion "
            "and does not provide score-native note/phoneme generation."
        ),
        "secondary_sources": [
            "SoulX SVC is seed-sensitive on rapid and large-interval material.",
            "Explicit F0 conditioning does not force source articulation and note timing to agree.",
            "The residual acoustic refiner cannot recover missing or wrong content/timbre structure.",
        ],
        "rapid_decode_sweep": {
            "variants": len(rapid["rows"]),
            "status": rapid["status"],
            "asr_similarity_range": span(rapid["rows"], "asr_lyric_similarity"),
            "hf_spike_range": span(rapid["rows"], "hf_spike_p99_over_median"),
            "voicing_accuracy_range": span(rapid["rows"], "voicing_accuracy"),
        },
        "large_interval_decode_sweep": {
            "variants": len(interval["rows"]),
            "observed_voiced_ratio_range": span(interval["rows"], "observed_voiced_ratio"),
            "target_voiced_ratio_range": span(interval["rows"], "target_voiced_ratio"),
            "pitch_mae_cents_range": span(interval["rows"], "pitch_mae_cents"),
            "asr_similarity_range": span(interval["rows"], "asr_lyric_similarity"),
        },
        "rejected_next_actions": [
            "generic denoising",
            "another SoulX steps/CFG sweep",
            "blind whole-system retraining",
            "stronger post-hoc residual filtering",
        ],
        "required_next_probe": (
            "A score-native phrase SVS acoustic path whose phoneme durations, note onsets, "
            "voicing and F0 are direct model inputs; validate rapid and large intervals before GYU adaptation."
        ),
    }
    out = ROOT / "artifacts/reports/rc6_human_failure_diagnosis.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    r = report
    body = f"""# RC6 human-listening failure diagnosis

Status: **FAIL — RC6 is frozen and final v1.0.0 remains forbidden.**

The primary failure is the current phrase-SVC construction, not the final residual refiner. OmniVoice receives only the whole lyric and total duration. Rapid mode then maps the resulting phrase to the score by holding one CTC phone-center SoulX hidden vector across every target phoneme window at full strength. That operation explains the heard fade/staccato joins and rapid-case voice drift: it changes the content/timbre representation while removing natural within-phone evolution.

This is supported by the measured sweeps, not inferred from listening alone:

- Rapid: {r['rapid_decode_sweep']['variants']} decoder variants; status `{r['rapid_decode_sweep']['status']}`; ASR similarity {r['rapid_decode_sweep']['asr_similarity_range']}; HF-spike ratio {r['rapid_decode_sweep']['hf_spike_range']}.
- Large interval: {r['large_interval_decode_sweep']['variants']} decoder variants; target voiced ratio {r['large_interval_decode_sweep']['target_voiced_ratio_range']}, observed {r['large_interval_decode_sweep']['observed_voiced_ratio_range']}; pitch MAE {r['large_interval_decode_sweep']['pitch_mae_cents_range']} cents.

Increasing SoulX steps, changing CFG/seed, generic denoising, or strengthening the post-filter is rejected. The next gate is a score-native phrase SVS probe in which phoneme duration, note onset, voicing, and F0 are direct acoustic-model inputs. It must first beat RC6 on rapid and large-interval timing, identity stability, and audible joins before any GYU adaptation or release-candidate packaging.
"""
    (ROOT / "docs/rc6_human_failure_diagnosis.md").write_text(body)


if __name__ == "__main__":
    main()
