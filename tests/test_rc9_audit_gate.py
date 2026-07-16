from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
from package_rc9 import audited_audio_evidence  # noqa: E402


def test_packaging_requires_free_stt_and_waveform_evidence():
    legacy = {"gates": {}, "waveform_analysis": {}, "phrases": [{}]}
    audited = {
        "gates": {
            "free_stt_present_for_every_phrase": True,
            "free_stt_mean_at_least_0_75": True,
            "free_stt_p10_at_least_0_50": True,
            "waveform_has_no_clipping": True,
            "waveform_peak_below_0_99": True,
        },
        "waveform_analysis": {"peak": .5},
        "phrases": [{"free_asr_transcript_sha256": "hash"}],
    }

    assert not audited_audio_evidence(legacy)
    assert audited_audio_evidence(audited)
