#!/usr/bin/env python3
from __future__ import annotations
import json,shutil
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
CORE={"ko_neutral":("artifacts/reports/rc5_selected_32/ko_neutral/selected.wav","artifacts/reports/rc5_candidate_core/ko_neutral/canonical_f0.npy"),"en":("artifacts/reports/rc5_selected_32/en/selected.wav","artifacts/reports/rc5_candidate_core/en/canonical_f0.npy"),"rapid_ko":("artifacts/reports/rc5_rapid_warp64_cfg15/rapid.wav","artifacts/reports/rc5_candidate_core/rapid_ko/canonical_f0.npy"),"large_interval_ko":("artifacts/reports/rc5_candidate_core/large_interval_ko/fixed_full.wav","artifacts/reports/rc5_candidate_core/large_interval_ko/canonical_f0.npy")}
def copy48(source,target):
 audio,rate=sf.read(source,dtype="float32");audio=resample_poly(audio,2,1) if rate==24000 else audio;gain=min(1.,.97/max(float(np.max(np.abs(audio))),1e-8));sf.write(target,audio*gain,48000,subtype="PCM_24");return {"path":str(target),"sample_rate":48000,"duration":sf.info(target).duration,"safety_gain":round(gain,6)}
def main():
 old=Path("artifacts/reports/rc5_stress_candidate");root=Path("artifacts/reports/rc5_stress_candidate3");shutil.rmtree(root,ignore_errors=True);listen=root/"listening";listen.mkdir(parents=True);files={case:copy48(Path(source),listen/f"{case}.wav")|{"target_f0":target,"source_candidate":source} for case,(source,target) in CORE.items()};prior=json.loads((old/"manifest.json").read_text())
 for case in ("ko_breathy","ko_energetic","ja","sustained_ko","phrase_boundary"):files[case]=copy48(Path(prior["files"][case]["path"]),listen/f"{case}.wav")|{k:v for k,v in prior["files"][case].items() if k in {"score","style","source"}}
 report={"status":"candidate_human_listening_pending","name":"RC5 audio-quality candidate 3 (not a tag or release)","decoder_policy":{"ko_en":"FP32 32 steps CFG 1.5","rapid":"FP32 64 steps CFG 1.5","large_interval":"FP32 64 steps CFG 2.0"},"timing_policy":{"standard":"canonical frontend/OpenUtau voicing","en":"CTC hidden timing strength 0.25","rapid":"CTC hidden timing strength 1.0"},"files":files};(root/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__":main()
