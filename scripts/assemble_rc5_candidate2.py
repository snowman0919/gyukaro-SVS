#!/usr/bin/env python3
from __future__ import annotations
import json,shutil
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
CORE={"ko_neutral":("artifacts/reports/rc5_selected_32/ko_neutral/selected.wav","artifacts/reports/rc5_candidate_core/ko_neutral/canonical_f0.npy"),"en":("artifacts/reports/rc5_selected_32/en/selected.wav","artifacts/reports/rc5_candidate_core/en/canonical_f0.npy"),"rapid_ko":("artifacts/reports/rc5_rapid_ctc32/rapid_ctc32.wav","artifacts/reports/rc5_ctc_voicing/rapid_ko/ctc_voiced_f0.npy"),"large_interval_ko":("artifacts/reports/rc5_candidate_core/large_interval_ko/fixed_full.wav","artifacts/reports/rc5_candidate_core/large_interval_ko/canonical_f0.npy")}
def copy48(source:Path,target:Path)->dict:
 audio,rate=sf.read(source,dtype="float32");audio=resample_poly(audio,2,1) if rate==24000 else audio;peak=float(np.max(np.abs(audio)));gain=min(1.0,.97/max(peak,1e-8));audio*=gain;sf.write(target,audio,48000,subtype="PCM_24");return {"path":str(target),"sample_rate":48000,"duration":sf.info(target).duration,"safety_gain":round(gain,6)}
def main():
 old=Path("artifacts/reports/rc5_stress_candidate");root=Path("artifacts/reports/rc5_stress_candidate2");shutil.rmtree(root,ignore_errors=True);listening=root/"listening";listening.mkdir(parents=True);files={}
 for case,(source,target_f0) in CORE.items():files[case]=copy48(Path(source),listening/f"{case}.wav")|{"target_f0":target_f0,"source_candidate":str(source)}
 old_manifest=json.loads((old/"manifest.json").read_text())
 for case in ("ko_breathy","ko_energetic","ja","sustained_ko","phrase_boundary"):files[case]=copy48(Path(old_manifest["files"][case]["path"]),listening/f"{case}.wav")|{k:v for k,v in old_manifest["files"][case].items() if k in {"score","style","source"}}
 report={"status":"candidate_human_listening_pending","name":"RC5 audio-quality candidate 2 (not a tag or release)","decoder_policy":{"standard":"FP32 32 steps CFG 1.5","large_interval":"FP32 64 steps CFG 2.0"},"timing_policy":{"standard":"canonical frontend/OpenUtau voicing","en":"CTC hidden timing strength 0.25","rapid":"CTC source-phone voicing"},"files":files};(root/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");print(json.dumps({"root":str(root),"cases":len(files)},indent=2))
if __name__=="__main__":main()
