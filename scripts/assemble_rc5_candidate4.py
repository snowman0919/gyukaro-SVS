#!/usr/bin/env python3
import json,shutil
from pathlib import Path
from assemble_rc5_candidate3 import copy48
CORE={"ko_neutral":("artifacts/reports/rc5_selected_32/ko_neutral/selected.wav","artifacts/reports/rc5_candidate_core/ko_neutral/canonical_f0.npy"),"en":("artifacts/reports/rc5_selected_32/en/selected.wav","artifacts/reports/rc5_candidate_core/en/canonical_f0.npy"),"rapid_ko":("artifacts/reports/rc5_rapid_hold/rapid.wav","artifacts/reports/rc5_candidate_core/rapid_ko/canonical_f0.npy"),"large_interval_ko":("artifacts/reports/rc5_large_interval_decode/s32_c2_seed21.wav","artifacts/reports/rc5_candidate_core/large_interval_ko/canonical_f0.npy")}
def main():
 root=Path("artifacts/reports/rc5_stress_candidate4");shutil.rmtree(root,ignore_errors=True);listen=root/"listening";listen.mkdir(parents=True);files={}
 for case,(source,target) in CORE.items():
  local_target=root/f"{case}_target_f0.npy";shutil.copy2(target,local_target);files[case]=copy48(Path(source),listen/f"{case}.wav")|{"target_f0":str(local_target),"source_candidate":source}
 prior=json.loads(Path("artifacts/reports/rc5_stress_candidate/manifest.json").read_text())
 for case in ("ko_breathy","ko_energetic","ja","sustained_ko","phrase_boundary"):files[case]=copy48(Path(prior["files"][case]["path"]),listen/f"{case}.wav")|{k:v for k,v in prior["files"][case].items() if k in {"score","style","source"}}
 (root/"manifest.json").write_text(json.dumps({"status":"candidate_human_listening_pending","name":"RC5 audio-quality candidate 4 (not a tag or release)","decoder_policy":{"ko_en":"FP32 32/CFG1.5","rapid":"FP32 64/CFG2","large":"FP32 32/CFG2 seed21"},"timing_policy":{"standard":"canonical","en":"CTC linear 0.25","rapid":"CTC phoneme hold"},"files":files},ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__":main()
