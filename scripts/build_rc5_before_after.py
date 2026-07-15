#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
WINDOWS={"ko_neutral":(0,3),"en":(0,3),"rapid_ko":(0,2),"large_interval_ko":(0,4.8)}
def write_clip(source:Path,target:Path,start:float,duration:float):
 audio,rate=sf.read(source,dtype="float32");audio=resample_poly(audio,2,1) if rate==24000 else audio;begin,end=round(start*48000),round((start+duration)*48000);sf.write(target,audio[begin:end],48000,subtype="PCM_24");return {"path":str(target),"sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"duration":sf.info(target).duration}
def main():
 root=Path("artifacts/reports/rc5_stress_candidate4/before_after");shutil.rmtree(root,ignore_errors=True);root.mkdir();matrix=json.loads(Path("artifacts/reports/rc5_isolation/matrix.json").read_text());rows=[]
 for case,(start,duration) in WINDOWS.items():
  before=write_clip(Path(matrix["cases"][case]["matrix"]["F"]["path"]),root/f"{case}_RC4.wav",start,duration);after=write_clip(Path(f"artifacts/reports/rc5_stress_candidate4/listening/{case}.wav"),root/f"{case}_candidate4.wav",start,duration);rows.append({"case":case,"window_start":start,"window_duration":duration,"rc4":before,"candidate4":after})
 (root/"manifest.json").write_text(json.dumps({"status":"human_listening_pending","same_windows":True,"rows":rows},indent=2)+"\n")
if __name__=="__main__":main()
