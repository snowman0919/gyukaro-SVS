#!/usr/bin/env python3
"""Build the nine-case post-RC4 listening candidate without publishing a release."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly
from gyu_singer.inference.v09 import GyuSingerV09Renderer

NEW = {"ko_breathy": ("examples/quality_ko.json", "breathy"), "ko_energetic": ("examples/quality_ko.json", "energetic"), "ja": ("examples/quality_ja.json", "neutral"), "sustained_ko": ("examples/review_sustain_ko.json", "neutral"), "phrase_boundary": ("examples/review_phrase_boundary_ko.json", "neutral")}
def main():
 root=Path("artifacts/reports/rc5_stress_candidate"); shutil.rmtree(root,ignore_errors=True); listening=root/"listening"; listening.mkdir(parents=True); selected={
  "ko_neutral":"artifacts/reports/rc5_candidate_core/ko_neutral/fixed_full.wav",
  "en":"artifacts/reports/rc5_latent_strength/en/strength_0.25.wav",
  "rapid_ko":"artifacts/reports/rc5_latent_timing/rapid_ko/latent_timing_full.wav",
  "large_interval_ko":"artifacts/reports/rc5_candidate_core/large_interval_ko/fixed_full.wav"}
 report={"status":"candidate_human_listening_pending","name":"RC5 audio-quality candidate (not a tag or release)","selection":{"standard":"canonical score/phone F0 + FP32 SoulX 64 steps CFG 2.0","en":"0.25 CTC hidden timing correction","rapid":"1.0 CTC hidden timing correction"},"files":{}}
 for case,path in selected.items():
  target=listening/f"{case}.wav"; audio,rate=sf.read(path,dtype="float32"); audio=resample_poly(audio,2,1) if rate==24000 else audio; sf.write(target,audio,48000,subtype="PCM_24"); info=sf.info(target); report["files"][case]={"path":str(target),"sample_rate":info.samplerate,"duration":info.duration,"source":"measured core candidate"}
 renderer=GyuSingerV09Renderer("data/processed/master/216.wav",root=Path.cwd())
 try:
  for case,(score_path,style) in NEW.items():
   score=json.loads(Path(score_path).read_text()); score.setdefault("style",{})["preset"]=style; target=listening/f"{case}.wav"; renderer.render_file(score_path if style=="neutral" else _write_score(root,case,score),target); info=sf.info(target); report["files"][case]={"path":str(target),"sample_rate":info.samplerate,"duration":info.duration,"source":"fresh canonical candidate render","score":score_path,"style":style}
 finally: renderer.close()
 (root/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({"output":str(root),"cases":len(report["files"])},indent=2))
def _write_score(root:Path,case:str,score:dict)->Path:
 path=root/f"{case}.json"; path.write_text(json.dumps(score,ensure_ascii=False,indent=2)+"\n"); return path
if __name__=="__main__":main()
