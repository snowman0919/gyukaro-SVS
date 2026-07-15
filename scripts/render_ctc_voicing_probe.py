#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, time
from pathlib import Path
import numpy as np
from gyu_singer.inference.content_timing import ctc_voicing_mask
from gyu_singer.inference.v09 import GyuSingerV09Renderer
def main():
 rc4,ctc,out=map(Path,("artifacts/reports/rc5_isolation","artifacts/reports/rc5_content_timing","artifacts/reports/rc5_ctc_voicing")); shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True); matrix=json.loads((rc4/"matrix.json").read_text()); renderer=GyuSingerV09Renderer("data/processed/master/216.wav",root=Path.cwd()); renderer.omnivoice.close(); report={"status":"rendered_not_reviewed","method":"CTC source-phone voicing mask; score-time pitch preserved","cases":{}}
 try:
  for case,data in matrix["cases"].items():
   directory=out/case; directory.mkdir(); base=np.load(rc4/case/"production_f0.npy"); alignment=json.loads((ctc/case/"alignment.json").read_text()); mask=ctc_voicing_mask(alignment,len(base)/50,len(base)); contour=directory/"ctc_voiced_f0.npy"; np.save(contour,base*mask); source=rc4/case/"production_adapted_source.wav"; target=directory/"ctc_voicing_full.wav"; started=time.perf_counter(); renderer.soulx.request({"source":str(source),"f0_npy":str(contour),"identity_npy":str(rc4/case/"identity.npy"),"style_npy":str(rc4/case/"style.npy"),"n_steps":64,"cfg":2.0,"seed":21,"output":str(target)}); report["cases"][case]={"score":data["score"],"target_voiced_ratio":round(float(mask.mean()),4),"path":str(target),"render_seconds":round(time.perf_counter()-started,3)}
  (out/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 finally: renderer.close()
if __name__=="__main__": main()
