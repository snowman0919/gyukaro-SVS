#!/usr/bin/env python3
from __future__ import annotations
import json,shutil,time
from pathlib import Path
from gyu_singer.inference.v09 import GyuSingerV09Renderer
def main():
 rc4,fixed,warp,out=map(Path,("artifacts/reports/rc5_isolation","artifacts/reports/rc5_candidate_core","artifacts/reports/rc5_latent_timing","artifacts/reports/rc5_selected_32"));shutil.rmtree(out,ignore_errors=True);out.mkdir(parents=True);matrix=json.loads((rc4/"matrix.json").read_text());renderer=GyuSingerV09Renderer("data/processed/master/216.wav",root=Path.cwd());renderer.omnivoice.close();rows=[]
 try:
  for case in ("ko_neutral","en","rapid_ko","large_interval_ko"):
   directory=out/case;directory.mkdir();body={"source":str(rc4/case/"production_adapted_source.wav"),"f0_npy":str(fixed/case/"canonical_f0.npy"),"identity_npy":str(rc4/case/"identity.npy"),"style_npy":str(rc4/case/"style.npy"),"n_steps":32,"cfg":1.5,"seed":21}
   if case in {"en","rapid_ko"}:body|={"content_warp_npy":str(warp/case/"content_warp.npy"),"content_warp_strength":.25 if case=="en" else 1.0}
   target=directory/"selected.wav";started=time.perf_counter();renderer.soulx.request(body|{"output":str(target)});rows.append({"case":case,"path":str(target),"score":matrix["cases"][case]["score"],"render_seconds":round(time.perf_counter()-started,3)})
  (out/"manifest.json").write_text(json.dumps({"status":"rendered_not_reviewed","decoder":{"precision":"fp32","steps":32,"cfg":1.5},"rows":rows},indent=2)+"\n")
 finally:renderer.close()
if __name__=="__main__":main()
