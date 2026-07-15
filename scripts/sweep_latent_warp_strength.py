#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, time
from pathlib import Path
from gyu_singer.inference.v09 import GyuSingerV09Renderer
def main():
 rc4,fixed,warp,out=map(Path,("artifacts/reports/rc5_isolation","artifacts/reports/rc5_candidate_core","artifacts/reports/rc5_latent_timing","artifacts/reports/rc5_latent_strength")); shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True); matrix=json.loads((rc4/"matrix.json").read_text()); renderer=GyuSingerV09Renderer("data/processed/master/216.wav",root=Path.cwd()); renderer.omnivoice.close(); rows=[]
 try:
  for case in ("en","rapid_ko"):
   directory=out/case; directory.mkdir()
   for strength in (.25,.5,.75):
    target=directory/f"strength_{strength:g}.wav"; started=time.perf_counter(); renderer.soulx.request({"source":str(rc4/case/"production_adapted_source.wav"),"f0_npy":str(fixed/case/"canonical_f0.npy"),"content_warp_npy":str(warp/case/"content_warp.npy"),"content_warp_strength":strength,"identity_npy":str(rc4/case/"identity.npy"),"style_npy":str(rc4/case/"style.npy"),"n_steps":64,"cfg":2.0,"seed":21,"output":str(target)}); rows.append({"case":case,"strength":strength,"path":str(target),"render_seconds":round(time.perf_counter()-started,3),"score":matrix["cases"][case]["score"]})
  (out/"manifest.json").write_text(json.dumps({"status":"bounded_sweep","rows":rows},indent=2)+"\n")
 finally:renderer.close()
if __name__=="__main__":main()
