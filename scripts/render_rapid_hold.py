#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,soundfile as sf
from gyu_singer.inference.content_timing import latent_content_hold
from gyu_singer.inference.v09 import GyuSingerV09Renderer
def main():
 out=Path("artifacts/reports/rc5_rapid_hold");out.mkdir(parents=True,exist_ok=True);source=Path("artifacts/reports/rc5_isolation/rapid_ko/production_adapted_source.wav");alignment=json.loads(Path("artifacts/reports/rc5_content_timing/rapid_ko/alignment.json").read_text());warp=latent_content_hold(alignment,sf.info(source).duration,len(np.load("artifacts/reports/rc5_candidate_core/rapid_ko/canonical_f0.npy")));np.save(out/"warp.npy",warp);r=GyuSingerV09Renderer("data/processed/master/216.wav",root=Path.cwd());r.omnivoice.close()
 try:r.soulx.request({"source":str(source),"f0_npy":"artifacts/reports/rc5_candidate_core/rapid_ko/canonical_f0.npy","content_warp_npy":str(out/"warp.npy"),"identity_npy":"artifacts/reports/rc5_isolation/rapid_ko/identity.npy","style_npy":"artifacts/reports/rc5_isolation/rapid_ko/style.npy","n_steps":64,"cfg":2.0,"seed":21,"output":str(out/"rapid.wav")})
 finally:r.close()
if __name__=="__main__":main()
