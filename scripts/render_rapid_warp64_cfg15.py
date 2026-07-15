#!/usr/bin/env python3
from pathlib import Path
from gyu_singer.inference.v09 import GyuSingerV09Renderer
def main():
 out=Path("artifacts/reports/rc5_rapid_warp64_cfg15");out.mkdir(parents=True,exist_ok=True);r=GyuSingerV09Renderer("data/processed/master/216.wav",root=Path.cwd());r.omnivoice.close()
 try:r.soulx.request({"source":"artifacts/reports/rc5_isolation/rapid_ko/production_adapted_source.wav","f0_npy":"artifacts/reports/rc5_candidate_core/rapid_ko/canonical_f0.npy","content_warp_npy":"artifacts/reports/rc5_latent_timing/rapid_ko/content_warp.npy","content_warp_strength":1.0,"identity_npy":"artifacts/reports/rc5_isolation/rapid_ko/identity.npy","style_npy":"artifacts/reports/rc5_isolation/rapid_ko/style.npy","n_steps":64,"cfg":1.5,"seed":21,"output":str(out/"rapid.wav")})
 finally:r.close()
if __name__=="__main__":main()
