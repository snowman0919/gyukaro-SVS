#!/usr/bin/env python3
from pathlib import Path
from gyu_singer.inference.v09 import GyuSingerV09Renderer
def main():
 out=Path("artifacts/reports/rc5_rapid_ctc32");out.mkdir(parents=True,exist_ok=True);r=GyuSingerV09Renderer("data/processed/master/216.wav",root=Path.cwd());r.omnivoice.close()
 try:r.soulx.request({"source":"artifacts/reports/rc5_isolation/rapid_ko/production_adapted_source.wav","f0_npy":"artifacts/reports/rc5_ctc_voicing/rapid_ko/ctc_voiced_f0.npy","identity_npy":"artifacts/reports/rc5_isolation/rapid_ko/identity.npy","style_npy":"artifacts/reports/rc5_isolation/rapid_ko/style.npy","n_steps":32,"cfg":1.5,"seed":21,"output":str(out/"rapid_ctc32.wav")})
 finally:r.close()
if __name__=="__main__":main()
