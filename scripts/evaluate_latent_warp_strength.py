#!/usr/bin/env python3
from __future__ import annotations
import json,os,sys
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor
from evaluate_rc4_artifact_matrix import acoustics,audio16,normalized,pitch
CACHE=Path(os.environ.get("GYU_SINGER_CACHE","data/cache"));sys.path.insert(0,str(CACHE/"soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor
def main():
 root=Path("artifacts/reports/rc5_latent_strength");data=json.loads((root/"manifest.json").read_text());ex=F0Extractor(str(CACHE/"soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),device="cuda",target_sr=24000,hop_size=480,verbose=False);rows=[]
 for row in data["rows"]: rows.append(row|acoustics(Path(row["path"]))|pitch(Path(row["path"]),np.load(Path("artifacts/reports/rc5_candidate_core")/row["case"]/"canonical_f0.npy"),ex))
 del ex;torch.cuda.empty_cache();p=AutoProcessor.from_pretrained(CACHE/"whisper-large-v3-turbo");m=AutoModelForSpeechSeq2Seq.from_pretrained(CACHE/"whisper-large-v3-turbo",dtype=torch.float16).cuda().eval()
 for row in rows:
  score=json.loads(Path(row["score"]).read_text());expected=" ".join(n["lyric"] for n in score["notes"]);x=p(audio16(Path(row["path"])),sampling_rate=16000,return_tensors="pt")
  with torch.inference_mode():ids=m.generate(x.input_features.cuda().half(),language=score["language"],task="transcribe",max_new_tokens=64)
  row["asr_transcript"]=p.batch_decode(ids,skip_special_tokens=True)[0];row["asr_lyric_similarity"]=round(SequenceMatcher(None,normalized(expected),normalized(row["asr_transcript"])).ratio(),4)
 eligible=[r for r in rows if r["asr_lyric_similarity"]>=.9];report={"status":"eligible" if {r["case"] for r in eligible}=={"en","rapid_ko"} else "reject","selected":{case:min((r for r in eligible if r["case"]==case),key=lambda r:r["spectral_flux_p95"],default=None) for case in ("en","rapid_ko")},"rows":rows};(root/"evaluation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
