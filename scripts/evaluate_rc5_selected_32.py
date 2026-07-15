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
 root=Path("artifacts/reports/rc5_selected_32");data=json.loads((root/"manifest.json").read_text());ex=F0Extractor(str(CACHE/"soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),device="cuda",target_sr=24000,hop_size=480,verbose=False);rows=[]
 for row in data["rows"]:rows.append(row|acoustics(Path(row["path"]))|pitch(Path(row["path"]),np.load(Path("artifacts/reports/rc5_candidate_core")/row["case"]/"canonical_f0.npy"),ex))
 del ex;torch.cuda.empty_cache();p=AutoProcessor.from_pretrained(CACHE/"whisper-large-v3-turbo");m=AutoModelForSpeechSeq2Seq.from_pretrained(CACHE/"whisper-large-v3-turbo",dtype=torch.float16).cuda().eval()
 for row in rows:
  score=json.loads(Path(row["score"]).read_text());expected=" ".join(n["lyric"] for n in score["notes"]);x=p(audio16(Path(row["path"])),sampling_rate=16000,return_tensors="pt")
  with torch.inference_mode():ids=m.generate(x.input_features.cuda().half(),language=score["language"],task="transcribe",max_new_tokens=64)
  row["asr_transcript"]=p.batch_decode(ids,skip_special_tokens=True)[0];row["asr_lyric_similarity"]=round(SequenceMatcher(None,normalized(expected),normalized(row["asr_transcript"])).ratio(),4)
 names=("pitch_mae_cents","voicing_accuracy","hf_energy_ratio_p95","hf_spike_p99_over_median","spectral_flatness_mean","spectral_flux_p95","sample_jump_p999","clip_fraction","asr_lyric_similarity");agg={n:round(float(np.mean([r[n] for r in rows if r[n] is not None])),6) for n in names};rc4=json.loads(Path("artifacts/reports/rc5_candidate_core/evaluation.json").read_text())["aggregate"]["rc4"];delta={n:round(agg[n]-rc4[n],6) for n in names};status="objective_candidate_human_pending" if min(r["asr_lyric_similarity"] for r in rows)>=.8 and agg["spectral_flux_p95"]<rc4["spectral_flux_p95"] and agg["sample_jump_p999"]<rc4["sample_jump_p999"] else "reject";report={"status":status,"aggregate":agg,"minus_rc4":delta,"rows":rows};(root/"evaluation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
