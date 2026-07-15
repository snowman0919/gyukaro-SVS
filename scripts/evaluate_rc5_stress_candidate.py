#!/usr/bin/env python3
"""Evaluate all mandatory RC5 stress cases; human listening remains separate."""
from __future__ import annotations
import argparse,json,os,sys
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np,soundfile as sf,torch
from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor
from evaluate_rc4_artifact_matrix import acoustics,audio16,normalized,pitch
from gyu_singer.data import acoustic_reference_features
from gyu_singer.inference.quality_controller import QualityPitchController
from gyu_singer.inference.soulx import SoulXPhraseRenderer
from gyu_singer.score import normalize_score
CACHE=Path(os.environ.get("GYU_SINGER_CACHE","data/cache"));sys.path.insert(0,str(CACHE/"soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor
SCORES={"ko_neutral":"examples/quality_ko.json","en":"examples/quality_en.json","rapid_ko":"examples/review_rapid_ko.json","large_interval_ko":"examples/review_large_interval_ko.json","ko_breathy":"artifacts/reports/rc5_stress_candidate/ko_breathy.json","ko_energetic":"artifacts/reports/rc5_stress_candidate/ko_energetic.json","ja":"examples/quality_ja.json","sustained_ko":"examples/review_sustain_ko.json","phrase_boundary":"examples/review_phrase_boundary_ko.json"}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--root",default="artifacts/reports/rc5_stress_candidate");args=parser.parse_args();root=Path(args.root);manifest=json.loads((root/"manifest.json").read_text());controller=QualityPitchController("checkpoints/gyu_prosody_v0.5.pt",acoustic_reference_features("data/processed/master/216.wav"));targets={};scores={}
 for case,path in SCORES.items():
  score=normalize_score(json.loads(Path(path).read_text()));duration=sf.info(manifest["files"][case]["path"]).duration;expressive=controller.predict(score,canonical_timing=True)[0]*score["style"]["prosody_strength"];target,_=SoulXPhraseRenderer._canonical_f0(score,duration,expressive.cpu().numpy());target_path=manifest["files"][case].get("target_f0");target=np.load(target_path) if target_path else target;targets[case]=target;scores[case]=score;np.save(root/f"{case}_target_f0.npy",target)
 del controller;torch.cuda.empty_cache();ex=F0Extractor(str(CACHE/"soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),device="cuda",target_sr=24000,hop_size=480,verbose=False);rows=[]
 for case,item in manifest["files"].items():
  path=Path(item["path"]);rows.append({"case":case,"path":str(path),"sample_rate":sf.info(path).samplerate}|acoustics(path)|pitch(path,targets[case],ex))
 del ex;torch.cuda.empty_cache();p=AutoProcessor.from_pretrained(CACHE/"whisper-large-v3-turbo");m=AutoModelForSpeechSeq2Seq.from_pretrained(CACHE/"whisper-large-v3-turbo",dtype=torch.float16).cuda().eval()
 for row in rows:
  score=scores[row["case"]];expected=" ".join(n["lyric"] for n in score["notes"]);x=p(audio16(Path(row["path"])),sampling_rate=16000,return_tensors="pt")
  with torch.inference_mode():ids=m.generate(x.input_features.cuda().half(),language=score["language"],task="transcribe",max_new_tokens=64)
  row["asr_transcript"]=p.batch_decode(ids,skip_special_tokens=True)[0];actual=normalized(row["asr_transcript"]);expected_norm=normalized(expected);matcher=SequenceMatcher(None,expected_norm,actual);row["asr_lyric_similarity"]=round(matcher.ratio(),4);row["asr_lyric_coverage"]=round(sum(block.size for block in matcher.get_matching_blocks())/max(len(expected_norm),1),4)
 names=("pitch_mae_cents","voicing_accuracy","hf_energy_ratio_p95","hf_spike_p99_over_median","spectral_flatness_mean","spectral_flux_p95","sample_jump_p999","clip_fraction","asr_lyric_similarity","asr_lyric_coverage");aggregate={n:round(float(np.mean([r[n] for r in rows if r[n] is not None])),6) for n in names};rc4=json.loads(Path("artifacts/reports/rc5_candidate_core/evaluation.json").read_text())["aggregate"]["rc4"];core=[r for r in rows if r["case"] in {"ko_neutral","en","rapid_ko","large_interval_ko"}];core_aggregate={n:round(float(np.mean([r[n] for r in core if r[n] is not None])),6) for n in names};delta={n:round(core_aggregate[n]-rc4[n],6) for n in names if n in rc4}
 objective=all(r["sample_rate"]==48000 and r["clip_fraction"]==0 and r["asr_lyric_coverage"]>=.8 for r in rows);report={"status":"objective_candidate_human_pending" if objective else "objective_reject","human_listening":"pending","aggregate_9":aggregate,"core_4":core_aggregate,"core_4_minus_rc4":delta,"rows":rows};(root/"evaluation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
