#!/usr/bin/env python3
"""Promote ASR-confirmed recording-script correspondence into training manifests."""
from __future__ import annotations

import json
from pathlib import Path

BLOCKS = [(106,117,"A"),(118,139,"B"),(140,157,"C"),(158,211,"D"),(212,221,"E"),(222,236,"F"),(237,237,"G")]
ITEMS = [
 (146,146,"C3-1","강물은 멀리 흘러간다."),(147,147,"C3-2","작은 별빛이 밤을 건넌다."),(148,148,"C3-3","찬 바람 끝에 꽃잎이 흔들린다."),(149,149,"C3-4","검은 숲 너머 새벽이 온다."),(150,150,"C3-5","햇살은 조용히 창문을 두드린다."),
 (151,151,"C4-1","빛 아래 작은 아이가 웃어."),(152,152,"C4-2","꽃잎은 바람을 따라 흘러가."),(153,153,"C4-3","깊은 어둠 안에서 노래가 들려."),(154,154,"C4-4","마음 안에 오래 남은 이름을 불러."),(155,155,"C4-5","길 위에 놓인 작은 돌을 넘어가."),
 (156,156,"C5-1","작은 밤빛 끝에서 / 깊은 숲길 안으로 / 붉은 꽃잎 사이로"),(157,157,"C5-2","맑은 하늘 끝까지 / 긴 꿈을 품고서 / 다시 한 걸음 앞으로"),
 (158,160,"D1-1","하늘 끝에 빛이 내려."),(161,163,"D1-2","조용한 밤을 지나서."),(164,166,"D1-3","작은 마음이 노래해."),(167,169,"D1-4","푸른 바람이 불어와."),(170,172,"D1-5","멀리 별 하나 떠오르네."),
 (173,175,"D2-1","오늘도 나는 걸어가."),(176,178,"D2-2","흔들려도 다시 서 있어."),(179,181,"D2-3","차가운 길 위에 서서."),(182,184,"D2-4","따뜻한 숨을 내쉰다."),(185,187,"D2-5","기다린 시간은 지나가."),
 (188,190,"D3-1","너의 이름을 불러본다."),(191,193,"D3-2","희미한 꿈을 따라간다."),(194,196,"D3-3","사라진 계절을 기억해."),(197,199,"D3-4","아직도 마음은 남아 있어."),(200,202,"D3-5","다시 또 노래가 시작돼."),
 (203,205,"D4-1","바람 따라 달려가 / 빛을 따라 날아가 / 두려움은 지나가 / 다시 시작해."),(206,208,"D4-2","손을 뻗어 앞으로 / 넘어져도 앞으로 / 작은 꿈을 따라서 / 계속 걸어가."),(209,211,"D5","빛 — 이 와 / 바 — 람 불어 / 꿈 — 을 따라가 / 다 — 시 노래해"),
 (212,213,"E1","아침이 오면 닫힌 창문 너머로 조용히 번지는 빛을 따라 걸어가 아직은 서툰 마음이라 해도 멈추지 않고 다시 노래할 거야"),(214,215,"E2","멀어진 계절의 끝에서 잊혀진 이름을 다시 불러 흩어진 바람 사이로 작은 목소리가 피어나"),(216,217,"E3","푸른 밤하늘 아래서 나는 나의 길을 찾아가 흔들리는 시간 속에도 이 노래만은 남아 있어 가장 낮은 마음의 자리에서 다시 작은 숨을 고르고 아직 보이지 않는 내일을 향해 천천히 걸어가"),(218,219,"E4","잠시 멈춰 서서 숨을 고르면 멀리 사라진 빛도 다시 보여 가장 낮은 마음의 자리에서 나는 또 하루를 노래해"),(220,221,"E5","오늘의 하늘은 맑아 가벼운 바람이 불어 닫힌 문을 열고서 다시 앞으로 가 손끝에 닿은 햇살 마음에 번지는 노래 넘어져도 괜찮아 다시 시작하면 돼"),
]
asr = {x["source_index"]: x for x in map(json.loads, Path("data/manifests/asr_transcripts.jsonl").read_text().splitlines())}
records = {x["source_index"]: x for x in map(json.loads, Path("data/manifests/real_recordings.jsonl").read_text().splitlines())}
known = {i: (item, text) for first,last,item,text in ITEMS for i in range(first,last+1)}
alignments, segments, review, supervised = [], [], [], []
for index in sorted(records):
    record = records[index]; block = next(b for lo,hi,b in BLOCKS if lo <= index <= hi)
    item, text = known.get(index, ("asr_only", asr[index]["transcript"]))
    confidence = .99 if index in known else (.70 if asr[index]["transcript"] else .10)
    segment_type = "phrase" if block in "CDE" else "exercise"
    row = {"id":record["id"],"source_file":record["source_file"],"source_index":index,"start_sec":0.0,"end_sec":record["duration_sec"],"language":"ko","script_block":block,"script_item":item,"text":text,"normalized_text":text.replace(" ",""),"asr_text":asr[index]["transcript"],"segment_type":segment_type,"alignment_confidence":confidence,"f0_median_hz":record["f0_median_hz"],"f0_min_hz":record["f0_min_hz"],"f0_max_hz":record["f0_max_hz"],"voiced_ratio":record["voiced_frame_ratio"],"quality_flags":[] if confidence >= .9 else ["needs_script_review"]}
    segments.append(row); alignments.append({k:row[k] for k in ("source_index","script_block","script_item","text","asr_text","alignment_confidence")})
    if confidence >= .9 and block in "CDE": supervised.append({"id":row["id"],"audio_path":record["pcm_master"],"text":text,"language":"ko","duration_sec":record["duration_sec"],"f0_median_hz":record["f0_median_hz"],"trust_weight":1.0,"split":"test" if index % 17 == 0 else "validation" if index % 11 == 0 else "train"})
    if confidence < .9: review.append(f"- {index}: {asr[index]['transcript'][:120]}")
Path("data/manifests/real_segments.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in segments))
Path("data/manifests/script_alignment.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in alignments))
Path("data/manifests/neural_supervision.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in supervised))
Path("artifacts/reports/alignment_review.md").write_text("# Needs manual script review\n\n" + "\n".join(review) + "\n")
print(f"aligned={sum(x['alignment_confidence'] >= .9 for x in segments)} review={len(review)}")
