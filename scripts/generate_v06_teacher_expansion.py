#!/usr/bin/env python3
"""Generate non-duplicated Fish/MOSS paired teacher audio for v0.6."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import soundfile as sf
import torch


TEXTS = {
    "ko": [
        "이른 새벽 골목에 빗방울 소리가 번진다.", "노란 우산 아래 두 사람이 천천히 걷는다.",
        "먼 산 위로 흰 구름이 길게 흘러간다.", "작은 찻집 창가에 따뜻한 김이 오른다.",
        "파도 끝에서 갈매기가 크게 원을 그린다.", "낡은 지도 위에 새로운 길을 표시한다.",
        "저녁 바람이 푸른 커튼을 살며시 흔든다.", "아이들은 운동장에서 둥근 공을 주고받는다.",
        "비밀스러운 편지가 책갈피 사이에 놓여 있다.", "눈 덮인 들판에 여우 발자국이 이어진다.",
        "기차 창문 밖으로 보랏빛 노을이 지나간다.", "낮은 다리 밑에서 맑은 물이 노래한다.",
        "봄비가 지난 뒤 화단에 새싹이 돋는다.", "조용한 도서관에서 시계 소리만 들린다.",
        "은빛 달이 호수 표면을 환하게 비춘다.", "오래된 사진 속 웃음이 다시 떠오른다.",
        "등불 하나가 어두운 길을 밝혀 준다.", "먼 바다 너머로 작은 배가 돌아온다.",
        "손바닥 위 눈송이가 금세 녹아내린다.", "새벽 종소리가 마을 전체에 울려 퍼진다.",
    ],
    "en": [
        "Rain taps softly against the narrow attic window.", "A paper boat turns slowly in the shallow stream.",
        "Morning bells travel across the empty market square.", "One lantern glows beside the quiet harbor wall.",
        "The old piano keeps a gentle room of echoes.", "Fresh paint brightens the door of the small bakery.",
        "A silver bicycle rests beneath the maple tree.", "Summer insects hum above the sleeping garden path.",
        "Two sparrows circle over the red tiled roof.", "The traveler folds a map beside the station clock.",
        "Cold stars appear beyond the dark pine forest.", "A warm cup steams near the open kitchen door.",
        "Soft snow settles on the wooden fence at dusk.", "The river carries fallen leaves toward the sea.",
        "Bright chalk marks a new game on the pavement.", "A distant trumpet calls from the hillside road.",
        "The moon paints a line across the still lake.", "Gentle thunder fades behind the western ridge.",
        "A small key waits under the blue flowerpot.", "Golden wheat bends when the evening wind arrives.",
    ],
    "ja": [
        "あめのあとににじがみずたまりにうつる。", "あさのえきであたたかいパンのにおいがする。",
        "しろいねこがまどべでゆっくりのびをする。", "やまのみちにちいさなすずのねがひびく。",
        "ゆうひがかわのうえをきんいろにそめる。", "ふるいとけいがしずかなへやでなる。",
        "あおいふうせんがそらのたかくへあがる。", "もりのおくでこだまがこえをかえす。",
        "ゆきのうえにうさぎのあしあとがつづく。", "ひかるほしをみあげてねがいをかける。",
        "ちいさなふねがうみのむこうへすすむ。", "はるのかぜがさくらのはなをはこぶ。",
        "あかりのついたみせにひとがあつまる。", "まっすぐなみちをじてんしゃでかけぬける。",
        "あたらしいほんをひらいてたびをはじめる。", "よるのひろばでこどもたちがわらう。",
        "ことりのうたがにわいっぱいにひろがる。", "つきあかりがしずかなみなとをてらす。",
        "こおったまどにゆびでえをかいてみる。", "とおいまちからでんしゃがゆっくりくる。",
    ],
}
REFERENCES = [(216, "low_register"), (220, "neutral_mid"), (215, "high_register"), (219, "expressive_phrase"), (212, "long_natural_phrase")]
LANGUAGE = {"ko": "Korean", "en": "English", "ja": "Japanese"}


def rows() -> list[dict]:
    segment = {json.loads(line)["source_index"]: json.loads(line)["text"] for line in Path("data/manifests/real_segments.jsonl").read_text().splitlines() if line}
    values = []
    for language, texts in TEXTS.items():
        for number, (text, (source, role)) in enumerate(zip(texts, cycle(REFERENCES)), 1):
            values.append({"id": f"teacher_v06_{language}_{number:03d}", "language": language, "text": text, "style": "neutral", "reference_ids": [f"gyu_real_{source:06d}"], "reference_role": role, "reference_audio_path": f"data/processed/master/{source}.wav", "reference_text": segment[source]})
    assert len(values) == 60 and len({(row["language"], row["text"], tuple(row["reference_ids"])) for row in values}) == 60
    return values


def fish(items: list[dict], output: Path) -> list[dict]:
    sys.path.insert(0, "data/cache/fish-speech")
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
    root = Path("data/cache/fish-s2-pro")
    queue = launch_thread_safe_queue(checkpoint_path=str(root), device="cuda", precision=torch.bfloat16, compile=False)
    engine = TTSInferenceEngine(queue, load_model("modded_dac_vq", str(root / "codec.pth"), device="cuda"), torch.bfloat16, False)
    result = []
    for index, row in enumerate(items):
        path = output / f"{row['id']}.wav"
        if not path.exists():
            request = ServeTTSRequest(text=row["text"], references=[ServeReferenceAudio(audio=Path(row["reference_audio_path"]).read_bytes(), text=row["reference_text"])], seed=606 + index, max_new_tokens=512, chunk_length=200)
            for value in engine.inference(request):
                if value.code == "error": raise value.error
                if value.code == "final":
                    rate, audio = value.audio; sf.write(path, audio, rate)
        result.append(row | {"teacher": "fish_s2_pro", "model_revision": "Fish Audio S2 Pro local pinned", "output_path": str(path), "sample_rate": 44100, "quality_status": "pending_gate"})
        print(row["id"], flush=True)
    return result


def moss(items: list[dict], output: Path) -> list[dict]:
    from transformers import AutoModel, AutoProcessor
    device = "cuda"; processor = AutoProcessor.from_pretrained("data/cache/moss-local-v1.5", trust_remote_code=True); processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained("data/cache/moss-local-v1.5", trust_remote_code=True, local_files_only=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16).to(device).eval()
    result = []
    for index, row in enumerate(items):
        path = output / f"{row['id']}.wav"
        if not path.exists():
            conversation = [[processor.build_user_message(text=row["text"], reference=[row["reference_audio_path"]], language=LANGUAGE[row["language"]])]]
            batch = processor(conversation, mode="generation")
            with torch.inference_mode(): output_ids = model.generate(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), max_new_tokens=1024, do_sample=True, audio_temperature=1.2, audio_top_p=.8, audio_top_k=25, audio_repetition_penalty=1.0)
            decoded = next(message for message in processor.decode(output_ids) if message is not None)
            sf.write(path, decoded.audio_codes_list[0].T.float().cpu().numpy(), processor.model_config.sampling_rate)
        result.append(row | {"teacher": "moss_local_v15", "model_revision": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5@be7766a6735b98bd793f7c79fb720b4d0f5d13b8", "output_path": str(path), "sample_rate": 48000, "quality_status": "pending_gate"})
        print(row["id"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--teacher", choices=("fish", "moss"), required=True); parser.add_argument("--limit", type=int); args = parser.parse_args()
    items = rows()[:args.limit] if args.limit else rows(); root = Path("data/teacher") / f"{args.teacher}_v06_unique"; root.mkdir(parents=True, exist_ok=True)
    generated = fish(items, root) if args.teacher == "fish" else moss(items, root)
    Path(f"data/manifests/teacher_v06_unique_{args.teacher}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated))


if __name__ == "__main__": main()
