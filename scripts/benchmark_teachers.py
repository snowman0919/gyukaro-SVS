#!/usr/bin/env python3
"""Generate a resumable teacher batch through an OpenAI-compatible TTS endpoint."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import requests
import soundfile as sf


LANGUAGES = {"ko": "Korean", "en": "English", "ja": "Japanese"}


def valid_audio(path: Path) -> bool:
    try:
        return sf.info(path).frames > 0
    except RuntimeError:
        return False


def request_payload(source: dict, protocol: str, max_new_tokens: int) -> dict:
    reference = {"text": source["reference_text"]}
    if protocol == "fish":
        reference["audio"] = base64.b64encode(Path(source["reference_audio_path"]).read_bytes()).decode()
        return {
            "text": source["text"], "references": [reference], "format": "wav", "max_new_tokens": max_new_tokens,
            "use_memory_cache": "on",
        }
    reference["audio_path"] = str(Path(source["reference_audio_path"]).resolve())
    return {
        "input": source["text"], "references": [reference], "language": LANGUAGES[source["language"]],
        "instructions": source["style"], "max_new_tokens": max_new_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--protocol", choices=("openai", "fish"), default="openai")
    parser.add_argument("--input", default="configs/teachers/trilingual_pilot.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {json.loads(line)["id"]: json.loads(line) for line in output.read_text().splitlines()} if output.exists() else {}
    generated = 0
    metadata_changed = False
    for source in rows:
        if args.limit is not None and generated >= args.limit:
            break
        target = Path("data/teacher") / args.teacher / f"{source['id']}.wav"
        if source["id"] in completed and valid_audio(target):
            if completed[source["id"]].get("model_revision") != args.model_revision:
                completed[source["id"]]["model_revision"] = args.model_revision
                metadata_changed = True
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        response = requests.post(args.endpoint, json=request_payload(source, args.protocol, args.max_new_tokens), timeout=600)
        response.raise_for_status()
        target.write_bytes(response.content)
        info = sf.info(target)
        completed[source["id"]] = source | {
            "teacher": args.teacher, "model_revision": args.model_revision, "output_path": str(target),
            "sample_rate": info.samplerate, "channels": info.channels,
            "generation_config": {"max_new_tokens": args.max_new_tokens, "protocol": args.protocol},
            "quality_status": "pending_gate",
        }
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in completed.values()))
        generated += 1
        print(f"generated={generated} id={source['id']}", flush=True)
    if metadata_changed:
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in completed.values()))
    print(f"completed={len(completed)} generated_now={generated} output={output}")


if __name__ == "__main__":
    main()
