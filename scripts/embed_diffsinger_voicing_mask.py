#!/usr/bin/env python3
"""Embed a token-derived unvoiced F0 mask in a DiffSinger acoustic ONNX.

Stock OpenUtau sends score F0 through consonant frames.  Checkpoints trained
with zero-F0 unvoiced frames can lose pronunciation when that happens.  This
transform derives the current phoneme token for every acoustic frame from the
model's existing length-regulator index and masks F0 inside the portable ONNX
graph.  It does not require an OpenUtau core fork.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


MASK_PREFIX = "gyu_voicing_mask"


def embed_voicing_mask(
    model_path: Path,
    phoneme_path: Path,
    unvoiced_phonemes: set[str],
) -> dict:
    model = onnx.load(model_path)
    graph = model.graph
    input_names = {value.name for value in graph.input}
    if not {"tokens", "f0"}.issubset(input_names):
        raise ValueError(f"unsupported acoustic inputs: {sorted(input_names)}")

    phonemes = json.loads(phoneme_path.read_text(encoding="utf-8"))
    missing = sorted(unvoiced_phonemes - set(phonemes))
    if missing:
        raise ValueError(f"unvoiced tokens missing from acoustic vocabulary: {missing}")

    mel2ph = next((output for node in graph.node for output in node.output
                   if output.endswith("/lr/ReduceSum_1_output_0")), None)
    if mel2ph is None:
        raise ValueError("length-regulator frame-to-token index was not found")

    f0_consumers = [index for index, node in enumerate(graph.node) if "f0" in node.input]
    if len(f0_consumers) != 1:
        raise ValueError(f"expected one direct F0 consumer, found {len(f0_consumers)}")
    insert_at = f0_consumers[0]

    zero_token_name = f"{MASK_PREFIX}/zero_token"
    table_name = f"{MASK_PREFIX}/voiced_table"
    padded_name = f"{MASK_PREFIX}/tokens_padded"
    frame_token_name = f"{MASK_PREFIX}/frame_tokens"
    frame_mask_name = f"{MASK_PREFIX}/frame_mask"
    masked_f0_name = f"{MASK_PREFIX}/f0"

    voiced_table = np.ones(max(phonemes.values()) + 1, dtype=np.float32)
    voiced_table[0] = 0.0
    for phoneme in unvoiced_phonemes:
        voiced_table[phonemes[phoneme]] = 0.0
    graph.initializer.extend([
        numpy_helper.from_array(np.zeros((1, 1), dtype=np.int64), zero_token_name),
        numpy_helper.from_array(voiced_table, table_name),
    ])
    nodes = [
        helper.make_node("Concat", [zero_token_name, "tokens"], [padded_name],
                         axis=1, name=f"{MASK_PREFIX}/pad_tokens"),
        helper.make_node("GatherElements", [padded_name, mel2ph], [frame_token_name],
                         axis=1, name=f"{MASK_PREFIX}/frame_tokens"),
        helper.make_node("Gather", [table_name, frame_token_name], [frame_mask_name],
                         axis=0, name=f"{MASK_PREFIX}/frame_mask"),
        helper.make_node("Mul", ["f0", frame_mask_name], [masked_f0_name],
                         name=f"{MASK_PREFIX}/apply"),
    ]
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == "f0":
                node.input[index] = masked_f0_name
    for offset, node in enumerate(nodes):
        graph.node.insert(insert_at + offset, node)

    model.metadata_props.add(
        key="gyu.voicing_mask",
        value="token-derived zero-F0 mask: " + ",".join(sorted(unvoiced_phonemes)),
    )
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    return {
        "model": str(model_path),
        "unvoiced_phonemes": sorted(unvoiced_phonemes),
        "unvoiced_token_ids": sorted(phonemes[p] for p in unvoiced_phonemes),
        "mask_embedded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--phonemes", type=Path, required=True)
    parser.add_argument("--unvoiced", action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(embed_voicing_mask(
        args.model.resolve(), args.phonemes.resolve(), set(args.unvoiced)),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
