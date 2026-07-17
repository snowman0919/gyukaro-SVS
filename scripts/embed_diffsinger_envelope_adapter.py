#!/usr/bin/env python3
"""Append a bounded constant mel-envelope direction to a DiffSinger ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--strength", type=float, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.input.resolve() != args.output.resolve():
        shutil.copy2(args.input, args.output)
    model = onnx.load(args.output)
    graph = model.graph
    producer = next((node for node in graph.node if "mel" in node.output), None)
    if producer is None:
        raise ValueError("mel output producer not found")
    base = "gyu_envelope_adapter/base_mel"
    producer.output[list(producer.output).index("mel")] = base
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    delta = saved["delta"].numpy().astype(np.float32) * args.strength
    graph.initializer.append(numpy_helper.from_array(delta, "gyu_envelope_adapter/delta"))
    graph.node.append(helper.make_node(
        "Add", [base, "gyu_envelope_adapter/delta"], ["mel"],
        name="gyu_envelope_adapter/apply"))
    model.metadata_props.add(key="gyu.envelope_adapter", value=str(args.checkpoint))
    model.metadata_props.add(key="gyu.envelope_adapter_strength", value=str(args.strength))
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
