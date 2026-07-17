#!/usr/bin/env python3
"""Append a trained bounded frame-level mel adapter to a DiffSinger ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch


PREFIX = "gyu_mel_adapter"


def initializer(graph, name: str, value: np.ndarray) -> None:
    graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))


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
    base = f"{PREFIX}/base_mel"
    producer.output[list(producer.output).index("mel")] = base

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = saved["model"]
    config = saved["config"]
    values = {
        "mean": state["mean"].numpy(),
        "std": state["std"].numpy(),
        "w1": state["fc1.weight"].numpy().T,
        "b1": state["fc1.bias"].numpy(),
        "w2": state["fc2.weight"].numpy().T,
        "b2": state["fc2.bias"].numpy(),
        "sqrt2": np.asarray(np.sqrt(2), dtype=np.float32),
        "one": np.asarray(1, dtype=np.float32),
        "half": np.asarray(.5, dtype=np.float32),
        "scale": np.asarray(float(config["limit"]) * args.strength, dtype=np.float32),
    }
    for name, value in values.items():
        initializer(graph, f"{PREFIX}/{name}", value)
    nodes = [
        helper.make_node("Sub", [base, f"{PREFIX}/mean"], [f"{PREFIX}/centered"]),
        helper.make_node("Div", [f"{PREFIX}/centered", f"{PREFIX}/std"], [f"{PREFIX}/normalized"]),
        helper.make_node("MatMul", [f"{PREFIX}/normalized", f"{PREFIX}/w1"], [f"{PREFIX}/hidden_mm"]),
        helper.make_node("Add", [f"{PREFIX}/hidden_mm", f"{PREFIX}/b1"], [f"{PREFIX}/hidden"]),
        helper.make_node("Div", [f"{PREFIX}/hidden", f"{PREFIX}/sqrt2"], [f"{PREFIX}/gelu_x"]),
        helper.make_node("Erf", [f"{PREFIX}/gelu_x"], [f"{PREFIX}/gelu_erf"]),
        helper.make_node("Add", [f"{PREFIX}/gelu_erf", f"{PREFIX}/one"], [f"{PREFIX}/gelu_plus"]),
        helper.make_node("Mul", [f"{PREFIX}/hidden", f"{PREFIX}/gelu_plus"], [f"{PREFIX}/gelu_mul"]),
        helper.make_node("Mul", [f"{PREFIX}/gelu_mul", f"{PREFIX}/half"], [f"{PREFIX}/activated"]),
        helper.make_node("MatMul", [f"{PREFIX}/activated", f"{PREFIX}/w2"], [f"{PREFIX}/delta_mm"]),
        helper.make_node("Add", [f"{PREFIX}/delta_mm", f"{PREFIX}/b2"], [f"{PREFIX}/delta_raw"]),
        helper.make_node("Tanh", [f"{PREFIX}/delta_raw"], [f"{PREFIX}/delta_tanh"]),
        helper.make_node("Mul", [f"{PREFIX}/delta_tanh", f"{PREFIX}/scale"], [f"{PREFIX}/delta"]),
        helper.make_node("Add", [base, f"{PREFIX}/delta"], ["mel"]),
    ]
    graph.node.extend(nodes)
    model.metadata_props.add(key="gyu.mel_adapter", value=str(args.checkpoint))
    model.metadata_props.add(key="gyu.mel_adapter_strength", value=str(args.strength))
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
