from pathlib import Path

import numpy as np
import torch

from gyu_singer.inference.acoustic_refiner import AcousticRefinerRuntime
from gyu_singer.model import VocalAcousticRefiner


def test_chunked_runtime_preserves_length_and_silence(tmp_path: Path):
    model = VocalAcousticRefiner(channels=16, blocks=2, adapter_rank=4)
    checkpoint = tmp_path / "refiner.pt"
    torch.save({"stage": "gyu", "model_config": model.config, "model": model.state_dict()}, checkpoint)
    runtime = AcousticRefinerRuntime(checkpoint, "cpu")
    source = np.zeros(20_000, dtype="float32")
    output = runtime.process(source, chunk_samples=4096, overlap=512)
    assert output.shape == source.shape
    assert np.array_equal(output, source)
