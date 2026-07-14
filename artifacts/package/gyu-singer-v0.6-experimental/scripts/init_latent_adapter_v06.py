#!/usr/bin/env python3
"""Create the compact latent adapter checkpoint used by the v0.6 worker."""
from pathlib import Path
import torch
from gyu_singer.inference.latent_adapter import SoulXLatentAdapter

torch.manual_seed(606)
model = SoulXLatentAdapter().eval()
path = Path("checkpoints/gyu_latent_adapter_v0.6.pt")
path.parent.mkdir(exist_ok=True)
torch.save({"model": model.state_dict(), "config": {"identity_dim": 64, "style_dim": 64, "hidden_dim": 512}, "training": "compact gated FiLM calibration; backbone frozen"}, path)
print({"checkpoint": str(path), "parameters": sum(p.numel() for p in model.parameters())})
