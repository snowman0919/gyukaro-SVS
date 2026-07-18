# SoulX Truncated Identity Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove or reject stable final-WAV gradient delivery through the frozen SoulX/vocoder path to the v0.7 identity FiLM adapter for K=2 and K=4.

**Architecture:** Add one experiment-only CLI containing an exact copy of the SoulX Euler/CFG step with the first `64-K` iterations under stop-gradient and the final K iterations differentiable. Reuse the existing SoulX loader, adapter, checkpoints, source audio, WavLM, and ECAPA; add one CPU unit test for gradient boundaries. Do not alter production modules or checkpoints.

**Tech Stack:** Python, PyTorch, SoulX-Singer, transformers WavLM x-vector, SpeechBrain ECAPA, torchaudio, pytest.

## Global Constraints

- Production remains one phrase-level 64-step SoulX decode.
- Run separate K=2 and K=4 diagnostics on a fixed 3-second crop.
- Trainable parameters are limited to `SoulXRealLatentAdapters.identity`.
- SoulX backbone and vocoder parameters remain frozen while their operations carry input gradients.
- A failed feasibility gate stops before optimizer training.
- Do not modify `data/source/`, RC7 artifacts, production checkpoints, renderer behavior, packaging, RC9, or OpenUtau.
- Do not use per-note TTS, final-WAV stitching, or waveform pitch shifting.

---

### Task 1: Truncated diffusion gradient boundary

**Files:**
- Create: `scripts/probe_truncated_identity_grad.py`
- Test: `tests/test_truncated_identity_grad.py`

**Interfaces:**
- Produces: `truncated_reverse_diffusion(flow, prompt, cond_frozen, cond_trainable, total_steps, grad_steps, cfg) -> torch.Tensor`
- Produces: `gradient_audit(adapter, frozen_modules) -> dict[str, object]`

- [ ] **Step 1: Write the failing CPU test**

```python
import importlib.util
from pathlib import Path
import torch

SPEC = importlib.util.spec_from_file_location("probe", Path("scripts/probe_truncated_identity_grad.py"))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

def test_truncated_reverse_matches_full_and_only_adapter_gets_gradient():
    torch.manual_seed(4)
    flow = probe.ToyFlow().eval()
    for parameter in flow.parameters():
        parameter.requires_grad_(False)
    adapter = torch.nn.Linear(3, 3, bias=False)
    prompt = torch.randn(1, 2, 3)
    target = torch.randn(1, 4, 3)
    frozen = torch.cat([prompt, adapter(target).detach()], 1)
    trainable = torch.cat([prompt, adapter(target)], 1)
    torch.manual_seed(9)
    expected = probe.full_reverse_diffusion(flow, prompt, frozen, 8, 2.0)
    torch.manual_seed(9)
    actual = probe.truncated_reverse_diffusion(flow, prompt, frozen, trainable, 8, 2, 2.0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert torch.isfinite(adapter.weight.grad).all() and adapter.weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in flow.parameters())
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest -q tests/test_truncated_identity_grad.py`

Expected: FAIL because `scripts/probe_truncated_identity_grad.py` does not exist.

- [ ] **Step 3: Implement the exact bounded helper**

Implement the production Euler/CFG equations in `truncated_reverse_diffusion`. Generate the same `torch.randn` target state as SoulX, run early iterations under `torch.no_grad()`, detach, then run only the final K iterations with `cond_trainable`. Reject `grad_steps` outside `1..total_steps`. Add `ToyFlow` and `full_reverse_diffusion` only for the runnable CPU test. `gradient_audit` must return adapter gradient norm and every unexpected frozen-parameter gradient name.

```python
def truncated_reverse_diffusion(flow, prompt, cond_frozen, cond_trainable, total_steps, grad_steps, cfg):
    if not 1 <= grad_steps <= total_steps:
        raise ValueError("grad_steps must be within total_steps")
    prompt_len = prompt.shape[1]
    target_len = cond_frozen.shape[1] - prompt_len
    noise = torch.randn(cond_frozen.shape[0], target_len, flow.mel_dim,
                        dtype=cond_frozen.dtype, device=cond_frozen.device)
    state = noise
    with torch.no_grad():
        for index in range(total_steps - grad_steps):
            state = diffusion_step(flow, state, prompt, cond_frozen, index, total_steps, cfg)
    state = state.detach()
    for index in range(total_steps - grad_steps, total_steps):
        state = diffusion_step(flow, state, prompt, cond_trainable, index, total_steps, cfg)
    return state
```

- [ ] **Step 4: Run the focused test**

Run: `pytest -q tests/test_truncated_identity_grad.py`

Expected: `1 passed`.

- [ ] **Step 5: Commit the gradient boundary**

```bash
git add scripts/probe_truncated_identity_grad.py tests/test_truncated_identity_grad.py
git commit -m "test: add truncated SoulX gradient probe"
```

### Task 2: Real 3-second final-WAV feasibility

**Files:**
- Modify: `scripts/probe_truncated_identity_grad.py`
- Test: `tests/test_truncated_identity_grad.py`
- Create at runtime, uncommitted: `artifacts/reports/truncated_identity_feasibility/k2/`
- Create at runtime, uncommitted: `artifacts/reports/truncated_identity_feasibility/k4/`

**Interfaces:**
- Consumes: `truncated_reverse_diffusion(...)` and `gradient_audit(...)` from Task 1.
- Produces: one `feasibility.json` and baseline/current/candidate WAV files per K.

- [ ] **Step 1: Add failing loss and audit tests**

Add tests asserting FFT sizes `256/1024/4096` are present, the combined loss is finite, zero candidate/baseline loss is zero for every preservation component, and an injected frozen gradient makes `gradient_audit` fail.

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q tests/test_truncated_identity_grad.py`

Expected: FAIL because the final-WAV loss and strict audit functions are absent.

- [ ] **Step 3: Implement the real feasibility CLI**

The CLI must:

1. load the pinned SoulX model/config/RMVPE, v0.7 adapter, fixed identity vector, and reference `data/processed/master/216.wav` through existing project loaders;
2. use the first voiced 3-second window from `artifacts/reports/rc8_sustained_decode_sweep/omnivoice_source.wav` and its RMVPE F0;
3. render identity OFF twice and require exact repeatability;
4. render current v0.7 production and the truncated candidate from identical seed/noise;
5. require candidate/current allclose at `atol=1e-5, rtol=1e-4` before loss backward;
6. load frozen WavLM and ECAPA, build fixed normalized centroids from recordings `171..194`, and keep their operations differentiable only with respect to candidate audio;
7. compute WavLM speaker, ECAPA speaker, waveform L1, log-STFT `256/1024/4096`, WavLM base-content feature, voiced-frame pitch-period, adapter-output, gate, and parameter-drift terms;
8. run one backward pass without an optimizer;
9. require finite nonzero identity-adapter gradient, no non-identity SoulX gradient, no vocoder gradient, finite losses/audio, and relative parameter drift exactly zero;
10. write peak allocated/reserved CUDA memory, wall time, gradient norm, every loss component, baseline reproduction deltas, WAV paths, revisions, crop interval, and `feasibility_pass`.

Use `torchaudio.functional.resample` so WavLM/ECAPA gradients remain connected. Use WavLM base hidden states for content preservation; do not use the non-differentiable SoulX Whisper wrapper. Compute pitch-period preservation from fixed RMVPE baseline periods with differentiable normalized autocorrelation on candidate frames.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_truncated_identity_grad.py`

Expected: all tests pass.

- [ ] **Step 5: Execute K=2 in a fresh process**

Run:

```bash
/home/kotori9/code/gyukaro/.venv-soulx/bin/python scripts/probe_truncated_identity_grad.py --grad-steps 2 --seed 21 --output artifacts/reports/truncated_identity_feasibility/k2
```

Expected: `feasibility.json` exists. If `feasibility_pass` is false, do not run optimizer training.

- [ ] **Step 6: Execute K=4 in a fresh process**

Run:

```bash
/home/kotori9/code/gyukaro/.venv-soulx/bin/python scripts/probe_truncated_identity_grad.py --grad-steps 4 --seed 21 --output artifacts/reports/truncated_identity_feasibility/k4
```

Expected: `feasibility.json` exists. If `feasibility_pass` is false, do not run optimizer training.

- [ ] **Step 7: Record the bounded result**

Add a concise feasibility section to `docs/rc8_quality_fixes.md` with K=2/K=4 gradient, frozen-gradient, VRAM, runtime, numerical-equivalence, and pass/fail evidence. Do not claim RC8 promotion.

- [ ] **Step 8: Verify repository invariants**

Run:

```bash
python scripts/validate_dataset.py
pytest -q tests/test_truncated_identity_grad.py
git diff --check
```

Expected: dataset validation passes, focused tests pass, and the diff has no whitespace errors. Package smoke is not required because no runtime or package file changes.

- [ ] **Step 9: Commit only reproducible code and concise report**

```bash
git add scripts/probe_truncated_identity_grad.py tests/test_truncated_identity_grad.py docs/rc8_quality_fixes.md artifacts/reports/truncated_identity_feasibility/k2/feasibility.json artifacts/reports/truncated_identity_feasibility/k4/feasibility.json
git commit -m "test: probe truncated identity gradients"
```

Do not stage generated WAVs, model caches, checkpoints, or unrelated existing artifacts.
