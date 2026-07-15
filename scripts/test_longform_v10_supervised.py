#!/usr/bin/env python3
"""Run the native two-minute OpenUtau export while sampling host and CUDA memory."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import torch


def mem_used_mb() -> float:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1); values[key] = int(value.split()[0])
    return (values["MemTotal"] - values["MemAvailable"]) / 1024


def tree_rss_mb(root: int | None) -> float:
    if root is None:
        return 0.0
    parents, rss = {}, {}
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in path.read_text().splitlines() if ":" in line}
            pid = int(path.parent.name); parents[pid] = int(fields["PPid"]); rss[pid] = int(fields.get("VmRSS", "0 kB").split()[0])
        except (FileNotFoundError, KeyError, ValueError, PermissionError):
            continue
    selected, changed = {root}, True
    while changed:
        before = len(selected); selected.update(pid for pid, parent in parents.items() if parent in selected); changed = len(selected) != before
    return sum(rss.get(pid, 0) for pid in selected) / 1024


def wait_health(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if json.load(urllib.request.urlopen(url, timeout=2))["status"] == "ok":
                return
        except Exception:
            time.sleep(.25)
    raise TimeoutError(f"resident health timeout: {url}")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/reports/longform_v10_supervised.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", default="gyu-singer-v0.8")
    parser.add_argument("--wav", default="artifacts/reports/longform_v10.wav")
    parser.add_argument("--metrics", default="artifacts/reports/longform_v10_render_metrics.json")
    parser.add_argument("--server-log", default="artifacts/reports/longform_v10_server.log")
    parser.add_argument("--render-log", default="artifacts/reports/longform_v10_render.log")
    args = parser.parse_args()
    root = Path.cwd(); reports = root / "artifacts/reports"; reports.mkdir(parents=True, exist_ok=True)
    home = Path("/tmp/gyu-longform-v10-supervised"); shutil.rmtree(home, ignore_errors=True); home.mkdir()
    wav, metrics = Path(args.wav).resolve(), Path(args.metrics).resolve()
    server_log_path, render_log_path = Path(args.server_log).resolve(), Path(args.render_log).resolve()
    for path in (wav, metrics, server_log_path, render_log_path): path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "PYTHONPATH": str(root / "src"), "GYU_SINGER_CACHE": str(root / "data/cache"),
        "GYU_SOULX_PYTHON": str(root / ".venv-soulx/bin/python"), "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    baseline_gpu = (gpu_total - gpu_free) / 2**20; baseline_system = mem_used_mb()
    server_log = server_log_path.open("w")
    server = subprocess.Popen([
        sys.executable, "-m", "gyu_singer.cli", "--backend", args.backend, "--reference",
        "data/processed/master/216.wav", "serve", "--port", str(args.port),
    ], cwd=root, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
    render_process: subprocess.Popen | None = None
    samples: list[dict] = []
    running = True

    def monitor() -> None:
        while running:
            free, total = torch.cuda.mem_get_info()
            samples.append({"system": mem_used_mb(), "gpu": (total - free) / 2**20,
                            "resident": tree_rss_mb(server.pid), "openutau": tree_rss_mb(render_process.pid if render_process else None)})
            time.sleep(.2)

    try:
        wait_health(f"http://127.0.0.1:{args.port}/health")
        before_rss = tree_rss_mb(server.pid)
        thread = threading.Thread(target=monitor, daemon=True); thread.start()
        command = [
            "nix", "shell", "nixpkgs#dotnet-sdk_8", "--command", "env", f"HOME={home}",
            f"GYU_RENDERER_URL=http://127.0.0.1:{args.port}/render", f"GYU_LONGFORM_OUTPUT={wav}",
            f"GYU_LONGFORM_METRICS={metrics}", "DOTNET_CLI_TELEMETRY_OPTOUT=1",
            "integrations/openutau/test_longform_fork.sh", "/tmp/OpenUtau-v10-fresh",
        ]
        started = time.perf_counter()
        with render_log_path.open("w") as render_log:
            render_process = subprocess.Popen(command, cwd=root, env=env, stdout=render_log, stderr=subprocess.STDOUT, start_new_session=True)
            code = render_process.wait()
        elapsed = time.perf_counter() - started
    finally:
        running = False
        if "thread" in locals(): thread.join(timeout=2)
        stop(server); server_log.close()

    render = json.loads(metrics.read_text()) if metrics.exists() else {}
    report = {
        "environment": {"device": torch.cuda.get_device_name(), "cuda_memory_source": "torch.cuda.mem_get_info; nvidia-smi reports Not Supported on unified-memory GB10"},
        "render_exit_code": code, "wall_seconds_including_build": round(elapsed, 3),
        "resident_rss_before_render_mb": round(before_rss, 2),
        "peak_resident_process_tree_rss_mb": round(max(row["resident"] for row in samples), 2),
        "peak_openutau_process_tree_rss_mb": round(max(row["openutau"] for row in samples), 2),
        "system_used_baseline_mb": round(baseline_system, 2), "peak_system_used_mb": round(max(row["system"] for row in samples), 2),
        "system_used_growth_mb": round(max(row["system"] for row in samples) - baseline_system, 2),
        "gpu_used_baseline_mb": round(baseline_gpu, 2), "peak_gpu_used_mb": round(max(row["gpu"] for row in samples), 2),
        "gpu_used_growth_mb": round(max(row["gpu"] for row in samples) - baseline_gpu, 2),
        "sample_count": len(samples), "render_metrics": render, "project": "examples/openutau_v10_longform.ustx",
        "backend": args.backend,
        "wav": str(wav), "render_log": str(render_log_path), "server_log": str(server_log_path),
        "pass": code == 0 and render.get("phrases") == 17 and render.get("failed_phrases") == 0 and render.get("stale_cache_files_after_repeat") == 0,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
