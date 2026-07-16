#!/usr/bin/env python3
"""Supervise the real OpenUtau reference-song export and record resources."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from test_longform_v10_supervised import mem_used_mb, stop, tree_rss_mb, wait_health


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openutau", type=Path, default=Path("/tmp/OpenUtau-rc9-fresh"))
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    root = Path.cwd()
    work = root / "data/external/work/rc9_reference"
    project = work / "nonbreath_oblige_gyu_rc9.ustx"
    wav = work / "openutau_render.wav"
    metrics = root / "artifacts/reports/reference_song_rc9_openutau_render.json"
    requests = work / "openutau_phrase_requests.json"
    report_path = root / "artifacts/reports/reference_song_rc9_runtime.json"
    server_log = work / "server.log"
    render_log = work / "openutau_render.log"
    home = Path("/tmp/gyu-reference-rc9-home")
    shutil.rmtree(home, ignore_errors=True)
    home.mkdir()
    metrics.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)
    env = os.environ | {
        "PYTHONPATH": str(root / "src"), "GYU_SINGER_CACHE": str(root / "data/cache"),
        "GYU_SOULX_PYTHON": str(root / ".venv-soulx/bin/python"), "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    baseline_gpu = (gpu_total - gpu_free) / 2**20
    baseline_system = mem_used_mb()
    server_stream = server_log.open("w")
    server = subprocess.Popen([
        sys.executable, "-m", "gyu_singer.cli", "--backend", "gyu-singer-rc9",
        "--reference", "data/processed/master/216.wav", "serve", "--port", str(args.port),
    ], cwd=root, env=env, stdout=server_stream, stderr=subprocess.STDOUT, start_new_session=True)
    render_process = None
    samples: list[dict] = []
    running = True
    code, elapsed, before_rss = -1, 0.0, 0.0

    def monitor() -> None:
        while running:
            free, total = torch.cuda.mem_get_info()
            samples.append({
                "system": mem_used_mb(), "gpu": (total - free) / 2**20,
                "resident": tree_rss_mb(server.pid),
                "openutau": tree_rss_mb(render_process.pid if render_process else None),
            })
            time.sleep(.2)

    try:
        wait_health(f"http://127.0.0.1:{args.port}/health", timeout=120)
        before_rss = tree_rss_mb(server.pid)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        command = [
            "nix", "shell", "nixpkgs#dotnet-sdk_8", "--command", "env", f"HOME={home}",
            f"GYU_RENDERER_URL=http://127.0.0.1:{args.port}/render",
            f"GYU_REFERENCE_PROJECT={project}", f"GYU_REFERENCE_OUTPUT={wav}",
            f"GYU_REFERENCE_METRICS={metrics}", "DOTNET_CLI_TELEMETRY_OPTOUT=1",
            f"GYU_REFERENCE_REQUESTS={requests}",
            "integrations/openutau/test_reference_song_fork.sh", str(args.openutau),
        ]
        started = time.perf_counter()
        with render_log.open("w") as stream:
            render_process = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            code = render_process.wait()
        elapsed = time.perf_counter() - started
    finally:
        running = False
        if "thread" in locals():
            thread.join(timeout=2)
        stop(server)
        server_stream.close()
    render = json.loads(metrics.read_text()) if metrics.exists() else {}
    report = {
        "status": "pass" if code == 0 and render.get("failed_phrases") == 0 else "fail",
        "openutau": {"repository": "https://github.com/stakira/OpenUtau.git", "commit": "27573ac5c888d927119d5f65a207312d79194b1f", "path": str(args.openutau)},
        "backend": "gyu-singer-rc9", "wall_seconds_including_build": round(elapsed, 3),
        "resident_rss_before_render_mb": round(before_rss, 2),
        "peak_resident_process_tree_rss_mb": round(max((row["resident"] for row in samples), default=0), 2),
        "peak_openutau_process_tree_rss_mb": round(max((row["openutau"] for row in samples), default=0), 2),
        "system_used_baseline_mb": round(baseline_system, 2), "peak_system_used_mb": round(max((row["system"] for row in samples), default=baseline_system), 2),
        "gpu_used_baseline_mb": round(baseline_gpu, 2), "peak_gpu_used_mb": round(max((row["gpu"] for row in samples), default=baseline_gpu), 2),
        "render_exit_code": code, "render_metrics": render,
        "local_project": str(project.relative_to(root)), "local_wav": str(wav.relative_to(root)),
        "local_phrase_requests": str(requests.relative_to(root)),
        "copyright": "project and rendered song remain local-only and are excluded from package",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "wall_seconds": report["wall_seconds_including_build"], "render": render}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
