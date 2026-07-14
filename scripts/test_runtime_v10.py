#!/usr/bin/env python3
"""Stress resident v1.0 candidate: repeat, concurrency, failure recovery, restart, memory."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def process_tree(pid: int) -> set[int]:
    found, pending = set(), [pid]
    while pending:
        current = pending.pop()
        if current in found or not Path(f"/proc/{current}").exists(): continue
        found.add(current)
        children = Path(f"/proc/{current}/task/{current}/children")
        if children.exists(): pending.extend(int(value) for value in children.read_text().split())
    return found


def memory_mb(pid: int) -> float:
    total = 0
    for current in process_tree(pid):
        status = Path(f"/proc/{current}/status")
        if not status.exists(): continue
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"): total += int(line.split()[1]); break
    return total / 1024


def wait_health(url: str, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(urllib.request.urlopen(url + "/health", timeout=2).read())
        except Exception:
            time.sleep(.25)
    raise TimeoutError("resident renderer did not become healthy")


def post(url: str, score: dict) -> tuple[str, float]:
    request = urllib.request.Request(url + "/render", data=json.dumps(score).encode(), headers={"Content-Type": "application/json"}, method="POST")
    start = time.monotonic()
    body = urllib.request.urlopen(request, timeout=600).read()
    return hashlib.sha256(body).hexdigest(), time.monotonic() - start


def start_server(root: Path, port: int, log) -> subprocess.Popen:
    environment = os.environ | {"PYTHONPATH": str(root / "src")}
    command = [sys.executable, "-m", "gyu_singer.cli", "--backend", "gyu-singer-v0.8", "--reference", "data/processed/master/216.wav", "serve", "--port", str(port)]
    return subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None: return
    os.kill(process.pid, signal.SIGINT)
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="artifacts/reports/runtime_v10_stress.json")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = {language: json.loads((root / f"artifacts/reports/openutau_v09/{language}.json").read_text()) for language in ("ko", "en", "ja")}
    url = f"http://127.0.0.1:{args.port}"
    log_path = output.with_name("runtime_v10_server.log")
    with log_path.open("w") as log:
        first = start_server(root, args.port, log)
        try:
            first_health = wait_health(url)
            warmup = post(url, scores["ko"])
            memory_before = memory_mb(first.pid)
            repeated = [post(url, scores["ko"]) for _ in range(args.repeats)]
            multilingual = {language: post(url, score) for language, score in scores.items()}
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                concurrent_results = list(pool.map(lambda _: post(url, scores["ko"]), range(4)))
            invalid = scores["ko"] | {"language": "invalid"}
            try: post(url, invalid); failure_status = None
            except urllib.error.HTTPError as error: failure_status = error.code
            recovered = post(url, scores["ko"])
            memory_after = memory_mb(first.pid)
        finally:
            first_children = process_tree(first.pid)
            stop_server(first)
        first_shutdown_clean = all(not Path(f"/proc/{pid}").exists() for pid in first_children)
        second = start_server(root, args.port, log)
        try:
            restart_health = wait_health(url)
            restarted = post(url, scores["ko"])
        finally:
            second_children = process_tree(second.pid)
            stop_server(second)
        second_shutdown_clean = all(not Path(f"/proc/{pid}").exists() for pid in second_children)
    hashes = [value[0] for value in repeated]
    latencies = [value[1] for value in repeated]
    checks = {
        "workers_healthy": first_health.get("status") == restart_health.get("status") == "ok",
        "twenty_repeats_stable": len(hashes) == args.repeats and len(set(hashes)) == 1,
        "multilingual_rendered": len(multilingual) == 3 and all(value[0] for value in multilingual.values()),
        "concurrent_requests_safe": len(concurrent_results) == 4 and len({value[0] for value in concurrent_results}) == 1,
        "failed_then_valid": failure_status == 400 and recovered[0] == hashes[0],
        "restart_stable": restarted[0] == hashes[0],
        "shutdown_clean": first_shutdown_clean and second_shutdown_clean,
        "memory_growth_acceptable": memory_after - memory_before < 256,
    }
    report = {
        "repeats": args.repeats, "repeat_unique_sha256": len(set(hashes)),
        "warmup_seconds": warmup[1], "repeat_latency_seconds": {"min": min(latencies), "mean": sum(latencies) / len(latencies), "max": max(latencies)},
        "multilingual": {key: {"sha256": value[0], "seconds": value[1]} for key, value in multilingual.items()},
        "concurrent_latency_seconds": [value[1] for value in concurrent_results], "failure_status": failure_status,
        "memory_mb": {"before": memory_before, "after": memory_after, "growth": memory_after - memory_before},
        "health_before": first_health, "health_after_restart": restart_health,
        "checks": checks, "pass": all(checks.values()), "server_log": str(log_path),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
