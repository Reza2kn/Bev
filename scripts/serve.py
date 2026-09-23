#!/usr/bin/env python3
"""Run Bev's native backend and API together in the foreground on any desktop OS."""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("BEV_ROOT") or os.environ.get("BEV_HOME") or Path.home() / ".local/share/bev").expanduser().resolve()
CODE = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/Ternary-Bonsai-2-27B-PQ2_0.gguf"


def port_owned(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(1)
        return client.connect_ex(("127.0.0.1", port)) == 0


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> None:
    system = platform.system()
    backend = os.environ.get("BEV_BACKEND", "metal" if system == "Darwin" else "cpu").lower()
    executable = "llama-server.exe" if system == "Windows" else "llama-server"
    build = ROOT / f"prism-build-{backend}" / "bin"
    server = next((p for p in (build / executable, build / "Release" / executable) if p.exists()), None)
    python = ROOT / ".venv" / ("Scripts/python.exe" if system == "Windows" else "bin/python")
    if server is None or not MODEL.is_file() or not python.is_file():
        raise SystemExit("Portable installation incomplete; run scripts/install-portable.py first")
    llama_port = int(os.environ.get("BEV_LLAMA_PORT", "18780"))
    api_port = int(os.environ.get("BEV_PORT", "18781"))
    if port_owned(llama_port) or port_owned(api_port):
        raise SystemExit("A service already owns a Bev port; choose BEV_LLAMA_PORT and BEV_PORT")
    env = os.environ.copy()
    env["BEV_ROOT"] = str(ROOT)
    env["BEV_LLAMA_URL"] = f"http://127.0.0.1:{llama_port}"
    env["BEV_MODEL"] = os.environ.get("BEV_MODEL", "bev-bonsai-27b")
    ctx = os.environ.get("BEV_CTX_SIZE", "4096")
    slots = os.environ.get("BEV_PARALLEL", "1")
    args = [str(server), "--model", str(MODEL), "--alias", env["BEV_MODEL"], "--host", "127.0.0.1", "--port", str(llama_port), "--n-gpu-layers", "99" if backend == "metal" else "0", "--ctx-size", ctx, "--parallel", slots, "--batch-size", "128", "--ubatch-size", "128", "--threads", "4", "--threads-batch", "4", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja", "--reasoning", "off", "--no-context-shift", "--no-webui"]
    native = subprocess.Popen(args, env=env)
    api = None
    try:
        deadline = time.monotonic() + int(os.environ.get("BEV_START_TIMEOUT_SECONDS", "180"))
        while time.monotonic() < deadline:
            if native.poll() is not None:
                raise RuntimeError(f"Native server exited with code {native.returncode}")
            if healthy(f"http://127.0.0.1:{llama_port}/health"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Native model did not become ready before timeout")
        api = subprocess.Popen([str(python), "-m", "uvicorn", "bev.app:app", "--host", "127.0.0.1", "--port", str(api_port), "--no-access-log"], cwd=CODE, env=env)
        while api.poll() is None and not healthy(f"http://127.0.0.1:{api_port}/health"):
            time.sleep(0.5)
        if api.poll() is not None:
            raise RuntimeError(f"API exited with code {api.returncode}")
        print(f"Bev ready: http://127.0.0.1:{api_port}/docs", flush=True)
        while native.poll() is None and api.poll() is None:
            time.sleep(1)
        raise RuntimeError("A Bev service exited unexpectedly")
    except KeyboardInterrupt:
        pass
    finally:
        for process in (api, native):
            if process and process.poll() is None:
                process.terminate()
        for process in (api, native):
            if process:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
