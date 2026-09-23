#!/usr/bin/env python3
"""Install the pinned Bev model and patched Prism server on macOS, Windows, or Linux CPU."""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MODEL_REV = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"
RUNTIME_TAG = "prism-b10709-9a9394a"
RUNTIME_REV = "9a9394a895b96003ca842a6041cb28ac49a108f7"
MODEL_FILE = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
MODEL_SHA = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
MODEL_SIZE = 7206168928
CODE = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("BEV_ROOT") or os.environ.get("BEV_HOME") or Path.home() / ".local/share/bev").expanduser().resolve()


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    pending = dest.with_name(dest.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Bev-portable-installer/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, pending.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    pending.replace(dest)


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or later is required")
    if os.environ.get("BEV_ROOT") and os.environ.get("BEV_HOME") and os.environ["BEV_ROOT"] != os.environ["BEV_HOME"]:
        raise SystemExit("BEV_ROOT and BEV_HOME disagree")
    for tool in ("git", "cmake"):
        if not shutil.which(tool):
            raise SystemExit(f"Missing required tool: {tool}")
    system = platform.system()
    if system not in ("Darwin", "Windows", "Linux"):
        raise SystemExit(f"Unsupported operating system: {system}")
    backend = os.environ.get("BEV_BACKEND", "metal" if system == "Darwin" else "cpu").lower()
    if backend not in ("cpu", "metal") or backend == "metal" and system != "Darwin":
        raise SystemExit("BEV_BACKEND must be cpu, or metal on macOS")
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / MODEL_FILE
    if model.exists() and sha256(model) != MODEL_SHA and model.stat().st_size >= MODEL_SIZE:
        raise SystemExit(f"Existing complete model has the wrong SHA-256: {model}")
    if not model.exists() or sha256(model) != MODEL_SHA:
        if shutil.disk_usage(ROOT).free < 11_000_000_000:
            raise SystemExit("Need at least 11 GB free disk space for a fresh installation")
        print(f"Downloading {MODEL_FILE} ({MODEL_SIZE:,} bytes)...", flush=True)
        download(f"https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/{MODEL_REV}/{MODEL_FILE}", model)
    if model.stat().st_size != MODEL_SIZE or sha256(model) != MODEL_SHA:
        raise SystemExit("Pinned model checksum or size mismatch")
    for notice in ("LICENSE", "NOTICE.txt"):
        if not (model_dir / notice).exists():
            download(f"https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/{MODEL_REV}/{notice}", model_dir / notice)
    source = ROOT / "prism-source"
    if not (source / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", RUNTIME_TAG, "https://github.com/PrismML-Eng/llama.cpp.git", str(source)], check=True)
    if run("git", "rev-parse", "HEAD", cwd=source) != RUNTIME_REV:
        raise SystemExit("Pinned Prism source revision mismatch")
    patch = CODE / "patches/prism-selected-logprobs.patch"
    if subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=source, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    build = ROOT / f"prism-build-{backend}"
    opts = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DGGML_METAL=" + ("ON" if backend == "metal" else "OFF")]
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), *opts], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "llama-server", "--config", "Release", "-j", os.environ.get("BEV_BUILD_JOBS", "4")], check=True)
    executable = "llama-server.exe" if system == "Windows" else "llama-server"
    candidates = [build / "bin" / executable, build / "bin" / "Release" / executable]
    server = next((p for p in candidates if p.exists()), None)
    if server is None:
        raise SystemExit("Patched llama-server build succeeded but executable was not found")
    version = run(str(server), "--version")
    if "9a9394a" not in version:
        raise SystemExit("Patched server version mismatch")
    venv = ROOT / ".venv"
    if not venv.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / ("Scripts/python.exe" if system == "Windows" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "-e", str(CODE)], check=True)
    print(f"Installed {backend} backend at {server}. Run: {python} {CODE / 'scripts/serve.py'}", flush=True)


if __name__ == "__main__":
    main()
