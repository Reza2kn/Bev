"""Portable launcher and ownership checks; these never launch model inference."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
pytestmark = pytest.mark.skipif(sys.platform != "linux" or not shutil.which("bash"), reason="Linux Bash runtime scripts")


def environment(**updates):
    env = {key: value for key, value in os.environ.items() if not key.startswith("BEV_")}
    env.update({key: str(value) for key, value in updates.items()})
    return env


def read_paths(env):
    return subprocess.run(
        ["bash", "-c", 'set -e; source "$1"; printf "%s\\n" "$BEV_ROOT" "$BEV_HOME" "$BEV_CODE"', "bash", str(SCRIPTS / "runtime-env.sh")],
        env=env, capture_output=True, text=True,
    )


def test_default_runtime_root_follows_xdg_and_code_follows_checkout(tmp_path):
    result = read_paths(environment(XDG_DATA_HOME=tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(tmp_path / "bev"), str(tmp_path / "bev"), str(SCRIPTS.parent)]


def test_legacy_home_override_is_preserved(tmp_path):
    result = read_paths(environment(BEV_HOME=tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:2] == [str(tmp_path), str(tmp_path)]


def test_conflicting_root_overrides_fail(tmp_path):
    result = read_paths(environment(BEV_HOME=tmp_path / "old", BEV_ROOT=tmp_path / "new"))
    assert result.returncode != 0
    assert "disagree" in result.stderr


def fake_executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '#!/usr/bin/env python3\nimport json, os, sys\n'
        'print(json.dumps({"argv":sys.argv[1:],"cwd":os.getcwd(),'
        '"url":os.environ.get("BEV_LLAMA_URL"),"model":os.environ.get("BEV_MODEL"),'
        '"backend":os.environ.get("GGML_BACKEND_PATH")}))\n'
    )
    path.chmod(0o755)


def test_native_launcher_preserves_paths_with_spaces_and_configuration(tmp_path):
    root = tmp_path / "runtime with spaces"
    fake_executable(root / "prism-build/bin/llama-server")
    result = subprocess.run(
        ["bash", str(SCRIPTS / "start-llama.sh")],
        env=environment(BEV_ROOT=root, BEV_LLAMA_PORT="28780", BEV_MODEL="test-alias", BEV_CTX_SIZE="4096", BEV_PARALLEL="1"),
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    args = data["argv"]
    assert args[args.index("--model") + 1] == str(root / "models/Ternary-Bonsai-2-27B-PQ2_0.gguf")
    assert args[args.index("--port") + 1] == "28780"
    assert args[args.index("--alias") + 1] == "test-alias"
    assert args[args.index("--ctx-size") + 1] == "4096"
    assert args[args.index("--parallel") + 1] == "1"
    assert data["backend"] == str(root / "prism-build/bin/libbev-cuda-loader.so")


def test_api_launcher_propagates_ports_and_source_directory(tmp_path):
    root = tmp_path / "runtime"
    code = tmp_path / "source with spaces"
    code.mkdir()
    fake_executable(root / ".venv/bin/python")
    result = subprocess.run(
        ["bash", str(SCRIPTS / "start-api.sh")],
        env=environment(BEV_ROOT=root, BEV_CODE=code, BEV_PORT="28781", BEV_LLAMA_PORT="28780", BEV_MODEL="custom"),
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    assert data["cwd"] == str(code)
    assert data["argv"] == ["-m", "uvicorn", "bev.app:app", "--host", "127.0.0.1", "--port", "28781", "--no-access-log"]
    assert data["url"] == "http://127.0.0.1:28780"
    assert data["model"] == "custom"


def test_stop_refuses_unrelated_live_pid(tmp_path):
    (tmp_path / "logs").mkdir()
    unrelated = subprocess.Popen(["sleep", "30"])
    try:
        (tmp_path / "logs/api.pid").write_text(str(unrelated.pid))
        result = subprocess.run(
            ["bash", str(SCRIPTS / "stop-services.sh")],
            env=environment(BEV_ROOT=tmp_path), capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "unexpected process" in result.stderr
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_start_services_refuses_external_backend_url(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPTS / "start-services.sh")],
        env=environment(BEV_ROOT=tmp_path, BEV_LLAMA_URL="http://remote.invalid:9999"),
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "external BEV_LLAMA_URL" in result.stderr
