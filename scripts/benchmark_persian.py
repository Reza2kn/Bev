#!/usr/bin/env python3
"""Run the pinned Persian benchmark unchanged against local Bev on Stallion.

Uses the real pinned TypeSafe SDK and upstream dataset, planner, runner and scorer.
No hosted model endpoint, external judge, prompt tuning, or gold-bearing payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

UPSTREAM_REVISION = "ac218d96630da9d9cc08fd897868c4d3c7048b0d"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def canonical_sha(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def upstream(path):
    path = Path(path).resolve()
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"Persian benchmark revision differs: {revision}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--", "jev_benchmark", "data"], text=True)
    if dirty.strip():
        raise RuntimeError("Upstream runner, scorer or dataset has local changes")
    sys.path.insert(0, str(path))
    from jev_benchmark.data import load_dataset

    return path, load_dataset(path / "data")


def health(client, base_url):
    response = client.get(base_url.rstrip("/") + "/health", timeout=30)
    response.raise_for_status()
    value = response.json()
    if value.get("status") != "ok":
        raise RuntimeError("Bev is not healthy")
    return value


def attestation(value, source):
    provenance = value.get("service_provenance", {})
    recorded = provenance.get("startup_source_sha256")
    if not recorded:
        raise RuntimeError("Health must expose startup-captured service_provenance.startup_source_sha256")
    files = sorted(Path(source).glob("*.py"))
    actual = {p.name: sha(p) for p in files}
    if not actual or recorded != actual:
        raise RuntimeError("Startup-captured source hashes differ from the declared service source")
    return provenance


def execute(args):
    import httpx2
    from typesafe_sdk import RetryPolicy, TypeSafeClient
    from jev_benchmark.report import build_report
    from jev_benchmark.runner import plan_jobs, run

    if platform.system() != "Linux":
        raise RuntimeError("Execute this evaluation on Stallion, not the local workstation")
    parsed = urlparse(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only the local Stallion HTTP endpoint is permitted")
    if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("base-url must be a plain loopback origin")
    output = Path(args.output).resolve()
    freeze_path = output.with_name(output.name + ".freeze.json")
    if output.exists() or freeze_path.exists():
        raise FileExistsError("Use a new output directory; old run artifacts are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(args.dataset, args.suite, args.model)
    for job in jobs:
        if set(job["request"]) != {"state", "questions", "model"}:
            raise RuntimeError("Unexpected payload fields")
        for question in job["request"]["questions"].values():
            if not set(question) <= {"type", "instructions", "criteria"}:
                raise RuntimeError("Metadata or gold leaked into question payload")
    transport = httpx2.Client(timeout=20.0, trust_env=False, follow_redirects=False)
    client = TypeSafeClient(api_key="local-bev-placeholder", base_url=args.base_url,
                           model=args.model, timeout=20.0,
                           retry=RetryPolicy(max_retries=2, timeout=45.0), http_client=transport)
    before = health(transport, args.base_url)
    if before.get("model") != args.model:
        raise RuntimeError("Requested model does not match the served health model")
    startup = attestation(before, args.service_source)
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "ArmanJR/Jev-Persian-Benchmark", "upstream_revision": UPSTREAM_REVISION,
        "suite": args.suite, "requested_model": args.model, "base_url": args.base_url,
        "dataset_manifest": args.dataset.manifest,
        "planned_requests": len(jobs),
        "planned_questions": sum(len(j["request"]["questions"]) for j in jobs),
        "ordered_request_sha256": [{"job_id": j["id"], "sha256": canonical_sha(j["request"])} for j in jobs],
        "health_before": before, "startup_service_provenance": startup,
        "upstream_code_sha256": {p.name: sha(p) for p in sorted((args.code / "jev_benchmark").glob("*.py"))},
        "evaluation_wrapper_sha256": sha(__file__),
        "packages": {name: version(name) for name in ("typesafe-sdk", "httpx2", "pydantic")},
        "transport": "Real TypeSafe SDK to explicit loopback endpoint; placeholder key; environment proxies disabled; 20-second timeout, two transport retries, 45-second retry budget",
        "comparison_limits": ["One published synthetic Persian suite; correlated questions and paired scenarios", "No calibration training or prompt selection on these cases", "Local laptop-GPU HTTP timing is not a controlled comparison to hosted Jev timing", "Choice/Score confidence is the maximum candidate probability, not calibrated correctness"],
    }
    write_json(freeze_path, frozen)
    changed = None
    try:
        # Upstream writes run.json (including local gold) before any inference;
        # only each job's allowlisted request is passed to the SDK.
        run(args.dataset, output, suite=args.suite, model=args.model, client=client)
    finally:
        if output.exists():
            write_json(output / "bev-provenance-before.json", frozen)
            try:
                after = health(transport, args.base_url)
                stable = attestation(after, args.service_source) == startup
                changed = None if stable else "API startup provenance changed during evaluation"
                write_json(output / "bev-provenance-after.json", {"health": after, "stable_service": stable})
            except Exception as exc:
                changed = f"Cannot attest stable final service: {type(exc).__name__}: {exc}"
                write_json(output / "bev-provenance-after.json", {"stable_service": False, "error": changed})
        client.close()
    summary = build_report(output)
    receipt = {
        "upstream_revision": UPSTREAM_REVISION, "stable_service": changed is None,
        "provenance_error": changed,
        "files_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        "comparison_complete": summary["completion"]["complete"] and not summary["model_mismatch_requests"] and changed is None,
        "warning": "Accuracy denominators exclude invalid answers upstream; compare headline accuracy only alongside complete coverage.",
    }
    write_json(output / "bev-receipt.json", receipt)
    # Summaries only: do not print question text, labels, rationales or failures.
    print(json.dumps({"completion": summary["completion"], "main": summary["main"],
                      "pairs": summary["pairs"], "latency_seconds": summary["latency_seconds"],
                      "comparison_complete": receipt["comparison_complete"], "output": str(output)}, ensure_ascii=False), flush=True)
    return 0 if receipt["comparison_complete"] else 2


def main():
    root = Path(__file__).resolve().parents[1]
    research_root = root / "research" if (root / "research/jev-persian-benchmark").is_dir() else root.parent / "research"
    service_source = root / "bev" if (root / "bev").exists() else root / "app/bev"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, default=research_root / "jev-persian-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    p = sub.add_parser("run")
    p.add_argument("--suite", choices=["smoke", "full"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="bev-bonsai-27b")
    p.add_argument("--base-url", default="http://127.0.0.1:18781")
    p.add_argument("--service-source", default=str(service_source))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("typesafe_sdk").setLevel(logging.CRITICAL)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    args.code, args.dataset = upstream(args.code)
    if version("typesafe-sdk") != "0.7.1":
        raise RuntimeError("typesafe-sdk==0.7.1 is required")
    if args.command == "validate":
        print(json.dumps({"revision": UPSTREAM_REVISION, "dataset_revision": args.dataset.manifest["revision"],
                          "main_counts": args.dataset.manifest["counts"], "files_verified": args.dataset.manifest["sha256"]}))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
