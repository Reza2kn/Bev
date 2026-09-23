# Reproduce the Persian evaluation

The release reports the original [Jev Persian Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark) at commit `ac218d96630da9d9cc08fd897868c4d3c7048b0d`, dataset v1.0.0 and scorer v1.0.1. Obtain it separately; this repository does not redistribute its source or dataset.

Start Bev with the documented default configuration. From the Bev checkout on the Linux GPU host:

```sh
git clone https://github.com/ArmanJR/Jev-Persian-Benchmark.git research/jev-persian-benchmark
git -C research/jev-persian-benchmark checkout ac218d96630da9d9cc08fd897868c4d3c7048b0d
export BEV_ROOT="${BEV_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/bev}"
"$BEV_ROOT/.venv/bin/pip" install 'typesafe-sdk==0.7.1'
"$BEV_ROOT/.venv/bin/python" scripts/benchmark_persian.py validate
"$BEV_ROOT/.venv/bin/python" scripts/benchmark_persian.py run \
  --suite smoke --output artifacts/persian-smoke-new
"$BEV_ROOT/.venv/bin/python" scripts/benchmark_persian.py run \
  --suite full --output artifacts/persian-full-new
```

Use a fresh output path for each run. The wrapper refuses existing run/freeze paths, verifies the upstream Git revision and data hashes, and rejects modified upstream runner/scorer/data files. It uses the real TypeSafe SDK with a local placeholder key and an explicit loopback URL; it does not call the hosted Jev API. Install/runtime startup and model download are outside benchmark timing.

The full run contains 480 main questions, 48 English-instruction counterparts, and 96 repeated evaluations, arranged in 106 requests. State, questions, option order and Unicode stay unchanged. Gold labels never enter inference requests. Model and prompt selection were not performed on this benchmark.

The wrapper requires `/health` startup source hashes to match the checkout's `bev` package, and checks that provenance again at completion. Use `--service-source` after `run` when the service package is elsewhere, and `--code` before `run` when the benchmark checkout is elsewhere. If an infrastructure change is needed, retain the failed run and start a new one.

## Read the results

`summary.json` is the original scorer output. `bev-receipt.json` records completion, source stability and SHA-256 values for the run files. The wrapper also retains before/after provenance and the original runner's request journal and answer records. Keep raw inputs and gold under the benchmark's own terms when sharing results.

Read accuracy together with coverage: upstream excludes invalid answers from accuracy denominators. The published Bev run has all 624 answers valid. Choice is exact-key accuracy; Noul thresholds P(true) at 0.5; Score succeeds within ±0.5 of the authored rubric level and separately reports MAE. There is no combined official accuracy.

The public [provenance projection](../evaluations/persian-v1/provenance.json) retains code, dataset and ordered request hashes. Its [summary](../evaluations/persian-v1/summary.json) is byte-identical to the original report output. The [original evaluation wrapper](../evaluations/persian-v1/runner.py) matches the captured evaluation-wrapper hash; the current script improves checkout-path discovery without changing the evaluation logic. For exact wrapper reproduction, substitute that file and pass explicit `--code` and `--service-source` paths.

The service source files match the baseline's captured hashes. Platform scheduling and floating-point arithmetic can change probabilities; the repeat subset had stable decisions but small probability variations. A new run should record its own exact environment and results rather than assuming bitwise equality.

The [general diagnostic](BENCHMARKS.md#earlier-120-request-general-diagnostic) is historical and has a missing loaded-source attestation. It is documented separately and is not presented as a reproducible official Decision Index score.
