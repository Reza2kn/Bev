# Bev v0.1.1

Bev brings Jevfire-style one-token decision scoring to Prism ML's unchanged **Ternary-Bonsai-2-27B PQ2_0** model. This release packages the inference service and reproducible configuration; it does not introduce trained or fine-tuned weights.

- **Typed decisions:** `/v1/decisions` for flat boolean/enum schemas; `/v1/systemone` for Choice, Noul and Score. Original option identities and rubric descriptions are preserved. Score is the probability-weighted rubric position.
- **Complete candidate scores:** a small Prism runtime patch returns selected raw token log probabilities while retaining full-vocabulary normalization. Missing candidates and oversized contexts fail explicitly.
- **Auditable service:** startup-captured source hashes, pinned model/runtime files, and a benchmark wrapper that checks source stability across evaluation.
- **Validation:** 53 Bev/package tests and 66 upstream Persian benchmark tests passed. The full Persian run completed **624/624 valid answers**, with **229/240 Choice**, **152/160 Noul**, and **70/80 Score within ±0.5**.
- **Measured footprint:** 7,206,168,928-byte GGUF; approximately 8,504 MiB observed serving-process GPU memory on an RTX 5080 Laptop GPU. Persian benchmark median latency was **2.136 seconds per request**, with six questions in main/repeat requests.

Candidate probabilities are relative preferences, not calibrated correctness guarantees. Fields are evaluated independently. Quality trails the published Jev reference on this Persian suite, and the earlier 120-request general diagnostic shows mixed results. No controlled full-precision comparison or ternary speedup claim is made.

See [benchmark results and limitations](https://github.com/Reza2kn/Bev/blob/v0.1.1/docs/BENCHMARKS.md), [aggregate Persian results](https://github.com/Reza2kn/Bev/blob/v0.1.1/evaluations/persian-v1/summary.json), and [provenance](https://github.com/Reza2kn/Bev/blob/v0.1.1/evaluations/persian-v1/provenance.json). Benchmark texts, labels and upstream scorer code are not bundled.
