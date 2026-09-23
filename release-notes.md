# Bev v0.1.2

**Memory first:** the 7.21 GB GGUF reached **7.28 GiB peak resident system RAM** in a short CPU-only inference with one 4,096-token slot on Stallion. The earlier **8.30 GiB** number is an observed **GPU-memory** snapshot for the separate Linux CUDA benchmark setup with two 16,384-token slots; it is not host RAM or a measured peak. Allow additional headroom.

Bev brings Jevfire-style one-token decision scoring to Prism ML's unchanged **Ternary-Bonsai-2-27B PQ2_0** model. This release packages the inference service and reproducible configuration; it does not introduce trained or fine-tuned weights.

- **Typed decisions:** `/v1/decisions` for flat boolean/enum schemas; `/v1/systemone` for Choice, Noul and Score. Original option identities and rubric descriptions are preserved. Score is the probability-weighted rubric position.
- **Complete candidate scores:** a small Prism runtime patch returns selected raw token log probabilities while retaining full-vocabulary normalization. Missing candidates and oversized contexts fail explicitly.
- **Auditable service:** startup-captured source hashes, pinned model/runtime files, and a benchmark wrapper that checks source stability across evaluation.
- **Validation:** 46 Bev tests passed and 7 optional-dependency tests were skipped on macOS; 14/14 real Metal API checks passed. The v0.1.1 Linux CUDA run previously passed 53 Bev/package tests and 66 upstream Persian benchmark tests. The full Persian run completed **624/624 valid answers**, with **229/240 Choice**, **152/160 Noul**, and **70/80 Score within ±0.5**.
- **Portable build:** the same pinned Prism source and scoring patch can now be built through `scripts/install-portable.py` for macOS Metal or Linux/Windows CPU. **14/14 API smoke checks passed on an Apple M2 Mac**; Windows remains an untested source-build path. The Linux CUDA path remains available.
- **Measured footprint:** 7,206,168,928-byte GGUF; 7,630,416 KiB CPU process high-water RSS on Stallion and 8,504 MiB observed GPU memory in the original CUDA setup. Persian benchmark median latency was **2.136 seconds per request**, with six questions in main/repeat requests.

Candidate probabilities are relative preferences, not calibrated correctness guarantees. Fields are evaluated independently. Quality trails the published Jev reference on this Persian suite, and the earlier 120-request general diagnostic shows mixed results. No controlled full-precision comparison or ternary speedup claim is made.

See [benchmark results and limitations](https://github.com/Reza2kn/Bev/blob/v0.1.2/docs/BENCHMARKS.md), [aggregate Persian results](https://github.com/Reza2kn/Bev/blob/v0.1.2/evaluations/persian-v1/summary.json), and [provenance](https://github.com/Reza2kn/Bev/blob/v0.1.2/evaluations/persian-v1/provenance.json). Benchmark texts, labels and upstream scorer code are not bundled.
