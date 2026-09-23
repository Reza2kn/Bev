---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
- fa
tags:
- gguf
- ternary
- bonsai
- decision-making
- structured-output
- zero-shot
---

# Bev

**A ternary decision engine built around Jevfire-style one-token scoring.**

[Code & documentation](https://github.com/Reza2kn/Bev) · [Release v0.1.1](https://github.com/Reza2kn/Bev/releases/tag/v0.1.1) · [Benchmarks](https://github.com/Reza2kn/Bev/blob/v0.1.1/docs/BENCHMARKS.md)

**The GGUF in this repository is a byte-identical redistribution of [Prism ML's Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf). Bev did not train or quantize these weights.** Prism supplies the ternary model, derived from Qwen3.8-27B. Bev adds a selected-token scoring extension, a local typed-decision API, portable setup, and measured evaluation. This is a model-and-software bundle, not a new fine-tune.

## What Bev does

Provide context and finite choices. Bev evaluates one next-token distribution per field, scores every candidate, and assembles structured JSON in Python. It supports:

| Primitive | Result |
|---|---|
| Boolean / enum | A typed value from the allowed set |
| Choice | The original option key and complete candidate probabilities |
| Noul | Probability assigned to true |
| Score | Probability-weighted position in an ordered rubric |

The runtime supports 2–255 candidates per field. The validated serving configuration has two slots and 16,384 tokens per field. The API rejects oversized inputs and incomplete score sets explicitly.

## Files and provenance

| Item | Value |
|---|---|
| Weights | `Ternary-Bonsai-2-27B-PQ2_0.gguf` |
| Size | **7,206,168,928 bytes** (7.21 GB; 6.71 GiB) |
| SHA-256 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |
| Immediate upstream | `prism-ml/Ternary-Bonsai-2-27B-gguf` |
| Upstream revision | `6ed5e12bf84b7a63069882c91dd9e9218647d17b` |
| Weight format | PQ2_0: ternary weights packed in two-bit slots with group scaling |
| Bev training / LoRA / new quantization | None |
| Weights license | Apache-2.0; original LICENSE and NOTICE.txt included |
| Code license | MIT; complete attribution in the source bundle |

The `model-manifest.json` records model/runtime pins and checksums. `bev-v0.1.1-source.tar.gz` contains the complete portable source, examples, tests, runtime patch, and documentation. The Python wheel packages the API only; the source installer is needed to set up the native backend. `SHA256SUMS` covers downloadable release artifacts.

## Run it

The supported setup for this release is **Linux x86_64 with an NVIDIA GPU and the pinned Prism CUDA 12.8 runtime**. It was validated on an RTX 5080 Laptop GPU with 16,303 MiB total memory. The serving process used about 8,504 MiB in one observation; this is not a peak-memory measurement or a hardware minimum guarantee.

Use the [installation guide](https://github.com/Reza2kn/Bev/blob/v0.1.1/docs/INSTALL.md) for prerequisites, then:

```sh
git clone --branch v0.1.1 https://github.com/Reza2kn/Bev.git
cd Bev
bash scripts/install.sh
bash scripts/start-services.sh

curl --fail-with-body http://127.0.0.1:18781/v1/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/support-request.json
```

The support example returns `{"route":"billing"}` in `parsed_json`, alongside complete candidate scores. Interactive API documentation is served at `http://127.0.0.1:18781/docs`. The API binds to loopback by default.

The installer verifies and downloads the original pinned Prism file. To use the identical copy from this repository instead, download it into the same model directory before installation:

```sh
export BEV_ROOT="${BEV_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/bev}"
hf download Reza2kn/Bev Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir "$BEV_ROOT/models"
```

This requires the Hugging Face CLI (`pip install huggingface_hub`). A generic GGUF viewer or stock upstream llama.cpp is not the validated runtime for PQ2_0. Use the pinned Prism fork and Bev adapter. This repository does not supply a Transformers classification head, a hosted inference endpoint, or a browser demo.

## Persian evaluation

On September 23, 2026, Bev v0.1.1 ran the complete [Jev Persian Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark) at commit `ac218d96630da9d9cc08fd897868c4d3c7048b0d`, using the original dataset, question order, batches and scorer. All **624/624 answers** were valid across **106/106 completed requests**.

| Main metric | Bev | Published Jev 1.13.0 reference |
|---|---:|---:|
| Choice: exact option | **229/240 · 95.42%** | 239/240 · 99.58% |
| Noul: true when probability ≥0.5 | **152/160 · 95.00%** | 159/160 · 99.38% |
| Score: within ±0.5 rubric levels | **70/80 · 87.50%** | 76/80 · 95.00% |
| Choice Brier ↓ | 0.067756 | 0.0112 |
| Noul Brier ↓ | 0.041239 | 0.0120 |
| Score MAE, levels ↓ | 0.191233 | 0.0709 |

Jev numbers are the benchmark author's published reference, not an independent Jev run here. The main evaluation has 480 questions; English and repeat diagnostics are separate. Bev had zero decision changes across 48 three-observation repeat groups, while some probabilities varied slightly. No training, prompt selection or calibration fitting used these cases.

The measured median was **2.136 seconds per request** and total request time **224.03 seconds**. Main/repeat batches each have six questions. The hosted Jev reference and this laptop GPU have different hardware and serving conditions. No matched full-precision or ternary speedup comparison was performed.

Aggregate results and provenance are included under `evaluations/`. Raw benchmark questions, gold labels, original scorer code, request journals and private host details are excluded. [Reproduction instructions](https://github.com/Reza2kn/Bev/blob/v0.1.1/docs/REPRODUCE.md) use the separately obtained upstream benchmark.

## Limits and intended use

Bev is intended for finite-label routing, classification and rubric evaluation where the application can define the allowed outputs. Fields are independent. Relative candidate probabilities are not calibrated confidence in correctness; confident mistakes occurred in evaluation.

The Persian benchmark is synthetic and correlated, without independent human annotation. An earlier small general diagnostic scored **7/12 MMLU** and **2/10 SimpleBench**, alongside stronger results on other small subsets. Its loaded-source attestation was incomplete; the [full report](https://github.com/Reza2kn/Bev/blob/v0.1.1/docs/BENCHMARKS.md) retains this limitation. Neither run establishes broad reliability, Jev parity, or a Decision Index rank.

One-token scoring can miss problems requiring multi-step reasoning, and the model inherits limitations and biases from its upstream models. A constrained output format does not guarantee a correct decision. No new calibration or independent production-domain validation is supplied by this release.

## Attribution

- **Jevfire / kikoncuo:** finite-choice scoring method and classification prompt, MIT.
- **Prism ML:** Ternary-Bonsai-2 model and the Prism llama.cpp fork.
- **Qwen / Alibaba Cloud:** Qwen3.8-27B base model.
- **ArmanJR and Decision Index authors:** evaluation protocols and tools, obtained separately.
- **Bev / Reza Sayar:** serving integration, typed API, packaging and evaluation, with OpenAI Codex assistance.

This independent bundle does not imply affiliation or endorsement. The original Prism Apache-2.0 LICENSE and NOTICE are preserved with the weights; the source bundle includes all code notices.
