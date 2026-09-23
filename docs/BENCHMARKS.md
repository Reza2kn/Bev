# Benchmarks

Bev v0.1.1 completed the pinned **Jev Persian Benchmark** with **95.42% Choice accuracy, 95.00% Noul accuracy and 87.50% Score success**. All 624 planned answers were valid. An earlier 120-request general diagnostic showed mixed quality and is documented separately below. Neither evaluation is an official Decision Index run.

Bev is an inference system using unchanged Ternary-Bonsai-2-27B weights. These results do not represent a newly trained model, a controlled comparison with full-precision Jevfire, or a measured speedup from ternary weights.

## Persian benchmark: v0.1.1

Measured on September 23, 2026, using [Jev-Persian-Benchmark at `ac218d9`](https://github.com/ArmanJR/Jev-Persian-Benchmark/tree/ac218d96630da9d9cc08fd897868c4d3c7048b0d), dataset revision 1.0.0 and scorer 1.0.1. The main suite contains 80 scenarios with six questions each: 240 Choice, 160 Noul and 80 Score. English-instruction and repeated observations are separate diagnostics.

| Main metric | Bev v0.1.1 | Published Jev 1.13.0 reference |
| --- | ---: | ---: |
| Choice: exact option | **229/240 · 95.42%** | 239/240 · 99.58% |
| Noul: true when probability ≥0.5 | **152/160 · 95.00%** | 159/160 · 99.38% |
| Score: within ±0.5 rubric levels | **70/80 · 87.50%** | 76/80 · 95.00% |
| Choice Brier score ↓ | 0.067756 | 0.0112 |
| Noul Brier score ↓ | 0.041239 | 0.0120 |
| Score MAE, rubric levels ↓ | 0.191233 | 0.0709 |

The reference values are the benchmark author's [published September 22 run](https://github.com/ArmanJR/Jev-Persian-Benchmark/blob/ac218d96630da9d9cc08fd897868c4d3c7048b0d/README.md#performance), not a Jev API run performed for Bev. No full-precision Jevfire control was evaluated, so the accuracy gap cannot be attributed specifically to ternary compression. Choice and Noul Brier scores use different scales; compare each primitive with its own reference.

Noul precision is 98.65% and recall is 91.25%, with seven false negatives and one false positive. Score normalized MAE is 0.095616 on the 0–2 rubric. The scorer does not define a combined accuracy across these different primitives.

### Coverage and diagnostics

| Check | Result |
| --- | --- |
| Answers | **624/624 valid**; no failed or unfinished answers |
| Requests | **106/106 completed**; no failures or model mismatches |
| Main evaluation | 480 answers in 80 requests |
| English counterparts | 48 answers in 10 requests |
| Additional repeated observations | 96 answers in 16 requests |
| Invariant pairs | Both correct in **23/24**; expected decision relation in 23/24 |
| Contrast pairs | Both correct in **24/24**; expected decision relation in 24/24 |
| Repeatability | **Zero decision changes** across 48 complete three-observation groups |

The native scorer excludes invalid answers from accuracy denominators and reports failures separately. Here every answer is valid, so the main denominators remain 240/160/80 without exclusions. Repeats and pairs are correlated observations, not extra independent examples. Six repeated Choice distributions varied slightly, with a maximum individual-option probability range of 0.00753; stable decisions do not imply bit-for-bit identical probabilities.

English counterparts retain the Persian state text and change only the fields specified by the benchmark:

| Matched subset | Persian instructions | English instructions |
| --- | ---: | ---: |
| Choice | 29/30 | 28/30 |
| Noul | 10/10 | 9/10 |
| Score | 7/8 | 8/8 |

Five decisions changed: two improved and three worsened. This small subset provides mixed evidence about instruction language, not a basis for universally switching to English.

### Category results

Each category has 24 Choice, 16 Noul and eight Score questions. Score counts use the same ±0.5 criterion.

| Category | Choice | Noul | Score |
| --- | ---: | ---: | ---: |
| Intent | 23/24 | 14/16 | 8/8 |
| Sentiment | 24/24 | 16/16 | 8/8 |
| Reading | 24/24 | 16/16 | 6/8 |
| Negation | 24/24 | 14/16 | 7/8 |
| Idioms | 22/24 | 15/16 | 8/8 |
| Pragmatics | 24/24 | 16/16 | 8/8 |
| Scenario decisions | 20/24 | 13/16 | 6/8 |
| Moderation | 23/24 | 16/16 | 6/8 |
| Semantic matching | 21/24 | 16/16 | 7/8 |
| Extraction | 24/24 | 16/16 | 6/8 |

Sentiment and pragmatics passed every main question. Scenario decisions were weakest, including conditional routing, permissions and severity. Four of ten Score failures had the correct most-likely rubric level but failed the required probability-weighted score criterion; the native semantics were preserved. Candidate-relative confidence is uncalibrated: the ≥0.8-confidence subset still contained one incorrect Choice and two incorrect Scores.

### Execution and provenance

The [evaluation wrapper](../scripts/benchmark_persian.py) uses the pinned upstream loader, planner, runner and scorer unchanged, through the real `typesafe-sdk==0.7.1`. State text, instructions, option keys, descriptions, order and Unicode are preserved. Only state, questions and model reach Bev; gold labels and evaluation metadata stay with the scorer.

Ordered request hashes were frozen before inference. Startup-captured service source hashes matched the declared package before the run and remained unchanged afterward. No prompt selection, calibration fitting, training or correctness-based retries were performed on this benchmark. A separate 12-question smoke run preceded the full run. The implementation passed 46 Bev tests; the pinned benchmark passed its 66 upstream tests.

Public evidence contains aggregate measurements and provenance:

- [Native summary](../evaluations/persian-v1/summary.json)
- [Pins, hashes and integrity receipt](../evaluations/persian-v1/provenance.json)

The source benchmark is a small synthetic suite, authored and reviewed using the same authoring model, without independent human annotation. Its explicit clues, short contexts and related questions limit generalization to natural Persian traffic or broader reasoning. Error inspection also makes these cases development evidence for any future method selection; subsequent quality claims need separate untouched evaluation data.

## Latency and memory

Both runs used one **NVIDIA GeForce RTX 5080 Laptop GPU**, reporting 16,303 MiB total GPU memory, with two native inference slots and a 16,384-token context limit per field. Another GPU service remained active. The serving process used approximately **8,504 MiB** in a recorded observation; this is a process-memory snapshot, not a peak-memory measurement or a minimum hardware guarantee.

| Measurement | Persian full run | Earlier general diagnostic |
| --- | ---: | ---: |
| Requests | 106 | 120 |
| Total recorded request time | 224.03 s | 286.32 s |
| Median request latency | 2.136 s | 0.619 s |
| Mean request latency | 2.113 s | 2.386 s |
| Maximum request latency | 2.343 s | Not reported |
| 95th percentile latency | Not reported | 15.072 s |

Times measure serial local HTTP requests, including prompt preparation and scheduling; model loading is outside these measurements. Main and repeat Persian requests have six questions, while some English batches are smaller. The general sample has variable request shapes; its many-field home-appliance requests took a median 15.35 seconds. Fields are scored independently, so neither table reports single-field latency or token-generation throughput.

Weights are the [Prism ML PQ2_0 GGUF at `6ed5e12`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b), **7,206,168,928 bytes**, SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`. The runtime is the [Prism llama.cpp fork at `9a9394a`](https://github.com/PrismML-Eng/llama.cpp/tree/9a9394a895b96003ca842a6041cb28ac49a108f7), with Bev's selected-log-probability patch. The patch exposes requested raw scores without changing weights or logits.

There is no matched hardware, request, cache and concurrency comparison against Jev or Jevfire. These measurements establish neither a ternary speedup nor a guaranteed memory/accuracy advantage in another deployment.

## Earlier 120-request general diagnostic

This was a fixed, source-derived sample using the [Decision Index reproduction kit at `52a6989`](https://github.com/apolinario/decision-index/tree/52a698928a9ae5bdf16b75687c903871db29c6e5). It was **not the official frozen Decision Index suite or its compatibility smoke set**. The advertised full dataset was inaccessible with the available access. Whole source groups were selected before inference using seed 20260923, hash ordering and round-robin allocation across ten benchmarks.

All **120/120 requests** returned valid distributions, covering 335 fields, up to 19 fields per request and 151 options per field. There were no errors or unsupported cases. The selected sample SHA-256 is `db4fed9913bb52d3a4962bf896f92a69602c9cc19a8de64b3cd06a2b9bf0e390`.

| Sample | Requests | Observed native metric |
| --- | ---: | --- |
| BANKING77 | 13 | Macro-F1 **0.4118**; 7/13 correct |
| CLINC150+OOS | 13 | Macro-F1 **1.0000**; 13/13 correct |
| Home appliance simulator | 12 | **5/12** whole cases exact; 94.47% field accuracy |
| MMLU | 12 | **7/12** correct |
| ARC-Easy | 12 | **12/12** correct |
| ARC-Challenge | 12 | **12/12** correct |
| WinoGrande | 12 | **10/12** correct |
| HellaSwag | 12 | **11/12** correct |
| SimpleBench public sample | 10 | **2/10** correct |
| iSarcasmEval | 12 | Mixed languages/subtasks; no single comparable aggregate |

These tiny samples show mixed quality, particularly on SimpleBench and whole-case home-appliance decisions. A perfect result on 12 or 13 cases is not a reliable full-benchmark estimate. The results cannot be averaged into a Decision Index score or compared to the leaderboard's full-suite entrants. No benchmark-specific training or prompt tuning was performed.

This run used the initial API process, whose loaded-source hashes were not captured because the original collector inspected the wrong directory. Two subsequent reliability fixes—numerical tolerance and cancellation after sibling failure—were not loaded during that run. The model, prompts and scoring method were unchanged, but the old diagnostic is **not an attested evaluation of the released v0.1.1 service**. Its provenance gap is preserved rather than filled retroactively.

See the [diagnostic summary](../evaluations/general-diagnostic-v1/summary.json) and [selection/provenance record](../evaluations/general-diagnostic-v1/provenance.json). These artifacts contain no raw benchmark inputs or answer journals.

## Benchmark content and licensing

This release publishes Bev's wrapper, aggregate measurements and provenance; it does not redistribute benchmark inputs, gold labels, raw answer journals or upstream scorer code. The pinned Persian benchmark repository has no explicit license declaration. Public availability should not be read as a redistribution grant. Obtain upstream material separately under its applicable terms; Bev's code license does not cover it. The general diagnostic's source datasets likewise retain their respective upstream terms.
