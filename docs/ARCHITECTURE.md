# Architecture

Bev turns finite decisions into candidate next-token scores. It uses the original Jevfire classification prompt with the loaded model's tokenizer and chat template, with thinking disabled. There is one independent field prompt and one next-token scoring step for every field.

## From a request to an answer

1. Validate the request and construct its ordered set of allowed values.
2. Map those values to unique single-token labels. At startup, verify 255 labels against the loaded tokenizer, including decode round trips.
3. Apply the loaded chat template, then tokenize the complete field prompt with special tokens handled explicitly. Check actual token capacity before inference.
4. Request the raw log probability for each candidate from the pinned Prism runtime.
5. Compute `p(i) = exp(logp(i) / T) / sum_j exp(logp(j) / T)` over allowed candidates. The default score temperature `T` is 1.
6. Return the original option key, a true probability, or an expected rubric position. Python constructs the JSON response.

The model never needs to generate a JSON document. For Choice, the highest probability identifies the answer. Noul returns `p(true)`, even when false is the winning option. Score returns `sum(i * p(i))`, with levels indexed from zero; it does not round to the most likely level. The exact rubric legend remains in the response.

## Complete scores

The [runtime extension](../patches/README.md) adds `selected_token_ids` to native `/completion`. It retains full-vocabulary normalization and bypasses the cost of serializing hundreds of thousands of vocabulary entries. It changes neither weights nor model logits. A loader shim connects the same pinned release's prebuilt CUDA backend to the host build.

On an unpatched **Prism** runtime, Bev requests top-N raw scores and retries with the full vocabulary if any candidate is absent. Missing, duplicate or nonfinite candidate scores fail the request. Grammar-restricted sampled probabilities are not substituted for raw model scores. Generic upstream llama.cpp support for this ternary format is outside the validated release.

The runtime uses float32 probability reduction. A recorded full-vocabulary sum was 1.00024960073 when re-summed in double precision; Bev allows 0.001 numerical tolerance and retains raw candidate mass. The selected-score probe compared 255 token IDs, including IDs outside the top 512, with maximum absolute log-probability difference 0.000000953674 against the full-vocabulary reference. This validates the extension against its own runtime, not against FP16 or vLLM.

## Capacity and scheduling

The validated configuration has two native slots and a 16,384-token limit per field, including one answer token. The API discovers runtime capacity. It rejects oversized field prompts rather than truncating them. Multiple fields share context text but each receives an independent forward scoring call; bounded concurrency matches available slots. Runtime cache reuse depends on the prompt and hybrid-state checkpoints.

`/v1/decisions` accepts up to 64 flat boolean/enum fields, with 2–255 candidates per enum. `/v1/systemone` supports Choice, Noul and Score; its actual limit is the backend capacity of each field prompt. Field results do not automatically condition other fields. Represent coupled actions as one joint enum or make sequential calls in the application.

`auto` currently selects `batch`. `prefill_then_batch` finishes the first field before submitting the others. There is no validated equivalent of vLLM's `aligned_prefill` or `cache_salt`; those settings are rejected. Changing score temperature changes relative probabilities, but does not change a field's argmax when all other inputs stay fixed.

## Integrity and failures

The service captures hashes of its Python source files once at startup. `/health` exposes these together with package version, runtime capabilities and model identity. The evaluation wrapper verifies the startup hashes against the declared source before a run and requires the same process provenance afterward.

Invalid input returns HTTP 422. Failed backend health checks or incomplete initialization return HTTP 503; a backend failure during a decision returns HTTP 502, and an inference timeout returns HTTP 504. A field failure cancels and drains sibling requests; the API does not substitute a default decision or return a silently incomplete result.

Candidate-relative probabilities are not calibrated estimates of correctness. A high relative score can be wrong, and one-token classification has no internal multi-step reasoning budget. No new fine-tuning, calibration fit, weight conversion, or knowledge distillation was performed for this release.
