# API

Default base URL: `http://127.0.0.1:18781`. Interactive OpenAPI documentation is at
`/docs`. Bev scores each requested finite-choice field independently and assembles
the result in application code. Every option is scored; the model’s generated
text is not used as a substitute for missing scores.

## `GET /health`

Returns `status`, actual `model`, runtime identity, supported primitives,
`max_model_len` per slot, `max_choices`, `backend_slots`, selected-token capability,
and `service_provenance`. The provenance includes package source hashes captured
once at process startup, process ID, startup time, and package version. Copying
new source files onto a running machine does not change its reported loaded
revision. Native unavailability returns HTTP 503.

## `POST /v1/decisions`

The JEVfire-style interface accepts a nonempty context and 1–64 independent
boolean or enum fields. Enums require 2–255 unique nonblank string choices.

```bash
curl --fail http://127.0.0.1:18781/v1/decisions \
  -H 'Content-Type: application/json' \
  --data '{
    "context": "The order has shipped. It has not been delivered.",
    "schema": {
      "delivered": {"type": "boolean", "description": "Has it been delivered?"}
    }
  }'
```

The response includes `parsed_json`, detailed `fields`, actual backend/model and
strategy metadata, per-field timings, logical token usage, and uncalibrated score
metadata. Boolean alternatives are true and false; enum outputs preserve the
supplied strings exactly. Each field reports its selected value, surrogate label,
token ID, candidate-relative probability, every candidate’s raw full-vocabulary
logprob, and `candidate_probability_mass`.

| Optional parameter | Default | Behavior |
|---|---|---|
| `strategy` | `auto` | Maps to `batch`, implemented as bounded concurrent native requests |
| `strategy: batch` | — | Scores fields concurrently up to native slot capacity |
| `strategy: prefill_then_batch` | — | Scores one real field first, then the rest concurrently to permit prefix-cache reuse |
| `score_temperature` | `1.0` | Rescales candidate logprobs before normalization; allowed range 0.05–10 |
| `min_probability` | absent | Sets `value` to null below the threshold; retains `selected_value` and all scores |

`aligned_prefill` and non-null `cache_salt` are rejected with HTTP 422 because
their upstream vLLM semantics are not implemented here. Fields have no access to
other fields’ selected answers. Context is limited to 100,000 characters and the
loaded token window; field descriptions to 4,000 characters, names to 128, and
individual enum values to 1,000. These are the native decision contract bounds.

## `POST /v1/systemone`

This interface supports Choice, Noul, and Score. `state` is a string or JSON
object; it may be empty. `questions` is a nonempty dictionary. The accepted model
is the loaded alias (default `bev-bonsai-27b`) or the literal `default`; other
model names are rejected rather than silently redirected.

```json
{
  "model": "bev-bonsai-27b",
  "state": {"text": "The customer requests a refund, not a replacement."},
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which action is requested?",
      "criteria": {"refund": "Refund the payment", "replace": "Replace the item"}
    },
    "wants_refund": {
      "type": "noul",
      "instructions": "Does the customer request a refund?"
    },
    "support": {
      "type": "score",
      "instructions": "How strongly does the text support a refund request?",
      "criteria": ["Contradicted", "Not stated", "Explicitly supported"]
    }
  }
}
```

Every answer appears under its original question key in `answers`:

- **Choice:** `{"type":"choice","choice":<key>,"probabilities":{<every key>:p},"confidence":p_max}`.
  The original criterion keys and descriptions are preserved separately in the
  prompt. There must be 2–255 choices.
- **Noul:** `{"type":"noul","noul":p_true}`. No criteria are supplied. This is
  always the probability of true, including when false is the winning alternative.
- **Score:** `{"type":"score","score":weighted_mean,"probabilities":{"0":p0,...},"legend":{"0":description0,...},"confidence":p_max}`.
  Supply 2–255 ordered, nonblank rubric descriptions. The score is
  `sum(index * probability)`, not the argmax index. Repeated descriptions retain
  separate rubric indices and the exact original legend.

Question instructions may be strings or JSON objects. Choice descriptions may
also be strings or JSON objects; Score descriptions are strings. Unlike
`/v1/decisions`, this adapter has no arbitrary 64-question or character cap, but
each rendered field must fit the loaded model context. It still evaluates fields
independently.

`confidence` is the maximum candidate-relative probability, **not calibrated
correctness confidence**. Top-level `scores_are_calibrated` is false and
`confidence_definition` records this meaning. `evaluation_trace` contains the
underlying fields and timing details. `usage.input_tokens` and
`usage.output_tokens` count logical prompt and one-token scoring work, including
any score-retrieval retries; they do not count words in the JSON response and are
not a cache-aware hardware-work or billing measurement.

## Score meaning and failure handling

Verified one-token surrogate labels are constructed with the loaded GGUF’s own
tokenizer. Chat rendering also uses the loaded server’s template, with thinking
disabled. For raw candidate logprobs `l_i`, reported relative probability is
`exp(l_i / T) / sum_j(exp(l_j / T))`. T is 1 for SystemOne.

The patched backend returns requested candidate scores from its full-vocabulary
softmax. With an unpatched compatible backend, Bev first requests top-N scores
and retries with the full vocabulary if any candidate is missing. It never fills
in an absent score. A small float32 accumulation tolerance applies to raw
candidate mass; the raw mass is returned unchanged.

| HTTP status | Meaning |
|---|---|
| 422 | Invalid request, unsupported semantics, wrong model alias, or context overflow |
| 502 | Invalid/missing backend scores or another backend error; no decision substituted |
| 503 | Service initialization or native health unavailable |
| 504 | Native scoring timeout; no decision substituted |

All field prompts are checked for context overflow before model scoring starts.
If a field fails, sibling HTTP tasks are canceled and drained before the request
fails. A valid finite output is a structural guarantee; it can still be the wrong
decision. Relative scores and model-format compliance do not establish Jevfire
parity, Jev parity, or correctness on a new workload.
