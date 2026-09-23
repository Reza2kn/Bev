#!/usr/bin/env python3
"""Verify Bev's selected raw logprobs against the full vocabulary on its server.

Run on Stallion, where inference and the large reference response stay local.
This uses three one-token inference calls plus input-validation requests.
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--output", default="reports/runtime-probe.json")
    parser.add_argument("--tolerance", type=float, default=3e-4)
    parser.add_argument("--mass-tolerance", type=float, default=1e-3,
                        help="Float32 softmax reduction can lose small tail terms across a large vocabulary")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)

    props = request("/props")
    assert props.get("bev_selected_token_logprobs") == 1, "Missing patched capability"
    n_vocab = request("/v1/models")["data"][0]["meta"]["n_vocab"]
    prompt = request("/apply-template", {
        "messages": [{"role": "user", "content": (
            "Select one letter. What is two plus two?\nA: four\nB: five\nC: six\n"
            "Reply with the letter only."
        )}], "chat_template_kwargs": {"enable_thinking": False},
        "add_generation_prompt": True,
    })["prompt"]
    tokenized = request("/tokenize", {
        "content": prompt, "add_special": False, "parse_special": True,
    })["tokens"]
    common = {"prompt": tokenized, "n_predict": 1, "temperature": 0,
              "seed": 1234, "cache_prompt": False, "post_sampling_probs": False,
              "repeat_penalty": 1.0, "return_tokens": True, "stream": False}
    started = time.monotonic()
    reference = request("/completion", {**common, "n_probs": n_vocab})
    all_tokens = reference["completion_probabilities"][0]["top_logprobs"]
    assert len(all_tokens) == n_vocab, (len(all_tokens), n_vocab)
    reference_by_id = {row["id"]: row["logprob"] for row in all_tokens}
    # Include an ID outside top-512 and request deliberately non-sorted order.
    ranks = list(range(min(254, n_vocab)))
    if n_vocab > 254:
        ranks.append(min(1024, n_vocab - 1))
    ranks.reverse()
    ids = list(dict.fromkeys(all_tokens[rank]["id"] for rank in ranks))
    selected = request("/completion", {**common, "selected_token_ids": ids})
    rows = selected["completion_probabilities"][0]["top_logprobs"]
    assert [row["id"] for row in rows] == ids, "Missing IDs or request order changed"
    errors = {row["id"]: abs(row["logprob"] - reference_by_id[row["id"]]) for row in rows}
    assert max(errors.values()) <= args.tolerance, errors
    assert selected["generation_settings"]["selected_token_ids"] == ids

    # Bias changes sampling, while the raw pre-sampling score must stay intact.
    biased = request("/completion", {
        **common, "selected_token_ids": ids,
        "logit_bias": [[ids[0], 100.0]], "temperature": 1.0,
        "top_k": 1, "min_p": 0.9,
    })
    biased_rows = biased["completion_probabilities"][0]["top_logprobs"]
    assert [row["id"] for row in biased_rows] == ids
    bias_errors = {row["id"]: abs(row["logprob"] - reference_by_id[row["id"]])
                   for row in biased_rows}
    assert max(bias_errors.values()) <= args.tolerance, bias_errors

    malformed = [([], False), ([ids[0], ids[0]], False), ([-1], False),
                 ([n_vocab], False), ([1.5], False), ([True], False), ([ids[0]], True)]
    rejected = 0
    for invalid, post in malformed:
        try:
            request("/completion", {**common, "selected_token_ids": invalid,
                                    "post_sampling_probs": post})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400, (invalid, exc.code)
            rejected += 1
        else:
            raise AssertionError(f"Accepted invalid selected_token_ids={invalid}, post={post}")

    vocabulary_mass = sum(math.exp(row["logprob"]) for row in all_tokens)
    selected_mass = sum(math.exp(row["logprob"]) for row in rows)
    assert abs(vocabulary_mass - 1.0) <= args.mass_tolerance, vocabulary_mass
    assert 0.0 <= selected_mass <= 1.0 + args.mass_tolerance, selected_mass
    report = {
        "passed": True, "base_url": base, "build_info": props.get("build_info"),
        "bev_selected_token_logprobs": props.get("bev_selected_token_logprobs"),
        "model_path": props.get("model_path"), "n_vocab": n_vocab,
        "prompt_tokens": len(tokenized), "selected_count": len(ids), "selected_token_ids": ids,
        "selected_logprobs": rows, "full_vocabulary_mass": vocabulary_mass,
        "full_vocabulary_mass_error": abs(vocabulary_mass - 1.0), "mass_tolerance": args.mass_tolerance,
        "mass_tolerance_reason": "Native float32 softmax accumulates small positive terms across a large vocabulary; its serialized probabilities need not sum to exactly one in double precision.",
        "selected_raw_mass": selected_mass, "max_absolute_logprob_error": max(errors.values()),
        "max_sampler_bias_logprob_error": max(bias_errors.values()),
        "tolerance": args.tolerance, "invalid_requests_rejected": rejected,
        "inference_calls": 3, "elapsed_seconds": time.monotonic() - started,
        "timings": {"reference": reference.get("timings"), "selected": selected.get("timings"),
                    "biased_selected": biased.get("timings")},
        "scope": "Same pinned runtime; full float32 softmax with different summation order. No vLLM or FP16 parity claim.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
