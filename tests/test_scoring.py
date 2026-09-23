import asyncio
import json
import math

import httpx
import pytest

from bev.core import BackendError, DecisionEngine, FieldPlan, Label, decode_scores, gather_or_cancel
from bev.models import DecisionRequest, SystemOneRequest


LABELS = [Label("A", 10), Label("B", 20), Label("C", 30)]


def completion(scores=None, **updates):
    if scores is None:
        scores = [(10, math.log(0.6)), (20, math.log(0.2))]
    result = {
        "model": "bev-test",
        "truncated": False,
        # Deliberately disagree with score argmax: sampled text is never the vote.
        "content": "B",
        "tokens": [20],
        "completion_probabilities": [
            {"top_logprobs": [{"id": tid, "logprob": score} for tid, score in scores]}
        ],
        "timings": {"prompt_n": 3, "predicted_n": 1},
    }
    result.update(updates)
    return result


def make_engine(handler, *, max_model_len=100, selected=False):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://backend"
    )
    return DecisionEngine(
        client,
        "bev-test",
        max_model_len,
        LABELS,
        vocab_size=1000,
        slots=2,
        initial_n_probs=5,
        selected_token_logprobs=selected,
    )


def decision(**updates):
    body = {
        "context": "The lamp is on.",
        "schema": {"lamp": {"type": "boolean", "description": "Is the lamp on?"}},
    }
    body.update(updates)
    return DecisionRequest.model_validate(body)


def preparation_response(request):
    payload = json.loads(request.content)
    if request.url.path == "/apply-template":
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["add_generation_prompt"] is True
        return httpx.Response(200, json={"prompt": "rendered chat"})
    if request.url.path == "/tokenize":
        assert payload["add_special"] is False
        assert payload["parse_special"] is True
        return httpx.Response(200, json={"tokens": [101, 102, 103]})
    return None


def test_raw_candidate_softmax_and_abstention():
    result = decode_scores(
        [math.log(0.06), math.log(0.02)],
        FieldPlan("multi", "Pick", ["multi token option", "no"]),
        LABELS[:2],
        1.0,
        0.8,
    )
    assert result["value"] is None
    assert result["selected_value"] == "multi token option"
    assert result["probability"] == pytest.approx(0.75)
    assert result["candidate_probability_mass"] == pytest.approx(0.08)
    assert result["abstained"] is True


def test_temperature_changes_relative_scores_without_altering_mass():
    field = FieldPlan("x", "Pick", [True, False])
    result = decode_scores([math.log(0.09), math.log(0.01)], field, LABELS[:2], 2, None)
    assert result["probability"] == pytest.approx(0.75)
    assert result["candidate_probability_mass"] == pytest.approx(0.1)


def test_float32_mass_tolerance_retains_raw_mass():
    field = FieldPlan("x", "Pick", [True, False])
    result = decode_scores([math.log(0.9), math.log(0.10025)], field, LABELS[:2], 1, None)
    assert result["candidate_probability_mass"] == pytest.approx(1.00025)
    assert sum(c["probability"] for c in result["candidates"]) == pytest.approx(1)
    with pytest.raises(BackendError, match="full-vocabulary"):
        decode_scores([math.log(0.9), math.log(0.11)], field, LABELS[:2], 1, None)


async def test_failure_cancels_and_drains_other_fields():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def pending():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def failing():
        await entered.wait()
        raise BackendError("failed")

    with pytest.raises(BackendError, match="failed"):
        await gather_or_cancel(pending(), failing())
    assert cancelled.is_set()


async def test_missing_top_n_score_retries_complete_vocab_before_selecting():
    requests = []

    def handler(request):
        if prepared := preparation_response(request):
            return prepared
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["prompt"] == [101, 102, 103]
        assert payload["post_sampling_probs"] is False
        assert "grammar" not in payload
        scores = [(20, math.log(0.2))] if len(requests) == 1 else None
        return httpx.Response(200, json=completion(scores))

    engine = make_engine(handler)
    try:
        result = await engine.classify(decision())
    finally:
        await engine.client.aclose()
    assert [r["n_probs"] for r in requests] == [5, 1000]
    assert result["parsed_json"] == {"lamp": True}
    assert result["fields"]["lamp"]["probability"] == pytest.approx(0.75)
    assert result["fields"]["lamp"]["candidate_probability_mass"] == pytest.approx(0.8)
    assert result["backend_requests"] == 2
    assert result["usage"]["prompt_tokens"] == 6
    assert result["strategy"] == "batch"
    assert result["requested_strategy"] == "auto"


async def test_advertised_selected_scores_use_one_request_with_exact_ids():
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json=completion())

    engine = make_engine(handler, selected=True)
    try:
        result = await engine.score_field([101], FieldPlan("x", "?", [True, False]), 1, None)
    finally:
        await engine.client.aclose()
    assert seen[0]["selected_token_ids"] == [10, 20]
    assert len(seen) == 1
    assert result["score_retrieval"] == "selected_token_logprobs"


async def test_missing_full_vocab_candidate_fails_instead_of_default_choice():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion([(10, math.log(0.6))]))

    engine = make_engine(handler)
    try:
        with pytest.raises(BackendError, match="omitted candidate"):
            await engine.score_field([1], FieldPlan("x", "?", [True, False]), 1, None)
    finally:
        await engine.client.aclose()
    assert len(calls) == 2


@pytest.mark.parametrize(
    "broken",
    [
        {"truncated": True},
        {"truncated": None},
        {"model": "different-model"},
        {"completion_probabilities": []},
        {"completion_probabilities": [{"top_probs": [{"id": 10, "prob": 0.9}]}]},
        {"completion_probabilities": [{"top_logprobs": [
            {"id": 10, "logprob": -1}, {"id": 10, "logprob": -2},
        ]}]},
        {"completion_probabilities": [{"top_logprobs": [{"id": 10, "logprob": 0.4}]}]},
        {"completion_probabilities": [{"top_logprobs": [{"id": 10, "logprob": "-1"}]}]},
    ],
)
def test_rejects_unverified_scores(broken):
    engine = make_engine(lambda _: httpx.Response(500))
    with pytest.raises(BackendError):
        engine.candidate_scores(completion(**broken), LABELS[:2])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_requested_candidate_is_rejected(bad):
    engine = make_engine(lambda _: httpx.Response(500))
    with pytest.raises(BackendError):
        engine.candidate_scores(completion([(10, bad), (20, -2)]), LABELS[:2])


async def test_all_contexts_validated_before_any_inference():
    inference_calls = []

    def handler(request):
        if prepared := preparation_response(request):
            return prepared
        inference_calls.append(request)
        return httpx.Response(200, json=completion())

    engine = make_engine(handler, max_model_len=3)
    try:
        with pytest.raises(ValueError, match="context window"):
            await engine.classify(decision())
    finally:
        await engine.client.aclose()
    assert inference_calls == []


@pytest.mark.parametrize("updates", [{"strategy": "aligned_prefill"}, {"cache_salt": "tenant"}])
async def test_unsupported_contract_semantics_rejected_without_backend_calls(updates):
    calls = []
    engine = make_engine(lambda r: calls.append(r) or httpx.Response(500))
    try:
        with pytest.raises(ValueError, match="unsupported"):
            await engine.classify(decision(**updates))
    finally:
        await engine.client.aclose()
    assert calls == []


async def test_systemone_preserves_key_and_description_and_all_probabilities():
    rendered = []

    def handler(request):
        if request.url.path == "/apply-template":
            rendered.append(json.loads(request.content))
        if prepared := preparation_response(request):
            return prepared
        return httpx.Response(200, json=completion())

    engine = make_engine(handler)
    payload = SystemOneRequest.model_validate({
        "model": "bev-test", "state": {"lamp": "on"},
        "questions": {"Q": {"type": "choice", "instructions": {"task": "Choose"},
            "criteria": {"a distinct key": "The lamp is on", "other": "It is off"}}},
    })
    try:
        result = await engine.classify_systemone(payload)
    finally:
        await engine.client.aclose()
    answer = result["answers"]["Q"]
    assert answer["choice"] == "a distinct key"
    assert answer["probabilities"] == pytest.approx({"a distinct key": 0.75, "other": 0.25})
    assert answer["confidence"] == pytest.approx(0.75)
    prompt = rendered[0]["messages"][1]["content"]
    assert '"context": {"lamp": "on"}' in prompt
    assert '"value":"a distinct key","description":"The lamp is on"' in prompt


async def test_systemone_noul_returns_true_probability_when_false_wins():
    rendered = []

    def handler(request):
        if request.url.path == "/apply-template":
            rendered.append(json.loads(request.content))
        if prepared := preparation_response(request):
            return prepared
        return httpx.Response(200, json=completion([(10, math.log(0.1)), (20, math.log(0.4))]))

    engine = make_engine(handler)
    payload = SystemOneRequest.model_validate({
        "state": {"text": "The lamp is off."},
        "questions": {"yes_no": {"type": "noul", "instructions": "Is it on?"}},
    })
    try:
        result = await engine.classify_systemone(payload)
    finally:
        await engine.client.aclose()
    assert result["answers"]["yes_no"] == {"type": "noul", "noul": pytest.approx(0.2)}
    assert result["evaluation_trace"]["fields"]["yes_no"]["selected_value"] is False
    assert '"options":{"A":true,"B":false}' in rendered[0]["messages"][1]["content"]


async def test_systemone_score_returns_mean_and_exact_duplicate_rubric():
    def handler(request):
        if prepared := preparation_response(request):
            return prepared
        return httpx.Response(200, json=completion([
            (10, math.log(0.2)), (20, math.log(0.3)), (30, math.log(0.5)),
        ]))

    engine = make_engine(handler)
    # Repeated descriptions are legal: rubric indices remain separate outcomes.
    criteria = ["کم", "متوسط", "متوسط"]
    payload = SystemOneRequest.model_validate({
        "state": "Some evidence", "questions": {
            "rubric": {"type": "score", "instructions": "Evaluate this rubric", "criteria": criteria},
        },
    })
    try:
        result = await engine.classify_systemone(payload)
    finally:
        await engine.client.aclose()
    answer = result["answers"]["rubric"]
    assert answer["type"] == "score"
    assert answer["score"] == pytest.approx(1.3)
    assert answer["score"] != 2  # The rubric result is not the modal index.
    assert answer["probabilities"] == pytest.approx({"0": 0.2, "1": 0.3, "2": 0.5})
    assert answer["legend"] == {"0": "کم", "1": "متوسط", "2": "متوسط"}
    assert answer["confidence"] == pytest.approx(0.5)
    assert result["scores_are_calibrated"] is False
    assert result["usage"]["output_tokens"] == 1
    assert result["usage_accounting"] == "logical_prompt_and_scoring_tokens_including_retries"


async def test_systemone_model_mismatch_is_explicit():
    engine = make_engine(lambda _: httpx.Response(500))
    payload = SystemOneRequest.model_validate({
        "model": "a-different-model", "state": "",
        "questions": {"x": {"type": "choice", "instructions": "Choose", "criteria": {"a": "a", "b": "b"}}},
    })
    try:
        with pytest.raises(ValueError, match="not served"):
            await engine.classify_systemone(payload)
    finally:
        await engine.client.aclose()


async def test_labels_require_single_token_and_exact_detokenization():
    def handler(request):
        payload = json.loads(request.content)
        if request.url.path == "/tokenize":
            assert payload["parse_special"] is False
            text = payload["content"]
            return httpx.Response(200, json={"tokens": {"A": [10], "B": [20], "C": [30]}.get(text, [1, 2])})
        content = {10: "A", 20: "wrong", 30: "C"}[payload["tokens"][0]]
        return httpx.Response(200, json={"content": content})

    engine = make_engine(handler)
    try:
        labels = await engine.make_labels(2)
    finally:
        await engine.client.aclose()
    assert labels == [LABELS[0], LABELS[2]]
