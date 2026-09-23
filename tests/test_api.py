import httpx
import pytest
from pydantic import ValidationError

from bev.app import create_app
from bev.models import DecisionRequest, SystemOneQuestion, SystemOneRequest
from test_scoring import completion, make_engine, preparation_response


BODY = {
    "context": "The lamp is on.",
    "schema": {"lamp": {"type": "boolean", "description": "Is it on?"}},
}


@pytest.mark.parametrize("change", [
    {"context": ""},
    {"unexpected": True},
    {"score_temperature": float("nan")},
    {"schema": {"x": {"type": "boolean", "description": "?", "choices": ["a", "b"]}}},
    {"schema": {"x": {"type": "enum", "description": "?", "choices": ["a", "a"]}}},
    {"schema": {"x": {"type": "enum", "description": "?", "choices": ["a"]}}},
    {"schema": {"x": {"type": "number", "description": "?"}}},
])
def test_strict_jevfire_contract(change):
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({**BODY, **change})


def test_systemone_has_no_jevfire_64_field_or_description_cap():
    question = {"type": "choice", "instructions": "long " * 1000, "criteria": {"a": "a", "b": "b"}}
    request = SystemOneRequest.model_validate({"state": "", "questions": {str(i): question for i in range(65)}})
    assert len(request.questions) == 65


@pytest.mark.parametrize("question", [
    {"type": "choice", "instructions": "?"},
    {"type": "choice", "instructions": "?", "criteria": ["a", "b"]},
    {"type": "noul", "instructions": "?", "criteria": {"yes": "yes", "no": "no"}},
    {"type": "score", "instructions": "?", "criteria": {"0": "low", "1": "high"}},
    {"type": "score", "instructions": "?", "criteria": ["only one"]},
    {"type": "score", "instructions": "?", "criteria": ["valid", " "]},
    {"type": "score", "instructions": "?", "criteria": ["valid", 5]},
    {"type": "score", "instructions": "?", "criteria": ["level"] * 256},
])
def test_systemone_rejects_malformed_primitive_contracts(question):
    with pytest.raises(ValidationError):
        SystemOneQuestion.model_validate(question)


async def test_api_no_decision_substituted_when_backend_loses_scores():
    def handler(request):
        if prepared := preparation_response(request):
            return prepared
        return httpx.Response(200, json=completion([]))

    engine = make_engine(handler)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine)), base_url="http://bev") as client:
            response = await client.post("/v1/decisions", json=BODY)
    finally:
        await engine.client.aclose()
    assert response.status_code == 502
    assert "no decision was substituted" in response.json()["detail"]
    assert "parsed_json" not in response.json()


async def test_api_health_reports_capabilities_and_backend_failure():
    engine = make_engine(lambda _: httpx.Response(503), selected=True)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine)), base_url="http://bev") as client:
            response = await client.get("/health")
    finally:
        await engine.client.aclose()
    assert response.status_code == 503


async def test_health_source_provenance_is_frozen_at_startup(monkeypatch):
    engine = make_engine(lambda _: httpx.Response(200, json={"status": "ok"}))
    app = create_app(engine)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bev") as client:
            first = (await client.get("/health")).json()
            # A disk change or a blocked read must not alter a running process's
            # provenance. Simulate it without editing real package source files.
            def no_read(*args, **kwargs):
                raise AssertionError("health must not reread source files")
            monkeypatch.setattr("pathlib.Path.read_bytes", no_read)
            second = (await client.get("/health")).json()
    finally:
        await engine.client.aclose()
    assert first["supported_primitives"] == ["choice", "noul", "score"]
    provenance = first["service_provenance"]
    assert provenance["version"] == "0.1.2"
    assert set(provenance["startup_source_sha256"]) == {"__init__.py", "app.py", "core.py", "models.py"}
    assert all(len(v) == 64 for v in provenance["startup_source_sha256"].values())
    assert second["service_provenance"] == provenance


async def test_api_rejects_unsupported_strategy_with_422():
    engine = make_engine(lambda _: httpx.Response(500))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine)), base_url="http://bev") as client:
            response = await client.post("/v1/decisions", json={**BODY, "strategy": "aligned_prefill"})
    finally:
        await engine.client.aclose()
    assert response.status_code == 422
