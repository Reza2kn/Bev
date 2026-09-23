"""Loopback HTTP API beside the Stallion Prism inference process."""

import hashlib
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException

from . import __version__
from .core import BackendError, DecisionEngine
from .models import DecisionRequest, SystemOneRequest


def create_app(engine: DecisionEngine | None = None) -> FastAPI:
    def capture_startup_provenance(application):
        application.state.startup_provenance = {
            "startup_source_sha256": {
                source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                for source in sorted(Path(__file__).parent.glob("*.py"))
            },
            "process_id": os.getpid(),
            "startup_utc": datetime.now(UTC).isoformat(),
            "version": __version__,
        }

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Snapshot once while initializing this process. Health never rereads
        # mutable source files and must not claim newly copied code is loaded.
        capture_startup_provenance(application)
        if engine is not None:
            application.state.engine = engine
            yield
            return
        headers = {}
        if os.environ.get("BEV_LLAMA_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["BEV_LLAMA_API_KEY"]
        async with httpx.AsyncClient(
            base_url=os.environ.get("BEV_LLAMA_URL", "http://127.0.0.1:18780"),
            timeout=httpx.Timeout(float(os.environ.get("BEV_HTTP_TIMEOUT", "300")), connect=5),
            headers=headers,
            trust_env=False,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        ) as client:
            application.state.engine = await DecisionEngine.connect(
                client,
                os.environ.get("BEV_MODEL", "bev-bonsai-27b"),
                initial_n_probs=int(os.environ.get("BEV_INITIAL_N_PROBS", "512")),
            )
            yield

    application = FastAPI(
        title="Bev",
        version=__version__,
        description=(
            "Independent boolean/enum decisions scored with a ternary Bonsai model through "
            "Prism llama.cpp. Probabilities are candidate-relative likelihoods and are not "
            "calibrated confidence. Deploy on loopback or behind an authenticated gateway."
        ),
        lifespan=lifespan,
    )
    if engine is not None:
        application.state.engine = engine
        capture_startup_provenance(application)

    def get_engine():
        available = getattr(application.state, "engine", None)
        if available is None:
            raise HTTPException(503, "Bev backend initialization is incomplete")
        return available

    async def execute(request):
        current = get_engine()
        try:
            if isinstance(request, SystemOneRequest):
                return await current.classify_systemone(request)
            return await current.classify(request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise HTTPException(504, "llama.cpp scoring timed out; no decision was substituted") from exc
        except (BackendError, httpx.HTTPError) as exc:
            raise HTTPException(502, "llama.cpp scoring failed; no decision was substituted") from exc

    @application.get("/health")
    async def health():
        current = get_engine()
        try:
            await current._json_response(await current.client.get("/health"))
        except (BackendError, httpx.HTTPError) as exc:
            raise HTTPException(503, "llama.cpp backend is unavailable") from exc
        return {
            "service_provenance": application.state.startup_provenance,
            "status": "ok",
            "model": current.model,
            "backend": "prism_llama_cpp",
            "backend_build": current.backend_build,
            "model_path": current.model_path,
            "max_model_len": current.max_model_len,
            "context_limit_scope": "per_slot_including_one_answer_token",
            "max_choices": len(current.labels),
            "max_decision_fields": 64,
            "systemone_field_limit": None,
            "supported_primitives": ["choice", "noul", "score"],
            "backend_slots": current.slots,
            "vocab_size": current.vocab_size,
            "selected_token_logprobs": current.selected_token_logprobs,
            "score_source": "pre_sampling_full_vocabulary_logprobs",
            "scores_are_calibrated": False,
        }

    @application.post("/v1/decisions")
    async def decisions(request: DecisionRequest):
        return await execute(request)

    @application.post("/v1/systemone")
    async def systemone(request: SystemOneRequest):
        return await execute(request)

    return application


app = create_app()
