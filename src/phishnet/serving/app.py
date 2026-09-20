"""Phase 6 serving app: `/health`, `/predict`, `/explain`.

Replaces the legacy ``phishnet.api`` (hard-vote urlset ensemble, MongoDB
logging, `/report`). Tier 1 is the Phase 3 row (a) LightGBM
(:class:`phishnet.serving.tier1.Tier1Servable`). In-band rows may run Tier 2
through an injected provider; the disposition mapping is fail-closed
(:mod:`phishnet.serving.cascade`). Shortener URLs are resolved before
scoring (:mod:`phishnet.serving.shortener`); an unresolved shortener is
``can't assess`` with no verdict score.

The `/explain` endpoint returns native tree SHAP for the scoring model, so
the old 501 path retires. Thresholds are read from the service by the
client (``/health`` and every ``/predict``); the extension never hard-codes
them.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from phishnet.serving import shortener
from phishnet.serving.cascade import Decision, Tier2Outcome, decide
from phishnet.serving.shortener import Resolution
from phishnet.serving.tier1 import Tier1Servable
from phishnet.serving.tier2 import provider_from_env

Resolver = Callable[[str], Resolution]


class Tier2Provider(Protocol):
    """A Tier-2 provider judges one URL, structurally."""

    def judge(self, url: str) -> Tier2Outcome | None: ...


class PredictRequest(BaseModel):
    url: HttpUrl


class ExplainRequest(BaseModel):
    url: HttpUrl
    top_k: int = Field(default=10, ge=1)


def predict_one(
    url: str,
    *,
    tier1: Tier1Servable,
    resolver: Resolver | None,
    tier2: Tier2Provider | None,
    t_alert: float,
    lower_edge: float,
    tier2_floor: float | None = None,
) -> dict[str, Any]:
    """Core serving path for one URL (shared by the endpoint and tests).

    ``tier2_floor`` lowers the Tier-2 trigger below the registered band edge
    (testing/experimentation only; defaults to ``lower_edge``). It changes
    only *when* the LLM is asked, never the model or thresholds.
    """
    floor = lower_edge if tier2_floor is None else tier2_floor
    scored_url = url
    unresolved = False
    if resolver is not None and shortener.is_shortener(url):
        result = resolver(url)
        if not result.resolved:
            unresolved = True
        else:
            assert result.final_url is not None
            scored_url = result.final_url

    tier1_score: float | None = None
    outcome: Tier2Outcome | None = None
    if not unresolved:
        tier1_score = tier1.score_one(scored_url)
        # Tier 2 runs only for rows at/above the floor (registered band edge
        # by default): rows below it already have a disposition, and a live
        # provider call for them would be pure waste.
        if tier2 is not None and floor <= tier1_score < t_alert:
            outcome = tier2.judge(scored_url)

    decision: Decision = decide(
        tier1_score,
        outcome,
        t_alert=t_alert,
        lower_edge=floor,
        unresolved=unresolved,
        no_verdict_reason=(
            "tier2_no_verdict" if tier2 is not None else "tier2_not_configured"
        ),
    )
    payload: dict[str, Any] = {
        "url": url,
        "scored_url": scored_url,
        "disposition": decision.disposition,
        "score": decision.score,
        "reason": decision.reason,
        "in_band": decision.in_band,
        "tier1_score": tier1_score,
        "tier2_floor": floor,
        "tier2_mode": getattr(tier2, "mode", "configured")
        if tier2 is not None
        else "disabled",
        "tier2": None
        if outcome is None
        else {"kind": outcome.kind, "reason": outcome.reason},
        "model_hash": tier1.model_hash,
        "thresholds_source": tier1.thresholds_source,
    }
    return payload


def create_app(
    *,
    servable: Tier1Servable | None = None,
    tier2: Tier2Provider | None = None,
    resolver: Resolver | None = None,
    extension_id: str | None = None,
    tier2_floor: float | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        # Startup verification lives in Tier1Servable: model/column SHA256 vs
        # model_manifest.json and thresholds vs reports/phase4.json, refusing
        # on mismatch (no degraded mode).
        app.state.tier1 = servable if servable is not None else Tier1Servable()
        app.state.tier2 = tier2
        app.state.resolver = resolver if resolver is not None else shortener.resolve
        app.state.tier2_mode = (
            getattr(tier2, "mode", "configured") if tier2 is not None else "disabled"
        )
        # Tier-2 trigger floor: explicit arg > env > registered lower_edge.
        env_floor = os.getenv("PHISHNET_TIER2_FLOOR")
        if tier2_floor is not None:
            app.state.tier2_floor = float(tier2_floor)
        elif env_floor:
            app.state.tier2_floor = float(env_floor)
        else:
            app.state.tier2_floor = app.state.tier1.thresholds["lower_edge"]
        yield

    app = FastAPI(title="PhishNet serving (Phase 6)", lifespan=lifespan)

    # CORS: pinned extension ID + localhost only; no wildcard, no credentials.
    origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
    ext = extension_id or os.getenv("PHISHNET_EXTENSION_ID")
    if ext:
        origins.append(f"chrome-extension://{ext}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        tier1: Tier1Servable = request.app.state.tier1
        return {
            "status": "ok",
            "model_hash": tier1.model_hash,
            "columns_hash": tier1.columns_hash,
            "n_columns": len(tier1.columns),
            "thresholds": tier1.thresholds,
            "thresholds_source": tier1.thresholds_source,
            "tier2_mode": request.app.state.tier2_mode,
            "tier2_floor": request.app.state.tier2_floor,
        }

    @app.post("/predict")
    def predict_url(request: Request, body: PredictRequest) -> dict[str, Any]:
        tier1: Tier1Servable = request.app.state.tier1
        return predict_one(
            str(body.url),
            tier1=tier1,
            resolver=request.app.state.resolver,
            tier2=request.app.state.tier2,
            t_alert=tier1.thresholds["t_alert"],
            lower_edge=tier1.thresholds["lower_edge"],
            tier2_floor=request.app.state.tier2_floor,
        )

    @app.post("/explain")
    def explain_url(request: Request, body: ExplainRequest) -> dict[str, Any]:
        tier1: Tier1Servable = request.app.state.tier1
        try:
            attribution = tier1.explain_one(str(body.url), top_k=body.top_k)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {
            "url": str(body.url),
            "model_hash": tier1.model_hash,
            "attribution": attribution,
        }

    return app


app = create_app(tier2=provider_from_env())
