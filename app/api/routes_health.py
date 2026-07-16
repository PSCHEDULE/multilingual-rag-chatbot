"""Health (liveness) and readiness endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response

from app import __version__
from app.api.readiness import check_readiness
from app.api.schemas import HealthResponse, ReadyResponse
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness probe — process is running.

    Does **not** check Qdrant or embeddings (use ``GET /ready`` for that).
    Suitable for basic uptime / container process checks.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        live=True,
        app=settings.app_name,
        llm_provider=settings.llm_provider,
        version=__version__,
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request, response: Response) -> dict[str, Any]:
    """
    Readiness probe — safe to send traffic for retrieval-backed chat.

    Lightweight: reuses startup embedder validation; optional Qdrant metadata
    recheck. Does **not** reload BGE or clear embedder caches. Returns HTTP
    **503** when not ready so orchestrators can delay traffic.
    """
    settings = get_settings()
    startup = getattr(request.app.state, "startup_retrieval_report", None)
    report = check_readiness(settings, startup_report=startup)
    if not report.get("ready"):
        response.status_code = 503
    return report
