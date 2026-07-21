"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.config import get_settings

_WIDGET_DIR = Path(__file__).resolve().parents[1] / "widget"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: production policy + retrieval cutover config (hash rollback stays default-safe)."""
    settings = get_settings()
    # Production + mock / missing provider secret already rejected in Settings
    # model_validator; re-assert at registered FastAPI lifespan for startup binding.
    if settings.is_production() and settings.mock_llm:
        raise RuntimeError(
            "Production startup rejected: MOCK_LLM must be false when APP_ENV is production"
        )
    from app.api.readiness import build_startup_readiness_snapshot
    from app.retrieval.config_guard import (
        RetrievalConfigError,
        run_retrieval_startup_checks,
    )

    try:
        # Legacy guard (config pair + BGE load when enabled)
        report = run_retrieval_startup_checks(settings)
        # Full readiness snapshot for lightweight GET /ready (no model reload on probe)
        snapshot = build_startup_readiness_snapshot(settings)
        # Prefer richer snapshot; keep legacy status field for log compatibility
        if report.get("status") and not snapshot.get("status"):
            snapshot["status"] = report["status"]
        elif report.get("status") in (
            "ok_bge_mode",
            "ok_hash_mode",
            "ok_bge_embedder_qdrant_unreachable",
        ):
            snapshot["legacy_startup_status"] = report.get("status")
            if snapshot.get("ready") and report.get("status"):
                snapshot["status"] = report["status"]
        app.state.startup_retrieval_report = snapshot
        logger.info(
            "Retrieval startup checks: %s (ready_snapshot=%s)",
            report.get("status"),
            snapshot.get("status"),
        )
    except RetrievalConfigError:
        app.state.startup_retrieval_report = {
            "status": "failed",
            "ready": False,
            "prefer_bge": settings.prefer_bge,
            "collection": settings.qdrant_collection,
            "checks": {},
            "errors": ["startup retrieval checks failed"],
        }
        logger.exception(
            "Retrieval startup check failed (PREFER_BGE=%s collection=%s). "
            "Rollback: QDRANT_COLLECTION=onlybook_faq PREFER_BGE=false",
            settings.prefer_bge,
            settings.qdrant_collection,
        )
        raise
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Multilingual RAG customer support chatbot (KO/EN/JA/ZH)",
        lifespan=lifespan,
    )
    # Derive CORS from settings: wildcard origins force credentials=False.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list(),
        allow_credentials=settings.cors_allow_credentials(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    if _WIDGET_DIR.is_dir():
        app.mount("/widget", StaticFiles(directory=str(_WIDGET_DIR), html=True), name="widget")
    return app


app = create_app()


def run() -> None:
    """Console script entry: `multilingual-rag-chatbot`."""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
