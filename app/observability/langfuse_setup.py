"""Langfuse tracing setup — no-op when credentials missing."""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
_client: Any | None = None
_initialized = False


def get_langfuse(settings: Settings | None = None) -> Any | None:
    """Return Langfuse client or None when disabled."""
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    cfg = settings or get_settings()
    if not cfg.langfuse_public_key or not cfg.langfuse_secret_key:
        logger.info("Langfuse disabled (missing LANGFUSE_PUBLIC_KEY / SECRET_KEY)")
        _client = None
        return None
    try:
        from langfuse import Langfuse

        kwargs: dict[str, Any] = {
            "public_key": cfg.langfuse_public_key,
            "secret_key": cfg.langfuse_secret_key,
        }
        if cfg.langfuse_host:
            kwargs["host"] = cfg.langfuse_host
        _client = Langfuse(**kwargs)
        logger.info("Langfuse enabled host=%s", cfg.langfuse_host or "default")
        return _client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse init failed: %s", exc)
        _client = None
        return None


def reset_langfuse_for_tests() -> None:
    global _client, _initialized
    _client = None
    _initialized = False
