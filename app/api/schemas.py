"""Shared API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness: process is up (always 200 if the app responds)."""

    status: str = Field(examples=["ok"])
    live: bool = True
    app: str | None = None
    llm_provider: str | None = None
    version: str | None = None


class ReadyCheckDetail(BaseModel):
    ok: bool
    detail: str | None = None

    model_config = {"extra": "allow"}


class ReadyResponse(BaseModel):
    """Readiness: dependencies OK to serve retrieval traffic."""

    status: str = Field(examples=["ready", "not_ready"])
    ready: bool
    prefer_bge: bool | None = None
    collection: str | None = None
    embedding_model: str | None = None
    expected_dense_dim: int | None = None
    checks: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    startup: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    language: str | None = Field(
        default=None,
        description="Optional explicit language: ko|en|ja|zh",
    )
    metadata_filters: dict[str, str] | None = None


class ChatSourceItem(BaseModel):
    title: str | None = None
    score: float | None = None
    source: str | None = None
