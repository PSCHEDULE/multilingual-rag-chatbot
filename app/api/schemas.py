"""Shared API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.retrieval.metadata_filters import (
    ALLOWED_METADATA_FILTER_KEYS,
    normalize_metadata_filters,
)


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
    metadata_filters: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional exact-match payload filters. Allowed keys: "
            + ", ".join(sorted(ALLOWED_METADATA_FILTER_KEYS))
        ),
    )

    @field_validator("metadata_filters", mode="before")
    @classmethod
    def _validate_metadata_filters(cls, value: Any) -> dict[str, str] | None:
        try:
            return normalize_metadata_filters(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc


class ChatSourceItem(BaseModel):
    title: str | None = None
    score: float | None = None
    source: str | None = None
