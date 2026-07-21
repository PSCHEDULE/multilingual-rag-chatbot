"""Allowlisted metadata filters for hybrid search (exact payload match).

Only fields that are written into Qdrant payload (and typically indexed) are
accepted. Callers must never forward arbitrary Qdrant query objects.
"""

from __future__ import annotations

from typing import Any

# Payload keys stored at ingest (see app.retrieval.qdrant_store.upsert_chunks).
ALLOWED_METADATA_FILTER_KEYS: frozenset[str] = frozenset(
    {
        "language",
        "category",
        "source",
        "doc_type",
        "faq_id",
        "intent",
    }
)

MAX_METADATA_FILTER_KEYS = 8
MAX_METADATA_FILTER_VALUE_LEN = 128


def normalize_metadata_filters(
    raw: dict[str, Any] | None,
) -> dict[str, str] | None:
    """
    Validate and normalize a client metadata filter map.

    Raises ``ValueError`` with a controlled message on unsupported keys,
    empty values, or oversized input. Returns ``None`` when empty/absent.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("metadata_filters must be an object of string key/value pairs")
    if len(raw) > MAX_METADATA_FILTER_KEYS:
        raise ValueError(
            f"metadata_filters supports at most {MAX_METADATA_FILTER_KEYS} keys"
        )
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata_filters keys must be non-empty strings")
        k = key.strip()
        if k not in ALLOWED_METADATA_FILTER_KEYS:
            allowed = ", ".join(sorted(ALLOWED_METADATA_FILTER_KEYS))
            raise ValueError(
                f"unsupported metadata_filters key {k!r}; allowed: {allowed}"
            )
        if value is None:
            raise ValueError(f"metadata_filters[{k!r}] must be a non-empty string")
        if not isinstance(value, str):
            raise ValueError(f"metadata_filters[{k!r}] must be a string")
        v = value.strip()
        if not v:
            raise ValueError(f"metadata_filters[{k!r}] must be a non-empty string")
        if len(v) > MAX_METADATA_FILTER_VALUE_LEN:
            raise ValueError(
                f"metadata_filters[{k!r}] exceeds max length "
                f"{MAX_METADATA_FILTER_VALUE_LEN}"
            )
        out[k] = v
    return out or None


def merge_language_and_metadata_filters(
    *,
    auto_language: str | None,
    metadata_filters: dict[str, str] | None,
) -> dict[str, str] | None:
    """
    Combine auto language filter with request metadata filters.

    Explicit ``metadata_filters['language']`` overrides the automatic language
    filter when present. Other keys are exact-match payload filters.
    """
    combined: dict[str, str] = {}
    if auto_language:
        combined["language"] = str(auto_language).strip()
    if metadata_filters:
        for k, v in metadata_filters.items():
            if k in ALLOWED_METADATA_FILTER_KEYS and v:
                combined[k] = v
    return combined or None
