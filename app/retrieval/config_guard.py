"""Retrieval cutover guards: collection ↔ embedder consistency and startup checks."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import Settings, get_settings
from app.retrieval.qdrant_store import DENSE_NAME, PROTECTED_COLLECTIONS

logger = logging.getLogger(__name__)

# Versioned BGE collections follow onlybook_faq_bge_m3_vN (and similar).
_BGE_COLLECTION_RE = re.compile(r"bge[_-]?m3", re.IGNORECASE)

EXPECTED_BGE_DIM = 1024
EXPECTED_HASH_DIM = 384


class RetrievalConfigError(ValueError):
    """Inconsistent PREFER_BGE / QDRANT_COLLECTION (or failed BGE startup check)."""


def looks_like_bge_collection(name: str) -> bool:
    return bool(_BGE_COLLECTION_RE.search(name or ""))


def looks_like_hash_baseline(name: str) -> bool:
    return (name or "").strip() in PROTECTED_COLLECTIONS


def validate_prefer_bge_collection_pair(
    *,
    collection: str,
    prefer_bge: bool,
) -> None:
    """
    Refuse known-mismatched collection / embedder pairs.

    - prefer_bge=True must not target protected hash baselines (onlybook_faq, support_faq).
    - prefer_bge=False must not target versioned BGE collections (*bge_m3*).
    """
    coll = (collection or "").strip()
    if not coll:
        raise RetrievalConfigError("QDRANT_COLLECTION / collection name is empty")

    if prefer_bge and looks_like_hash_baseline(coll):
        raise RetrievalConfigError(
            f"PREFER_BGE=true is incompatible with protected hash collection {coll!r}. "
            f"Use onlybook_faq_bge_m3_v1 (or set PREFER_BGE=false for rollback)."
        )
    if not prefer_bge and looks_like_bge_collection(coll):
        raise RetrievalConfigError(
            f"PREFER_BGE=false is incompatible with BGE collection {coll!r}. "
            f"Set PREFER_BGE=true for cutover, or QDRANT_COLLECTION=onlybook_faq for rollback."
        )


def collection_dense_dim(client: Any, collection: str) -> int | None:
    """Return dense vector size for a named collection, or None if unknown."""
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        v = vectors.get(DENSE_NAME) or next(iter(vectors.values()), None)
        return int(getattr(v, "size", None)) if v is not None else None
    size = getattr(vectors, "size", None)
    return int(size) if size is not None else None


def run_retrieval_startup_checks(settings: Settings | None = None) -> dict[str, Any]:
    """
    Validate retrieval config at process start.

    When PREFER_BGE=true:
      - config pair consistent
      - BGE embedder loads with measured dim == 1024
      - if Qdrant reachable: active collection dense dim == 1024

    When PREFER_BGE=false: only config pair consistency (allows hash rollback without BGE).

    Raises RetrievalConfigError on hard failures.
    """
    cfg = settings or get_settings()
    report: dict[str, Any] = {
        "prefer_bge": cfg.prefer_bge,
        "collection": cfg.qdrant_collection,
        "retrieval_language_filter": cfg.retrieval_language_filter,
    }

    validate_prefer_bge_collection_pair(
        collection=cfg.qdrant_collection,
        prefer_bge=cfg.prefer_bge,
    )

    if not cfg.prefer_bge:
        report["status"] = "ok_hash_mode"
        logger.info(
            "Retrieval startup: hash mode collection=%s prefer_bge=false",
            cfg.qdrant_collection,
        )
        return report

    from app.retrieval.embeddings import BGEEmbedder, get_dense_embedder

    # Use the canonical factory without clearing: startup and request share one instance.
    emb = get_dense_embedder(
        prefer_bge=True,
        model_name=cfg.embedding_model,
        require_bge=True,
    )
    if not isinstance(emb, BGEEmbedder):
        raise RetrievalConfigError(
            f"PREFER_BGE=true but embedder is {type(emb).__name__}, not BGEEmbedder"
        )
    dim = int(emb.dim)
    report["embedder_dim"] = dim
    if dim != EXPECTED_BGE_DIM:
        raise RetrievalConfigError(
            f"BGE embedder dim={dim}, expected {EXPECTED_BGE_DIM} for collection cutover"
        )

    from app.retrieval.qdrant_store import get_qdrant_client, qdrant_reachable

    if not qdrant_reachable(cfg):
        logger.warning(
            "PREFER_BGE=true but Qdrant not reachable; skipped collection dim check "
            "(collection=%s)",
            cfg.qdrant_collection,
        )
        report["status"] = "ok_bge_embedder_qdrant_unreachable"
        return report

    client = get_qdrant_client(cfg)
    names = {c.name for c in client.get_collections().collections}
    if cfg.qdrant_collection not in names:
        raise RetrievalConfigError(
            f"PREFER_BGE=true but collection {cfg.qdrant_collection!r} does not exist on Qdrant"
        )
    coll_dim = collection_dense_dim(client, cfg.qdrant_collection)
    report["collection_dim"] = coll_dim
    if coll_dim != EXPECTED_BGE_DIM:
        raise RetrievalConfigError(
            f"collection {cfg.qdrant_collection!r} dense dim={coll_dim}, "
            f"expected {EXPECTED_BGE_DIM} when PREFER_BGE=true"
        )

    report["status"] = "ok_bge_mode"
    logger.info(
        "Retrieval startup OK: collection=%s prefer_bge=true embedder_dim=%s collection_dim=%s",
        cfg.qdrant_collection,
        dim,
        coll_dim,
    )
    return report
