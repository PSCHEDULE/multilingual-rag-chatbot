"""Hybrid retrieve + rerank facade with latency metrics."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.retrieval.qdrant_store import hybrid_search
from app.retrieval.rerank import RankedHit, rerank_hits

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    top_n: int = 0
    top_k: int = 0


@dataclass
class RetrievalResult:
    hits: list[RankedHit] = field(default_factory=list)
    metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, idx: int) -> RankedHit:
        return self.hits[idx]


def retrieve_and_rerank(
    query: str,
    *,
    language: str | None = None,
    category: str | None = None,
    source: str | None = None,
    top_n: int | None = None,
    top_k: int | None = None,
    collection: str | None = None,
    prefer_bge: bool | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    """
    Hybrid search top_n → rerank to top_k.

    When ``prefer_bge`` / ``collection`` are omitted, uses settings
    (``PREFER_BGE``, ``QDRANT_COLLECTION``). Logs collection and flags for cutover ops.
    """
    from app.retrieval.config_guard import validate_prefer_bge_collection_pair

    cfg = settings or get_settings()
    n = top_n if top_n is not None else cfg.retrieval_top_n
    k = top_k if top_k is not None else cfg.retrieval_top_k
    use_bge = cfg.prefer_bge if prefer_bge is None else prefer_bge
    coll = collection if collection is not None else cfg.qdrant_collection
    lang_filter = language  # caller decides; graph applies RETRIEVAL_LANGUAGE_FILTER

    try:
        validate_prefer_bge_collection_pair(collection=coll, prefer_bge=use_bge)
    except ValueError as exc:
        logger.error("Retrieval config mismatch: %s", exc)
        raise

    t0 = time.perf_counter()
    try:
        raw = hybrid_search(
            query,
            collection=coll,
            top_n=n,
            language=lang_filter,
            category=category,
            source=source,
            prefer_bge=use_bge,
            settings=cfg,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Hybrid search failed: %s", exc)
        raw = []
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    ranked, rerank_ms = rerank_hits(query, raw, top_k=k, prefer_bge=use_bge)
    total_ms = (time.perf_counter() - t0) * 1000.0
    metrics = RetrievalMetrics(
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        total_ms=total_ms,
        top_n=n,
        top_k=k,
    )
    logger.info(
        "retrieve_and_rerank collection=%s prefer_bge=%s language_filter=%s "
        "query_len=%s hits=%s retrieval_ms=%.1f rerank_ms=%.1f total_ms=%.1f",
        coll,
        use_bge,
        lang_filter,
        len(query),
        len(ranked),
        retrieval_ms,
        rerank_ms,
        total_ms,
    )
    return RetrievalResult(hits=ranked, metrics=metrics)
