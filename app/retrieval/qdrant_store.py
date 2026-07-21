"""Qdrant collection management and hybrid upsert/search."""

from __future__ import annotations

import logging
import uuid
from typing import Any, cast

from app.config import Settings, get_settings
from app.ingestion.chunking import DocumentChunk
from app.retrieval.embeddings import (
    DEFAULT_DENSE_DIM,
    OfflineHashEmbedder,
    embed_for_index,
    get_dense_embedder,
    sparse_bm25_vector,
)

logger = logging.getLogger(__name__)

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"

# Operational baselines — never allow recreate/delete via ensure_collection.
PROTECTED_COLLECTIONS: frozenset[str] = frozenset(
    {
        "onlybook_faq",
        "support_faq",
    }
)


class ProtectedCollectionError(ValueError):
    """Raised when a write would destroy or recreate a protected collection."""


def assert_collection_write_allowed(
    collection: str,
    *,
    recreate: bool = False,
) -> None:
    """
    Block destructive ops on protected operational collections.

    Stage 2 (and later) may only recreate versioned collections such as
    ``onlybook_faq_bge_m3_v1``. Hash baselines ``onlybook_faq`` / ``support_faq``
    must never be recreated or deleted through this path.
    """
    name = (collection or "").strip()
    if not name:
        raise ProtectedCollectionError("collection name is required for write operations")
    if recreate and name in PROTECTED_COLLECTIONS:
        raise ProtectedCollectionError(
            f"refusing --recreate on protected collection {name!r}; "
            f"use a versioned collection (e.g. onlybook_faq_bge_m3_v1)"
        )


def get_qdrant_client(settings: Settings | None = None):
    from qdrant_client import QdrantClient

    cfg = settings or get_settings()
    kwargs: dict[str, Any] = {"url": cfg.qdrant_url}
    if cfg.qdrant_api_key:
        kwargs["api_key"] = cfg.qdrant_api_key
    return QdrantClient(**kwargs)


def qdrant_reachable(settings: Settings | None = None) -> bool:
    try:
        client = get_qdrant_client(settings)
        client.get_collections()
        return True
    except Exception:
        return False


def ensure_collection(
    collection: str,
    *,
    dense_dim: int = DEFAULT_DENSE_DIM,
    recreate: bool = False,
    settings: Settings | None = None,
) -> None:
    from qdrant_client.http import models as rest

    assert_collection_write_allowed(collection, recreate=recreate)

    client = get_qdrant_client(settings)
    names = {c.name for c in client.get_collections().collections}
    if recreate and collection in names:
        client.delete_collection(collection)
        names.discard(collection)

    if collection not in names:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_NAME: rest.VectorParams(size=dense_dim, distance=rest.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_NAME: rest.SparseVectorParams(
                    index=rest.SparseIndexParams(on_disk=False),
                ),
            },
        )
        for field in (
            "language",
            "category",
            "source",
            "doc_type",
            "faq_id",
            "intent",
        ):
            try:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Payload index %s: %s", field, exc)


def upsert_chunks(
    chunks: list[DocumentChunk],
    *,
    collection: str | None = None,
    recreate: bool = False,
    prefer_bge: bool = False,
    settings: Settings | None = None,
) -> int:
    """Embed (context+text) and upsert into Qdrant. Returns point count written."""
    from qdrant_client.http import models as rest

    cfg = settings or get_settings()
    coll = collection or cfg.qdrant_collection
    if not chunks:
        return 0

    assert_collection_write_allowed(coll, recreate=recreate)
    # Never write BGE (1024-d) vectors into hash baselines (384-d).
    if prefer_bge and coll in PROTECTED_COLLECTIONS:
        raise ProtectedCollectionError(
            f"refusing prefer_bge=True upsert into protected collection {coll!r}; "
            f"use a versioned BGE collection (e.g. onlybook_faq_bge_m3_v1)"
        )

    texts = [c.contextual_text or c.text for c in chunks]
    dense, sparse = embed_for_index(texts, settings=cfg, prefer_bge=prefer_bge)
    dim = len(dense[0]) if dense else DEFAULT_DENSE_DIM
    if prefer_bge and dense and dim != 1024:
        # Soft check: Stage 1 measured BGE-M3 at 1024; warn loudly if unexpected.
        logger.warning(
            "prefer_bge=True but dense dim=%s (expected 1024 for BGE-M3); "
            "verify embedder did not fall back to hash",
            dim,
        )
    ensure_collection(coll, dense_dim=dim, recreate=recreate, settings=cfg)

    client = get_qdrant_client(cfg)
    points = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        payload = {
            "text": chunk.text,
            "contextual_text": chunk.contextual_text or chunk.text,
            "language": meta.get("language"),
            "source": meta.get("source"),
            "title": meta.get("title"),
            "category": meta.get("category"),
            "chunk_id": chunk.id,
            # FAQ atomic metadata (optional for generic chunks)
            "doc_type": meta.get("doc_type"),
            "faq_id": meta.get("faq_id"),
            "intent": meta.get("intent"),
            "question": meta.get("question"),
            "answer": meta.get("answer"),
            "chunk_schema_version": meta.get("chunk_schema_version"),
            "faq_part_index": meta.get("faq_part_index"),
            "faq_part_count": meta.get("faq_part_count"),
        }
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))
        points.append(
            rest.PointStruct(
                id=pid,
                vector={
                    DENSE_NAME: dense[i],
                    SPARSE_NAME: rest.SparseVector(
                        indices=sparse[i]["indices"],
                        values=sparse[i]["values"],
                    ),
                },
                payload=payload,
            )
        )

    client.upsert(collection_name=coll, points=points)
    return len(points)


def hybrid_search(
    query: str,
    *,
    collection: str | None = None,
    top_n: int = 40,
    language: str | None = None,
    category: str | None = None,
    source: str | None = None,
    metadata_filters: dict[str, str] | None = None,
    prefer_bge: bool = False,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Hybrid dense + sparse search with optional exact-match metadata filters.

    Filter fields are restricted to allowlisted payload keys
    (``app.retrieval.metadata_filters.ALLOWED_METADATA_FILTER_KEYS``).
    Callers must not inject arbitrary Qdrant query objects.
    """
    from qdrant_client.http import models as rest

    from app.retrieval.metadata_filters import (
        ALLOWED_METADATA_FILTER_KEYS,
        merge_language_and_metadata_filters,
    )

    cfg = settings or get_settings()
    coll = collection or cfg.qdrant_collection
    if prefer_bge:
        dense_q = get_dense_embedder(prefer_bge=True).embed_query(query)
    else:
        dense_q = OfflineHashEmbedder().embed_query(query)
    sparse_q = sparse_bm25_vector(query)

    # Merge legacy kwargs into allowlisted exact-match map
    base: dict[str, str] = {}
    if language:
        base["language"] = language
    if category:
        base["category"] = category
    if source:
        base["source"] = source
    combined = merge_language_and_metadata_filters(
        auto_language=base.get("language"),
        metadata_filters={
            **{k: v for k, v in base.items() if k != "language"},
            **(metadata_filters or {}),
        }
        or None,
    )

    # FieldCondition is a valid Filter.must member. qdrant-client stubs type
    # ``must`` as a broad condition union; list is invariant, so cast at the
    # third-party boundary only (runtime value is list[FieldCondition]).
    must: list[rest.FieldCondition] = []
    if combined:
        for key, value in combined.items():
            if key not in ALLOWED_METADATA_FILTER_KEYS:
                # Defense in depth: never pass unknown keys into Qdrant filters
                logger.warning("Ignoring non-allowlisted filter key %r", key)
                continue
            must.append(
                rest.FieldCondition(key=key, match=rest.MatchValue(value=value))
            )
    query_filter = rest.Filter(must=cast(Any, must)) if must else None

    client = get_qdrant_client(cfg)
    # Prefer query_points API (qdrant-client >= 1.12)
    try:
        from qdrant_client.http.models import Fusion, FusionQuery, Prefetch

        results = client.query_points(
            collection_name=coll,
            prefetch=[
                Prefetch(
                    query=dense_q,
                    using=DENSE_NAME,
                    limit=top_n,
                    filter=query_filter,
                ),
                Prefetch(
                    query=rest.SparseVector(
                        indices=sparse_q["indices"],
                        values=sparse_q["values"],
                    ),
                    using=SPARSE_NAME,
                    limit=top_n,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_n,
            with_payload=True,
        )
        points = results.points
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hybrid query_points failed (%s); dense-only search", exc)
        hits = client.search(
            collection_name=coll,
            query_vector=(DENSE_NAME, dense_q),
            query_filter=query_filter,
            limit=top_n,
            with_payload=True,
        )
        points = hits

    out: list[dict[str, Any]] = []
    for p in points:
        payload = p.payload or {}
        out.append(
            {
                "id": str(p.id),
                "score": float(p.score or 0.0),
                "text": payload.get("text", ""),
                "metadata": {
                    "language": payload.get("language"),
                    "source": payload.get("source"),
                    "title": payload.get("title"),
                    "category": payload.get("category"),
                    "chunk_id": payload.get("chunk_id"),
                    # FAQ fields for serve-time display title (no re-index required)
                    "question": payload.get("question"),
                    "doc_type": payload.get("doc_type"),
                    "faq_id": payload.get("faq_id"),
                },
            }
        )
    return out
