"""Dense (+ sparse) embeddings for hybrid retrieval. Production: BGE-M3."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_DENSE_DIM = 384
_TOKEN = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]+", re.UNICODE)


class DenseEmbedder:
    """Protocol-like base."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class OfflineHashEmbedder(DenseEmbedder):
    """Deterministic dense vectors for offline gates (not production quality)."""

    def __init__(self, dim: int = DEFAULT_DENSE_DIM) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _TOKEN.findall(text.lower()) or ["_empty_"]
        for t in toks:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
            vec[(h // self.dim) % self.dim] += 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class BGEEmbedder(DenseEmbedder):
    """Sentence-transformers BGE-M3 wrapper (dense vectors; optional ``bge`` dep group)."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        logger.info("Loading SentenceTransformer model %s", model_name)
        self._model = SentenceTransformer(model_name)
        # Measure at load time — do not hardcode dim (BGE-M3 is typically 1024).
        measured = self._model.get_sentence_embedding_dimension()
        if measured is None:
            raise RuntimeError(
                f"SentenceTransformer {model_name!r} did not report an embedding dimension"
            )
        self.dim = int(measured)
        logger.info("Loaded dense embedder %s dim=%s", model_name, self.dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embs]

    def embed_query(self, text: str) -> list[float]:
        emb = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return emb.tolist()


def sparse_bm25_vector(text: str, *, dim_mod: int = 30_000) -> dict[str, list[Any]]:
    """
    Very small sparse encoder: term frequency hashed into indices.

    Returns Qdrant-compatible dict with ``indices`` and ``values``.
    """
    toks = [t.lower() for t in _TOKEN.findall(text)]
    if not toks:
        return {"indices": [0], "values": [1.0]}
    tf: dict[int, float] = {}
    for t in toks:
        idx = int(hashlib.md5(t.encode()).hexdigest(), 16) % dim_mod
        tf[idx] = tf.get(idx, 0.0) + 1.0
    # log tf
    indices = sorted(tf.keys())
    values = [1.0 + math.log(tf[i]) for i in indices]
    return {"indices": indices, "values": values}


@lru_cache
def _get_dense_embedder_cached(
    prefer_bge: bool,
    resolved_model_name: str,
) -> DenseEmbedder:
    """
    Construct and cache a dense embedder by canonical identity.

    Cache key is only ``(prefer_bge, resolved_model_name)``. Callers must resolve
    ``model_name=None`` to the configured model before invoking this function.
    ``require_bge`` is intentionally not part of the key (validation semantics only).
    """
    if prefer_bge:
        return BGEEmbedder(resolved_model_name)
    return OfflineHashEmbedder()


def get_dense_embedder(
    prefer_bge: bool = True,
    model_name: str | None = None,
    *,
    require_bge: bool = False,
) -> DenseEmbedder:
    """
    Return a dense embedder.

    When ``prefer_bge`` is True, attempts BGE-M3 (or ``model_name`` / settings).
    On failure, falls back to OfflineHashEmbedder unless ``require_bge`` is True
    (used by Stage 1 smoke tests so fallback cannot look successful).

    Semantically equivalent calls share one cached instance. In particular::

        get_dense_embedder(prefer_bge=True, model_name="BAAI/bge-m3", require_bge=True)
        get_dense_embedder(prefer_bge=True)

    resolve to the same cache key when the configured embedding model is
    ``BAAI/bge-m3``.
    """
    if require_bge and not prefer_bge:
        raise RuntimeError("require_bge=True but prefer_bge=False")

    cfg = get_settings()
    # Resolve before cache lookup so model_name=None and explicit default share a key.
    name = model_name or cfg.embedding_model

    if not prefer_bge:
        return _get_dense_embedder_cached(False, name)

    try:
        emb = _get_dense_embedder_cached(True, name)
    except Exception as exc:  # noqa: BLE001
        if require_bge:
            raise RuntimeError(
                f"BGE embedder required but failed to load ({name}): {exc}"
            ) from exc
        logger.warning("BGE embedder unavailable (%s); using OfflineHashEmbedder", exc)
        return _get_dense_embedder_cached(False, name)

    if require_bge and not isinstance(emb, BGEEmbedder):
        raise RuntimeError(
            f"BGE embedder required but failed to load ({name}): "
            f"got {type(emb).__name__}"
        )
    return emb


# Tests and diagnostics may reset the model cache between cases.
get_dense_embedder.cache_clear = _get_dense_embedder_cached.cache_clear  # type: ignore[attr-defined]
get_dense_embedder.cache_info = _get_dense_embedder_cached.cache_info  # type: ignore[attr-defined]


def embed_for_index(
    texts: list[str],
    *,
    settings: Settings | None = None,
    prefer_bge: bool = False,
) -> tuple[list[list[float]], list[dict[str, list[Any]]]]:
    """Return dense vectors and sparse BM25-like vectors for each text."""
    _ = settings or get_settings()
    dense_model = get_dense_embedder(prefer_bge=prefer_bge)
    # Clear cache confusion: Offline when prefer_bge False
    if not prefer_bge:
        dense_model = OfflineHashEmbedder()
    dense = dense_model.embed_documents(texts)
    sparse = [sparse_bm25_vector(t) for t in texts]
    return dense, sparse
