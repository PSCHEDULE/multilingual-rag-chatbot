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
        self.dim = int(self._model.get_sentence_embedding_dimension())
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
    """
    cfg = get_settings()
    name = model_name or cfg.embedding_model
    if prefer_bge:
        try:
            return BGEEmbedder(name)
        except Exception as exc:  # noqa: BLE001
            if require_bge:
                raise RuntimeError(
                    f"BGE embedder required but failed to load ({name}): {exc}"
                ) from exc
            logger.warning("BGE embedder unavailable (%s); using OfflineHashEmbedder", exc)
    elif require_bge:
        raise RuntimeError("require_bge=True but prefer_bge=False")
    return OfflineHashEmbedder()


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
