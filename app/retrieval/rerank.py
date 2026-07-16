"""BGE-reranker-v2-m3 with offline lexical fallback."""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
_TOKEN = re.compile(r"[\w\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]+", re.UNICODE)


@dataclass
class RankedHit:
    text: str
    score: float
    metadata: dict[str, Any]
    id: str | None = None


class BaseReranker:
    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """Return list of (original_index, score) sorted desc."""
        raise NotImplementedError


class LexicalReranker(BaseReranker):
    """Offline fallback: token overlap + length-normalized score."""

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        q = set(t.lower() for t in _TOKEN.findall(query))
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            toks = [t.lower() for t in _TOKEN.findall(doc)]
            if not toks:
                scored.append((i, 0.0))
                continue
            d = set(toks)
            overlap = len(q & d)
            score = overlap / (math.sqrt(len(q) or 1) * math.sqrt(len(d) or 1))
            scored.append((i, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class BGEReranker(BaseReranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        pairs = [[query, d] for d in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        return [(i, float(s)) for i, s in ranked[:top_k]]


@lru_cache
def get_reranker(prefer_bge: bool = True) -> BaseReranker:
    cfg = get_settings()
    if prefer_bge:
        try:
            logger.info("Loading reranker %s", cfg.reranker_model)
            return BGEReranker(cfg.reranker_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BGE reranker unavailable (%s); using LexicalReranker", exc)
    return LexicalReranker()


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int = 6,
    prefer_bge: bool = False,
) -> tuple[list[RankedHit], float]:
    """Rerank hybrid hits; returns (ranked, rerank_ms)."""
    if not hits:
        return [], 0.0
    docs = [h.get("text") or "" for h in hits]
    reranker = get_reranker(prefer_bge=prefer_bge) if prefer_bge else LexicalReranker()
    t0 = time.perf_counter()
    ranked_idx = reranker.rerank(query, docs, top_k=min(top_k, len(docs)))
    rerank_ms = (time.perf_counter() - t0) * 1000.0
    out: list[RankedHit] = []
    for idx, score in ranked_idx:
        h = hits[idx]
        out.append(
            RankedHit(
                id=h.get("id"),
                text=h.get("text") or "",
                score=score,
                metadata=dict(h.get("metadata") or {}),
            )
        )
    return out, rerank_ms
