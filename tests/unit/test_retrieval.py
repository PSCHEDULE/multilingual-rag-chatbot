"""Unit tests for rerank / retrieval facade with mocks."""

from app.retrieval.hybrid import retrieve_and_rerank
from app.retrieval.rerank import LexicalReranker, rerank_hits


def test_lexical_rerank_orders_overlap() -> None:
    r = LexicalReranker()
    docs = ["unrelated weather report", "refund policy within 14 days", "shipping times"]
    ranked = r.rerank("refund policy", docs, top_k=2)
    assert ranked[0][0] == 1


def test_rerank_hits_empty() -> None:
    hits, ms = rerank_hits("q", [], top_k=5)
    assert hits == []
    assert ms == 0.0


def test_retrieve_and_rerank_handles_qdrant_down(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise ConnectionError("no qdrant")

    monkeypatch.setattr("app.retrieval.hybrid.hybrid_search", _boom)
    result = retrieve_and_rerank("refund", top_n=10, top_k=3, prefer_bge=False)
    assert len(result) == 0
    assert result.metrics.total_ms >= 0
