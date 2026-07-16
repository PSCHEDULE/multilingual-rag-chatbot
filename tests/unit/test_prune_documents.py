"""Phase-1 post-rerank document pruning."""

from app.retrieval.prune import prune_reranked_documents


def _docs(*scores: float) -> list[dict]:
    return [
        {"text": f"doc{i}", "score": s, "title": f"T{i}", "source": f"s{i}"}
        for i, s in enumerate(scores)
    ]


def test_only_top1_when_others_near_zero() -> None:
    """Strong single match: low-scoring neighbors pruned."""
    docs = _docs(0.85, 0.0004, 0.0003, 0.0002, 0.0001, 0.00005)
    kept = prune_reranked_documents(
        docs, absolute_threshold=0.05, relative_threshold=0.05
    )
    assert len(kept) == 1
    assert kept[0]["score"] == 0.85
    assert kept[0]["text"] == "doc0"


def test_keep_top2_and_top3_when_scores_qualify() -> None:
    """Multiple relevant docs: Top-2/3 kept if abs + relative thresholds pass."""
    docs = _docs(1.0, 0.4, 0.2, 0.01, 0.001, 0.0001)
    kept = prune_reranked_documents(
        docs, absolute_threshold=0.05, relative_threshold=0.05
    )
    assert len(kept) == 3
    assert [d["score"] for d in kept] == [1.0, 0.4, 0.2]


def test_top2_fails_relative_threshold() -> None:
    """Top-2 above absolute floor but below 5% of top-1 is pruned."""
    # top1=1.0, relative floor=0.05; score 0.04 fails abs if abs=0.05
    docs = _docs(1.0, 0.04, 0.03)
    kept = prune_reranked_documents(
        docs, absolute_threshold=0.05, relative_threshold=0.05
    )
    assert len(kept) == 1

    # top1 very high; 0.06 passes abs but fails relative (need >= 0.1)
    docs2 = _docs(2.0, 0.06, 0.05)
    kept2 = prune_reranked_documents(
        docs2, absolute_threshold=0.05, relative_threshold=0.05
    )
    assert len(kept2) == 1
    assert kept2[0]["score"] == 2.0


def test_max_three_even_if_more_qualify() -> None:
    docs = _docs(1.0, 0.5, 0.4, 0.3, 0.2, 0.1)
    kept = prune_reranked_documents(
        docs, absolute_threshold=0.05, relative_threshold=0.05, max_keep=3
    )
    assert len(kept) == 3
    assert [d["score"] for d in kept] == [1.0, 0.5, 0.4]


def test_empty_input() -> None:
    assert prune_reranked_documents([]) == []


def test_simple_retrieve_applies_prune(monkeypatch) -> None:
    from app.config import get_settings
    from app.graph import nodes
    from app.retrieval.hybrid import RetrievalMetrics, RetrievalResult
    from app.retrieval.rerank import RankedHit

    get_settings.cache_clear()
    monkeypatch.setenv("PREFER_BGE", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    get_settings.cache_clear()

    hits = [
        RankedHit(text="a", score=0.9, metadata={"title": "A", "source": "a", "language": "en"}),
        RankedHit(text="b", score=0.001, metadata={"title": "B", "source": "b", "language": "en"}),
        RankedHit(text="c", score=0.0005, metadata={"title": "C", "source": "c", "language": "en"}),
        RankedHit(text="d", score=0.0001, metadata={"title": "D", "source": "d", "language": "en"}),
        RankedHit(text="e", score=0.0, metadata={"title": "E", "source": "e", "language": "en"}),
        RankedHit(text="f", score=0.0, metadata={"title": "F", "source": "f", "language": "en"}),
    ]

    def fake_retrieve(*args, **kwargs):
        return RetrievalResult(hits=hits, metrics=RetrievalMetrics())

    monkeypatch.setattr(nodes, "retrieve_and_rerank", fake_retrieve)
    out = nodes.simple_retrieve(
        {"messages": [{"role": "user", "content": "refund?"}], "language": "en"}
    )
    docs = out["documents"]
    assert len(docs) == 1
    assert docs[0]["text"] == "a"
