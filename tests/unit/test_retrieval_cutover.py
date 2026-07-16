"""Stage 3: config-driven BGE cutover wiring and guards."""

from __future__ import annotations

from typing import Any

import pytest

from app.config import get_settings
from app.retrieval.config_guard import (
    RetrievalConfigError,
    looks_like_bge_collection,
    validate_prefer_bge_collection_pair,
)


@pytest.fixture(autouse=True)
def _clear_settings() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_prefer_bge_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PREFER_BGE", raising=False)
    monkeypatch.delenv("RETRIEVAL_LANGUAGE_FILTER", raising=False)
    get_settings.cache_clear()
    s = get_settings()
    assert s.prefer_bge is False
    assert s.retrieval_language_filter is True


def test_settings_prefer_bge_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq_bge_m3_v1")
    monkeypatch.setenv("RETRIEVAL_LANGUAGE_FILTER", "false")
    get_settings.cache_clear()
    s = get_settings()
    assert s.prefer_bge is True
    assert s.qdrant_collection == "onlybook_faq_bge_m3_v1"
    assert s.retrieval_language_filter is False
    assert s.language_filter_value("ko") is None


def test_language_filter_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    monkeypatch.setenv("RETRIEVAL_LANGUAGE_FILTER", "true")
    get_settings.cache_clear()
    s = get_settings()
    assert s.language_filter_value("en") == "en"
    assert s.language_filter_value(None) is None

    monkeypatch.setenv("RETRIEVAL_LANGUAGE_FILTER", "false")
    get_settings.cache_clear()
    s2 = get_settings()
    assert s2.language_filter_value("en") is None


def test_validate_pair_blocks_bge_on_hash() -> None:
    with pytest.raises(RetrievalConfigError):
        validate_prefer_bge_collection_pair(
            collection="onlybook_faq", prefer_bge=True
        )
    with pytest.raises(RetrievalConfigError):
        validate_prefer_bge_collection_pair(
            collection="support_faq", prefer_bge=True
        )


def test_validate_pair_blocks_hash_on_bge_collection() -> None:
    with pytest.raises(RetrievalConfigError):
        validate_prefer_bge_collection_pair(
            collection="onlybook_faq_bge_m3_v1", prefer_bge=False
        )


def test_validate_pair_allows_cutover_and_rollback() -> None:
    validate_prefer_bge_collection_pair(
        collection="onlybook_faq_bge_m3_v1", prefer_bge=True
    )
    validate_prefer_bge_collection_pair(
        collection="onlybook_faq", prefer_bge=False
    )
    assert looks_like_bge_collection("onlybook_faq_bge_m3_v1")


def test_simple_retrieve_passes_settings_to_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq_bge_m3_v1")
    monkeypatch.setenv("RETRIEVAL_LANGUAGE_FILTER", "true")
    get_settings.cache_clear()

    captured: dict[str, Any] = {}

    class FakeHit:
        text = "hit"
        score = 1.0
        metadata = {"title": "t", "source": "s", "language": "ko"}

    class FakeResult:
        hits = [FakeHit()]

    def fake_retrieve(query: str, **kwargs: Any) -> FakeResult:
        captured["query"] = query
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr("app.graph.nodes.retrieve_and_rerank", fake_retrieve)

    from app.graph.nodes import simple_retrieve

    out = simple_retrieve(
        {
            "messages": [{"role": "user", "content": "환불?"}],
            "language": "ko",
        }
    )
    assert out["documents"]
    assert captured["prefer_bge"] is True
    assert captured["collection"] == "onlybook_faq_bge_m3_v1"
    assert captured["language"] == "ko"
    assert captured["query"] == "환불?"


def test_simple_retrieve_rollback_hash_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback: PREFER_BGE=false + onlybook_faq still wires correctly."""
    monkeypatch.setenv("PREFER_BGE", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    monkeypatch.setenv("RETRIEVAL_LANGUAGE_FILTER", "false")
    get_settings.cache_clear()

    captured: dict[str, Any] = {}

    class FakeResult:
        hits: list[Any] = []

    def fake_retrieve(query: str, **kwargs: Any) -> FakeResult:
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr("app.graph.nodes.retrieve_and_rerank", fake_retrieve)

    from app.graph.nodes import simple_retrieve

    simple_retrieve(
        {
            "messages": [{"role": "user", "content": "refund?"}],
            "language": "en",
        }
    )
    assert captured["prefer_bge"] is False
    assert captured["collection"] == "onlybook_faq"
    assert captured["language"] is None  # filter disabled


def test_retrieve_and_rerank_logs_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq_bge_m3_v1")
    get_settings.cache_clear()
    cfg = get_settings()

    monkeypatch.setattr(
        "app.retrieval.hybrid.hybrid_search",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "app.retrieval.hybrid.rerank_hits",
        lambda *a, **k: ([], 0.0),
    )

    from app.retrieval.hybrid import retrieve_and_rerank

    result = retrieve_and_rerank(
        "test",
        language="en",
        prefer_bge=True,
        collection="onlybook_faq_bge_m3_v1",
        settings=cfg,
    )
    assert len(result.hits) == 0

    with pytest.raises(RetrievalConfigError):
        retrieve_and_rerank(
            "test",
            prefer_bge=True,
            collection="onlybook_faq",
            settings=cfg,
        )


def test_startup_hash_mode_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    get_settings.cache_clear()
    from app.retrieval.config_guard import run_retrieval_startup_checks

    report = run_retrieval_startup_checks(get_settings())
    assert report["status"] == "ok_hash_mode"


def test_startup_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    get_settings.cache_clear()
    from app.retrieval.config_guard import run_retrieval_startup_checks

    with pytest.raises(RetrievalConfigError):
        run_retrieval_startup_checks(get_settings())
