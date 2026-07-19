"""Unit tests for dense embedder factory cache identity (no real BGE weights)."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.retrieval.embeddings import OfflineHashEmbedder, get_dense_embedder


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    get_settings.cache_clear()
    get_dense_embedder.cache_clear()
    yield
    get_dense_embedder.cache_clear()
    get_settings.cache_clear()


class _FakeBGE:
    """Stand-in for BGEEmbedder that never loads SentenceTransformer."""

    dim = 1024
    constructions = 0

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        type(self).constructions += 1
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dim


@pytest.fixture
def fake_bge(monkeypatch: pytest.MonkeyPatch) -> type[_FakeBGE]:
    _FakeBGE.constructions = 0
    monkeypatch.setattr("app.retrieval.embeddings.BGEEmbedder", _FakeBGE)
    # readiness imports BGEEmbedder at module scope for isinstance checks
    monkeypatch.setattr("app.api.readiness.BGEEmbedder", _FakeBGE)
    return _FakeBGE


def test_canonical_bge_key_reuses_single_instance(
    fake_bge: type[_FakeBGE],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    get_settings.cache_clear()
    get_dense_embedder.cache_clear()

    a = get_dense_embedder(
        prefer_bge=True,
        model_name="BAAI/bge-m3",
        require_bge=True,
    )
    b = get_dense_embedder(
        prefer_bge=True,
        model_name="BAAI/bge-m3",
        require_bge=False,
    )
    c = get_dense_embedder(prefer_bge=True)
    d = get_dense_embedder(True)
    e = get_dense_embedder(True, None)
    f = get_dense_embedder(True, "BAAI/bge-m3")

    assert fake_bge.constructions == 1
    assert a is b is c is d is e is f
    assert isinstance(a, _FakeBGE)
    assert a.model_name == "BAAI/bge-m3"


def test_startup_guard_readiness_request_share_one_instance(
    fake_bge: type[_FakeBGE],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate config guard → readiness snapshot → request hybrid_search path."""
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq_bge_m3_v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    get_settings.cache_clear()
    get_dense_embedder.cache_clear()

    from app.api.readiness import _validate_embedder_once
    from app.retrieval.config_guard import run_retrieval_startup_checks

    # Avoid real Qdrant in the guard after embedder validation.
    monkeypatch.setattr(
        "app.retrieval.qdrant_store.qdrant_reachable",
        lambda cfg: False,
    )

    cfg = get_settings()
    report = run_retrieval_startup_checks(cfg)
    assert report["status"] == "ok_bge_embedder_qdrant_unreachable"
    assert fake_bge.constructions == 1
    emb_guard = get_dense_embedder(
        prefer_bge=True,
        model_name=cfg.embedding_model,
        require_bge=True,
    )

    emb_check = _validate_embedder_once(cfg)
    assert emb_check["ok"] is True
    assert emb_check["type"] == "_FakeBGE"
    assert fake_bge.constructions == 1

    # Request path: hybrid_search uses get_dense_embedder(prefer_bge=True)
    emb_req = get_dense_embedder(prefer_bge=True)
    assert emb_req is emb_guard
    assert fake_bge.constructions == 1


def test_hash_behavior_and_require_bge(
    fake_bge: type[_FakeBGE],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    get_settings.cache_clear()
    get_dense_embedder.cache_clear()

    h1 = get_dense_embedder(prefer_bge=False)
    h2 = get_dense_embedder(prefer_bge=False)
    assert isinstance(h1, OfflineHashEmbedder)
    assert h1 is h2
    assert fake_bge.constructions == 0

    bge = get_dense_embedder(prefer_bge=True)
    assert isinstance(bge, _FakeBGE)
    assert bge is not h1
    assert fake_bge.constructions == 1

    with pytest.raises(RuntimeError, match="require_bge=True but prefer_bge=False"):
        get_dense_embedder(prefer_bge=False, require_bge=True)


def test_different_explicit_model_names_use_distinct_entries(
    fake_bge: type[_FakeBGE],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    get_settings.cache_clear()
    get_dense_embedder.cache_clear()

    a = get_dense_embedder(prefer_bge=True, model_name="BAAI/bge-m3")
    b = get_dense_embedder(prefer_bge=True, model_name="BAAI/other-bge")
    assert fake_bge.constructions == 2
    assert a is not b
    assert a.model_name == "BAAI/bge-m3"
    assert b.model_name == "BAAI/other-bge"


def test_require_bge_failure_does_not_return_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BoomBGE:
        def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
            raise RuntimeError("weights missing")

    monkeypatch.setattr("app.retrieval.embeddings.BGEEmbedder", _BoomBGE)
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    get_settings.cache_clear()
    get_dense_embedder.cache_clear()

    with pytest.raises(RuntimeError, match="BGE embedder required"):
        get_dense_embedder(prefer_bge=True, require_bge=True)

    # Without require_bge, fallback to hash (not silent BGE success).
    emb = get_dense_embedder(prefer_bge=True, require_bge=False)
    assert isinstance(emb, OfflineHashEmbedder)


def test_cache_clear_delegates_to_private_cache(
    fake_bge: type[_FakeBGE],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    get_settings.cache_clear()
    get_dense_embedder.cache_clear()

    a = get_dense_embedder(prefer_bge=True)
    assert fake_bge.constructions == 1
    get_dense_embedder.cache_clear()
    b = get_dense_embedder(prefer_bge=True)
    assert fake_bge.constructions == 2
    assert a is not b


def test_no_production_startup_cache_clear_on_bge_validate(
    fake_bge: type[_FakeBGE],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_validate_embedder_once must not clear the embedder cache."""
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq_bge_m3_v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    get_settings.cache_clear()
    get_dense_embedder.cache_clear()

    cleared = {"n": 0}
    real_clear = get_dense_embedder.cache_clear

    def counting_clear() -> None:
        cleared["n"] += 1
        real_clear()

    monkeypatch.setattr(
        "app.api.readiness.get_dense_embedder.cache_clear", counting_clear
    )
    # Pre-warm one instance
    get_dense_embedder(prefer_bge=True)
    assert fake_bge.constructions == 1

    from app.api.readiness import _validate_embedder_once

    _validate_embedder_once(get_settings())
    assert cleared["n"] == 0
    assert fake_bge.constructions == 1
