"""Liveness /health and readiness /ready endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient

from app.api.readiness import (
    build_startup_readiness_snapshot,
    check_readiness,
    expected_collection_dim,
)
from app.config import get_settings
from app.main import app
from app.retrieval.config_guard import EXPECTED_BGE_DIM, EXPECTED_HASH_DIM


async def test_health_returns_ok_liveness() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body.get("live") is True
    assert body.get("llm_provider") == "openai"


def test_default_llm_is_gpt4o_mini() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"


def test_expected_dim_by_mode() -> None:
    assert expected_collection_dim(True) == EXPECTED_BGE_DIM
    assert expected_collection_dim(False) == EXPECTED_HASH_DIM


def _fake_qdrant(monkeypatch, *, name: str, dim: int, reachable: bool = True) -> None:
    class FakeColl:
        def __init__(self, n: str) -> None:
            self.name = n

    class FakeParams:
        class V:
            def __init__(self, size: int) -> None:
                self.size = size

        def __init__(self, size: int) -> None:
            self.vectors = {"dense": self.V(size)}

    class FakeInfo:
        def __init__(self, size: int) -> None:
            self.config = type("C", (), {"params": FakeParams(size)})()
            self.points_count = 10

    class FakeClient:
        def get_collections(self):
            return type("R", (), {"collections": [FakeColl(name)]})()

        def get_collection(self, coll_name: str):
            assert coll_name == name
            return FakeInfo(dim)

    monkeypatch.setattr(
        "app.api.readiness.qdrant_reachable", lambda s=None: reachable
    )
    monkeypatch.setattr(
        "app.api.readiness.get_qdrant_client", lambda s=None: FakeClient()
    )


def test_check_readiness_config_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    get_settings.cache_clear()
    startup = {
        "status": "failed",
        "prefer_bge": True,
        "collection": "onlybook_faq",
        "checks": {"embedder": {"ok": False, "dim": 1024}},
    }
    report = check_readiness(get_settings(), startup_report=startup)
    assert report["ready"] is False
    assert report["status"] == "not_ready"
    assert report["checks"]["config_pair"]["ok"] is False


def test_check_readiness_hash_mode_mocked_qdrant(monkeypatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    get_settings.cache_clear()
    cfg = get_settings()
    _fake_qdrant(monkeypatch, name="onlybook_faq", dim=384)

    startup = {
        "status": "ok_hash_mode",
        "prefer_bge": False,
        "collection": "onlybook_faq",
        "checks": {
            "embedder": {
                "ok": True,
                "type": "OfflineHashEmbedder",
                "dim": 384,
                "expected_dim": 384,
            }
        },
    }
    report = check_readiness(cfg, startup_report=startup)
    assert report["ready"] is True
    assert report["status"] == "ready"
    assert report["checks"]["embedder"]["ok"] is True
    assert report["checks"]["embedder"]["source"] == "startup_snapshot"
    assert report["checks"]["collection_dim"]["dim"] == 384


def test_check_readiness_bge_dim_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq_bge_m3_v1")
    get_settings.cache_clear()
    cfg = get_settings()
    _fake_qdrant(monkeypatch, name="onlybook_faq_bge_m3_v1", dim=384)

    startup = {
        "status": "ok_bge_mode",
        "prefer_bge": True,
        "collection": "onlybook_faq_bge_m3_v1",
        "checks": {
            "embedder": {
                "ok": True,
                "type": "BGEEmbedder",
                "dim": 1024,
                "expected_dim": 1024,
            }
        },
    }
    report = check_readiness(cfg, startup_report=startup)
    assert report["ready"] is False
    assert report["checks"]["collection_dim"]["ok"] is False


def test_check_readiness_qdrant_down(monkeypatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    get_settings.cache_clear()
    _fake_qdrant(monkeypatch, name="onlybook_faq", dim=384, reachable=False)
    startup = {
        "status": "ok_hash_mode",
        "prefer_bge": False,
        "collection": "onlybook_faq",
        "checks": {
            "embedder": {"ok": True, "type": "OfflineHashEmbedder", "dim": 384}
        },
    }
    report = check_readiness(get_settings(), startup_report=startup)
    assert report["ready"] is False
    assert report["checks"]["qdrant_reachable"]["ok"] is False


def test_ready_does_not_clear_or_rebuild_embedder(monkeypatch) -> None:
    """Repeated lightweight readiness must not touch embedder cache/factory."""
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq_bge_m3_v1")
    get_settings.cache_clear()
    _fake_qdrant(monkeypatch, name="onlybook_faq_bge_m3_v1", dim=1024)

    calls: list[str] = []

    def boom_clear() -> None:
        calls.append("cache_clear")

    def boom_get(*args: Any, **kwargs: Any) -> Any:
        calls.append("get_dense_embedder")
        raise AssertionError("get_dense_embedder must not be called on /ready path")

    monkeypatch.setattr(
        "app.api.readiness.get_dense_embedder.cache_clear", boom_clear
    )
    monkeypatch.setattr("app.api.readiness.get_dense_embedder", boom_get)

    startup = {
        "status": "ok_bge_mode",
        "prefer_bge": True,
        "collection": "onlybook_faq_bge_m3_v1",
        "checks": {
            "embedder": {
                "ok": True,
                "type": "BGEEmbedder",
                "dim": 1024,
                "expected_dim": 1024,
            }
        },
    }
    cfg = get_settings()
    for _ in range(5):
        report = check_readiness(cfg, startup_report=startup)
        assert report["ready"] is True
        assert report.get("probe") == "lightweight"
    assert calls == []


def test_build_startup_snapshot_hash_no_bge(monkeypatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    get_settings.cache_clear()
    _fake_qdrant(monkeypatch, name="onlybook_faq", dim=384)

    clear = MagicMock()
    monkeypatch.setattr("app.api.readiness.get_dense_embedder.cache_clear", clear)
    snap = build_startup_readiness_snapshot(get_settings())
    assert snap["checks"]["embedder"]["type"] == "OfflineHashEmbedder"
    assert snap["checks"]["collection_dim"]["dim"] == 384
    # hash path should not need BGE cache_clear
    clear.assert_not_called()


async def test_ready_endpoint_503_on_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "true")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq_bge_m3_v1")
    get_settings.cache_clear()
    _fake_qdrant(monkeypatch, name="onlybook_faq_bge_m3_v1", dim=384)

    # Inject startup snapshot on app.state (lifespan may use real settings)
    app.state.startup_retrieval_report = {
        "status": "ok_bge_mode",
        "prefer_bge": True,
        "collection": "onlybook_faq_bge_m3_v1",
        "checks": {
            "embedder": {"ok": True, "type": "BGEEmbedder", "dim": 1024}
        },
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False


async def test_ready_endpoint_200_hash_mocked(monkeypatch) -> None:
    monkeypatch.setenv("PREFER_BGE", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    get_settings.cache_clear()
    _fake_qdrant(monkeypatch, name="onlybook_faq", dim=384)
    app.state.startup_retrieval_report = {
        "status": "ok_hash_mode",
        "prefer_bge": False,
        "collection": "onlybook_faq",
        "checks": {
            "embedder": {"ok": True, "type": "OfflineHashEmbedder", "dim": 384}
        },
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True
