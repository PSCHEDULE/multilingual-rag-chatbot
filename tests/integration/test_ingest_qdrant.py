"""Integration: ingest sample docs into an isolated disposable Qdrant collection.

Never uses or recreates protected operational collections (onlybook_faq,
support_faq, etc.). Skips cleanly when Qdrant is unavailable.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.config import get_settings
from app.retrieval.qdrant_store import (
    PROTECTED_COLLECTIONS,
    assert_collection_write_allowed,
    get_qdrant_client,
    qdrant_reachable,
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "sample_docs"

pytestmark = pytest.mark.integration

# Unmistakable disposable prefix — never a protected baseline name.
_DISPOSABLE_PREFIX = "pytest_ingest_"


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    get_settings.cache_clear()


def _disposable_collection_name() -> str:
    name = f"{_DISPOSABLE_PREFIX}{uuid.uuid4().hex[:12]}"
    assert name not in PROTECTED_COLLECTIONS
    assert not name.startswith("onlybook_faq")
    assert name != "support_faq"
    # Must be allowed for recreate
    assert_collection_write_allowed(name, recreate=True)
    return name


def test_ingest_sample_docs() -> None:
    if not qdrant_reachable():
        pytest.skip("Qdrant not reachable")

    from app.ingestion.pipeline import run_ingest

    collection = _disposable_collection_name()
    client = get_qdrant_client(get_settings())
    try:
        stats = run_ingest(
            SAMPLE,
            collection=collection,
            recreate=True,
            prefer_bge=False,
            contextualize=True,
        )
        assert stats["points"] > 0
        assert stats["collection"] == collection
        assert set(stats["languages"]) >= {"ko", "en", "ja", "zh"}

        info = client.get_collection(collection)
        assert info.points_count and info.points_count > 0

        points, _ = client.scroll(
            collection_name=collection,
            limit=100,
            with_payload=True,
        )
        langs = {p.payload.get("language") for p in points if p.payload}
        assert langs >= {"ko", "en", "ja", "zh"}
        assert all(p.payload.get("text") for p in points if p.payload)

        # Never target the process-configured protected baseline
        configured = get_settings().qdrant_collection
        assert collection != configured or configured not in PROTECTED_COLLECTIONS
        assert collection not in PROTECTED_COLLECTIONS
    finally:
        # Cleanup must run on failure as well
        try:
            client.delete_collection(collection)
        except Exception:  # noqa: BLE001
            pass
