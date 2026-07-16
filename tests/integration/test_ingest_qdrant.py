"""Integration: ingest sample docs into Qdrant (skipped if Qdrant down)."""

from pathlib import Path

import pytest

from app.config import get_settings
from app.retrieval.qdrant_store import get_qdrant_client, qdrant_reachable

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "sample_docs"

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    get_settings.cache_clear()


def test_ingest_sample_docs() -> None:
    if not qdrant_reachable():
        pytest.skip("Qdrant not reachable")

    from app.ingestion.pipeline import run_ingest

    settings = get_settings()
    stats = run_ingest(
        SAMPLE,
        collection=settings.qdrant_collection,
        recreate=True,
        prefer_bge=False,
        contextualize=True,
    )
    assert stats["points"] > 0
    assert set(stats["languages"]) >= {"ko", "en", "ja", "zh"}

    client = get_qdrant_client(settings)
    info = client.get_collection(settings.qdrant_collection)
    assert info.points_count and info.points_count > 0

    # language filter presence via scroll
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=100,
        with_payload=True,
    )
    langs = {p.payload.get("language") for p in points if p.payload}
    assert langs >= {"ko", "en", "ja", "zh"}
    assert all(p.payload.get("text") for p in points if p.payload)
