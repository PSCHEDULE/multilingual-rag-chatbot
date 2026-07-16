"""
M8-B Stage 1: BGE-M3 load + multilingual embed smoke.

Requires optional dependency group:
  uv sync --group bge

No Qdrant collections are created or modified.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.bge

# Short FAQ-like strings (one per product language).
_SAMPLES = {
    "en": "How do I get a refund?",
    "ko": "환불은 어떻게 받나요?",
    "ja": "返金はどうすればいいですか？",
    "zh": "如何申请退款？",
}
_UNRELATED_EN = "The weather forecast predicts rain tomorrow."


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@pytest.fixture(scope="module")
def bge_embedder():
    """Load BGE once per module; skip cleanly if optional deps missing."""
    if not _sentence_transformers_available():
        pytest.skip(
            "optional bge group not installed (uv sync --group bge)"
        )

    from app.retrieval.embeddings import BGEEmbedder, OfflineHashEmbedder, get_dense_embedder

    get_dense_embedder.cache_clear()
    try:
        emb = get_dense_embedder(
            prefer_bge=True,
            model_name="BAAI/bge-m3",
            require_bge=True,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"BGE-M3 model not loadable: {exc}")

    # Must not be silent hash fallback
    assert isinstance(emb, BGEEmbedder), (
        f"expected BGEEmbedder, got {type(emb).__name__} (silent fallback?)"
    )
    assert not isinstance(emb, OfflineHashEmbedder)
    assert emb.dim != OfflineHashEmbedder().dim, (
        f"BGE dim {emb.dim} unexpectedly equals hash dim"
    )
    return emb


def test_bge_m3_loads_and_reports_dimension(bge_embedder) -> None:
    """Measure dense dimension at runtime (do not hardcode as sole gate)."""
    dim = bge_embedder.dim
    # Sanity band for BGE-M3-class dense models; actual value is measured below.
    assert dim > DEFAULT_MIN_DIM, f"unexpectedly small dim={dim}"
    vec = bge_embedder.embed_query("dimension probe")
    assert len(vec) == dim
    assert all(math.isfinite(x) for x in vec)
    # encode(..., normalize_embeddings=True) → unit-ish L2 norm
    norm = math.sqrt(sum(x * x for x in vec))
    assert 0.99 <= norm <= 1.01, f"expected normalized vector, L2={norm}"
    # Emit measured dim for operators / Stage 1 report (pytest -s shows it)
    print(f"\n[M8-B Stage1] BGE model={bge_embedder.model_name} measured_dense_dim={dim}")


# Lower bound only — real BGE-M3 is typically 1024; we measure, not hard-require 1024.
DEFAULT_MIN_DIM = 256


def test_bge_m3_multilingual_smoke(bge_embedder) -> None:
    """Embed KO/EN/JA/ZH samples; vectors valid; refunds closer than unrelated EN."""
    dim = bge_embedder.dim
    vectors: dict[str, list[float]] = {}
    for lang, text in _SAMPLES.items():
        v = bge_embedder.embed_query(text)
        assert len(v) == dim, f"{lang}: len {len(v)} != dim {dim}"
        assert all(math.isfinite(x) for x in v), f"{lang}: non-finite values"
        vectors[lang] = v

    unrelated = bge_embedder.embed_query(_UNRELATED_EN)
    # Semantic sanity: EN refund should be closer to KO refund than to weather text
    sim_en_ko = _cosine(vectors["en"], vectors["ko"])
    sim_en_unrelated = _cosine(vectors["en"], unrelated)
    assert sim_en_ko > sim_en_unrelated, (
        f"expected EN~KO refund similarity ({sim_en_ko:.4f}) > "
        f"EN~unrelated ({sim_en_unrelated:.4f})"
    )
    # All four language refund queries should be mutually more similar than EN–unrelated
    for lang in ("ja", "zh"):
        sim = _cosine(vectors["en"], vectors[lang])
        assert sim > sim_en_unrelated, (
            f"expected EN~{lang} ({sim:.4f}) > EN~unrelated ({sim_en_unrelated:.4f})"
        )


def test_get_dense_embedder_require_bge_no_silent_fallback(bge_embedder) -> None:
    """Factory with require_bge=True returns BGEEmbedder, never OfflineHashEmbedder."""
    from app.retrieval.embeddings import BGEEmbedder, OfflineHashEmbedder, get_dense_embedder

    get_dense_embedder.cache_clear()
    emb = get_dense_embedder(prefer_bge=True, model_name="BAAI/bge-m3", require_bge=True)
    assert isinstance(emb, BGEEmbedder)
    assert not isinstance(emb, OfflineHashEmbedder)
    assert emb.dim == bge_embedder.dim
