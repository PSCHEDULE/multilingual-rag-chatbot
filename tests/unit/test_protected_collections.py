"""Guardrails: never recreate operational hash collections."""

import pytest

from app.retrieval.qdrant_store import (
    PROTECTED_COLLECTIONS,
    ProtectedCollectionError,
    assert_collection_write_allowed,
)


def test_protected_names_include_baselines() -> None:
    assert "onlybook_faq" in PROTECTED_COLLECTIONS
    assert "support_faq" in PROTECTED_COLLECTIONS


def test_recreate_blocked_on_protected() -> None:
    with pytest.raises(ProtectedCollectionError):
        assert_collection_write_allowed("onlybook_faq", recreate=True)
    with pytest.raises(ProtectedCollectionError):
        assert_collection_write_allowed("support_faq", recreate=True)


def test_recreate_allowed_on_versioned_bge_collection() -> None:
    assert_collection_write_allowed("onlybook_faq_bge_m3_v1", recreate=True)


def test_non_recreate_write_name_ok_for_protected() -> None:
    # Upsert without recreate is allowed at the guard layer (Stage 2 still
    # must not *target* protected names operationally).
    assert_collection_write_allowed("onlybook_faq", recreate=False)
