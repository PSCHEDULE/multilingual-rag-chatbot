"""Langfuse no-op when keys missing."""

from app.observability.langfuse_setup import get_langfuse, reset_langfuse_for_tests


def test_langfuse_disabled_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    reset_langfuse_for_tests()
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_langfuse() is None
