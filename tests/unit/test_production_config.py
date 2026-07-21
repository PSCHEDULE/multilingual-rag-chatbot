"""Production packaging policy: mock rejection and provider secret requirements."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_production_rejects_mock_llm_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MOCK_LLM", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
    with pytest.raises(ValidationError, match="MOCK_LLM must be false"):
        Settings(_env_file=None)


def test_prod_alias_rejects_mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
    with pytest.raises(ValidationError, match="MOCK_LLM must be false"):
        Settings(_env_file=None)


def test_production_requires_provider_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="OPENAI_API_KEY or LLM_API_KEY"):
        Settings(_env_file=None)


def test_production_ok_with_mock_false_and_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
    cfg = Settings(_env_file=None)
    assert cfg.is_production() is True
    assert cfg.mock_llm is False
    assert cfg.resolved_llm_api_key() == "sk-test-not-real"


def test_non_production_allows_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("MOCK_LLM", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = Settings(_env_file=None)
    assert cfg.is_production() is False
    assert cfg.mock_llm is True
