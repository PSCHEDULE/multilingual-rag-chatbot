"""CORS middleware policy: wildcard disables credentials; production rejects *."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_wildcard_origin_disables_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    monkeypatch.setenv("MOCK_LLM", "true")
    cfg = Settings(_env_file=None)
    assert cfg.cors_origin_list() == ["*"]
    assert cfg.cors_allow_credentials() is False
    kwargs = cfg.cors_middleware_kwargs()
    assert kwargs["allow_origins"] == ["*"]
    assert kwargs["allow_credentials"] is False


def test_explicit_origins_enable_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com,https://admin.example.com")
    monkeypatch.setenv("MOCK_LLM", "true")
    cfg = Settings(_env_file=None)
    assert cfg.cors_origin_list() == [
        "https://app.example.com",
        "https://admin.example.com",
    ]
    assert cfg.cors_allow_credentials() is True
    kwargs = cfg.cors_middleware_kwargs()
    assert kwargs["allow_credentials"] is True


def test_production_rejects_wildcard_cors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(ValidationError, match="CORS_ORIGINS"):
        Settings(_env_file=None)


def test_production_accepts_explicit_cors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MOCK_LLM", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
    cfg = Settings(_env_file=None)
    assert cfg.cors_allow_credentials() is True
    assert cfg.cors_origin_list() == ["https://app.example.com"]
