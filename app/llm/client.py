"""LLM client factory. Default: OpenAI gpt-4o-mini (used in M2, M4, M7)."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Minimal chat completion interface used across the app."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ): ...


class MockLLMClient:
    """Deterministic client for tests and offline gates."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return f"[mock] Acknowledged: {last[:200]}"

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        text = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        yield from text


class OpenAICompatibleClient:
    """OpenAI Chat Completions client (also works with OpenAI-compatible base URLs)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        default_temperature: float = 0.2,
        default_max_tokens: int = 1024,
    ) -> None:
        from openai import OpenAI

        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=(
                self._default_temperature if temperature is None else temperature
            ),
            max_tokens=self._default_max_tokens if max_tokens is None else max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=(
                self._default_temperature if temperature is None else temperature
            ),
            max_tokens=self._default_max_tokens if max_tokens is None else max_tokens,
            stream=True,
        )
        for chunk in stream:
            # Streaming iterators may yield non-chunk control items; only process
            # objects that expose the OpenAI chat chunk shape.
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta.content
            if delta:
                yield delta


def get_llm_client(settings: Settings | None = None) -> Any:
    """Return an LLM client. Uses MockLLM when MOCK_LLM=1 or no API key."""
    cfg = settings or get_settings()
    if cfg.mock_llm:
        logger.info("Using MockLLMClient (MOCK_LLM=true)")
        return MockLLMClient()

    provider = (cfg.llm_provider or "openai").lower()
    api_key = cfg.resolved_llm_api_key()

    if provider in {"openai", "xai", "grok"}:
        if not api_key:
            logger.warning(
                "No LLM API key set; falling back to MockLLMClient. "
                "Set OPENAI_API_KEY or LLM_API_KEY for live gpt-4o-mini calls."
            )
            return MockLLMClient()

        base_url = cfg.llm_base_url
        if provider == "openai" and "x.ai" in (base_url or ""):
            base_url = "https://api.openai.com/v1"
        if provider in {"xai", "grok"} and "openai.com" in (base_url or ""):
            base_url = "https://api.x.ai/v1"

        logger.info(
            "Using OpenAI-compatible client provider=%s model=%s",
            provider,
            cfg.llm_model,
        )
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=cfg.llm_model,
            default_temperature=cfg.llm_temperature,
            default_max_tokens=cfg.llm_max_tokens,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r}. Supported: openai, xai, grok"
    )
