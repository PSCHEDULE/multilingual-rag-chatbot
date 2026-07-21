"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Production-like APP_ENV values that require hardened runtime policy.
_PRODUCTION_ENVS = frozenset({"prod", "production"})
# Providers that require a configured API key when mock is disabled in production.
_SECRET_REQUIRED_PROVIDERS = frozenset({"openai", "xai", "grok"})


class Settings(BaseSettings):
    """Runtime configuration. Secrets come from env only — never commit `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Multilingual RAG Chatbot"
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    cors_origins: str = Field(
        default="*",
        validation_alias="CORS_ORIGINS",
        description="Comma-separated origins, or *",
    )

    # LLM — single model strategy: OpenAI gpt-4o-mini (M2, M4, M7)
    llm_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="LLM_BASE_URL",
    )
    llm_temperature: float = Field(default=0.2, validation_alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, validation_alias="LLM_MAX_TOKENS")
    mock_llm: bool = Field(default=False, validation_alias="MOCK_LLM")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        default="support_faq",
        validation_alias="QDRANT_COLLECTION",
        description=(
            "Active retrieval collection. Cutover target: onlybook_faq_bge_m3_v1. "
            "Rollback: onlybook_faq (hash)."
        ),
    )

    # Embeddings / reranker
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        validation_alias="EMBEDDING_MODEL",
    )
    prefer_bge: bool = Field(
        default=False,
        validation_alias="PREFER_BGE",
        description=(
            "Use BGE-M3 dense embeddings for retrieval. Must match collection dim: "
            "true with onlybook_faq_bge_m3_v1; false with hash onlybook_faq."
        ),
    )
    retrieval_language_filter: bool = Field(
        default=True,
        validation_alias="RETRIEVAL_LANGUAGE_FILTER",
        description="When true, pass detected language into hybrid search filters.",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        validation_alias="RERANKER_MODEL",
    )
    retrieval_top_n: int = Field(default=40, validation_alias="RETRIEVAL_TOP_N")
    retrieval_top_k: int = Field(default=6, validation_alias="RETRIEVAL_TOP_K")
    # Post-rerank pruning (generation context + sources event)
    prune_absolute_threshold: float = Field(
        default=0.05,
        validation_alias="PRUNE_ABSOLUTE_THRESHOLD",
        description="Min score for Top-2/Top-3 to be kept after rerank.",
    )
    prune_relative_threshold: float = Field(
        default=0.05,
        validation_alias="PRUNE_RELATIVE_THRESHOLD",
        description="Min fraction of Top-1 score required for Top-2/Top-3 (e.g. 0.05 = 5%).",
    )

    # Ingestion
    contextualize: bool = Field(default=True, validation_alias="CONTEXTUALIZE")
    chunk_breakpoint_percentile: int = Field(
        default=95,
        validation_alias="CHUNK_BREAKPOINT_PERCENTILE",
    )

    # Langfuse (optional)
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")

    # Defaults
    default_language: str = Field(default="en", validation_alias="DEFAULT_LANGUAGE")
    session_history_max_turns: int = Field(
        default=20,
        validation_alias="SESSION_HISTORY_MAX_TURNS",
    )

    def resolved_llm_api_key(self) -> str | None:
        """Prefer OPENAI_API_KEY, then LLM_API_KEY."""
        return self.openai_api_key or self.llm_api_key

    def is_production(self) -> bool:
        """True when APP_ENV is a production deployment label."""
        return (self.app_env or "").strip().lower() in _PRODUCTION_ENVS

    def cors_origin_list(self) -> list[str]:
        """Parsed CORS origins. A sole ``*`` means reflect-any (credentials off)."""
        raw = (self.cors_origins or "").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    def cors_allow_credentials(self) -> bool:
        """
        Browser-safe credentials flag for CORSMiddleware.

        Credentials must never be enabled with a wildcard origin list.
        Explicit non-wildcard origins may enable credentials.
        """
        origins = self.cors_origin_list()
        if not origins or any(o == "*" for o in origins):
            return False
        return True

    def cors_middleware_kwargs(self) -> dict[str, Any]:
        """Keyword args for ``CORSMiddleware`` derived from settings."""
        return {
            "allow_origins": self.cors_origin_list(),
            "allow_credentials": self.cors_allow_credentials(),
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

    def language_filter_value(self, language: str | None) -> str | None:
        """Return language for hybrid filter, or None when filter is disabled."""
        if not self.retrieval_language_filter:
            return None
        if not language:
            return None
        return str(language).strip() or None

    @model_validator(mode="after")
    def enforce_production_runtime_policy(self) -> Self:
        """Hard-reject unsafe production combinations at settings initialization.

        Rules (APP_ENV in {prod, production}):
        - MOCK_LLM must not be enabled
        - selected secret-backed providers require OPENAI_API_KEY or LLM_API_KEY
        - CORS_ORIGINS must be explicit (not wildcard) so credentials-safe origins
          can be used without an invalid browser CORS combination
        """
        if not self.is_production():
            return self
        if self.mock_llm:
            raise ValueError(
                "Production startup rejected: MOCK_LLM must be false when "
                f"APP_ENV is production (got APP_ENV={self.app_env!r}, MOCK_LLM=true)"
            )
        provider = (self.llm_provider or "").strip().lower()
        if provider in _SECRET_REQUIRED_PROVIDERS and not self.resolved_llm_api_key():
            raise ValueError(
                "Production startup rejected: OPENAI_API_KEY or LLM_API_KEY is "
                f"required when APP_ENV is production and LLM_PROVIDER={provider!r}"
            )
        origins = self.cors_origin_list()
        if not origins or any(o == "*" for o in origins):
            raise ValueError(
                "Production startup rejected: CORS_ORIGINS must list explicit "
                "origins (not *) when APP_ENV is production"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Module-level convenience; tests may call get_settings.cache_clear()
settings = get_settings()
