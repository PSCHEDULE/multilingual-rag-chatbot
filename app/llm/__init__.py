"""LLM client factory (default: OpenAI gpt-4o-mini)."""

from app.llm.client import get_llm_client

__all__ = ["get_llm_client"]
