"""LangGraph state and structured analysis models."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

RouteName = Literal[
    "simple_retrieve",
    "multi_hop_retrieve",
    "clarify",
    "out_of_scope",
]


class ChatMessage(TypedDict, total=False):
    role: str
    content: str


class QueryAnalysis(BaseModel):
    """Structured output from query analysis node."""

    intent: str = Field(description="Short intent label, e.g. refund_policy")
    complexity: Literal["simple", "multi_hop", "ambiguous", "out_of_scope"] = "simple"
    language: str = Field(default="en", description="ko|en|ja|zh")
    ambiguity: bool = False
    rationale: str = ""
    route: RouteName = "simple_retrieve"


class GraphState(TypedDict, total=False):
    messages: list[dict[str, str]]
    analysis: dict[str, Any]
    route: str
    documents: list[dict[str, Any]]
    answer: str
    needs_clarification: bool
    session_id: str
    language: str | None
    # Allowlisted exact-match Qdrant payload filters from the chat request
    metadata_filters: dict[str, str] | None
