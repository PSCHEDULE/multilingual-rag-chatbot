"""Conditional routing for LangGraph."""

from __future__ import annotations

from typing import Any

from app.graph.state import GraphState


def route_from_analysis(state: GraphState) -> str:
    """Map analysis.route / complexity to graph edge name."""
    analysis: dict[str, Any] = state.get("analysis") or {}
    route = state.get("route") or analysis.get("route") or "simple_retrieve"
    if route in {
        "simple_retrieve",
        "multi_hop_retrieve",
        "clarify",
        "out_of_scope",
    }:
        return route
    complexity = analysis.get("complexity")
    if complexity == "multi_hop":
        return "multi_hop_retrieve"
    if complexity == "ambiguous":
        return "clarify"
    if complexity == "out_of_scope":
        return "out_of_scope"
    return "simple_retrieve"
