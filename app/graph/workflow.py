"""Compiled LangGraph workflow and run_turn entrypoint."""

from __future__ import annotations

import logging
from typing import Any

from app.graph import nodes
from app.graph.router import route_from_analysis
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

_GRAPH = None


def build_graph(use_spies: bool = False):
    from langgraph.graph import END, StateGraph

    analyze = nodes.analyze_query_traced if use_spies else nodes.analyze_query
    generate = nodes.generate_answer_traced if use_spies else nodes.generate_answer

    g = StateGraph(GraphState)
    g.add_node("analyze_query", analyze)
    g.add_node("simple_retrieve", nodes.simple_retrieve)
    g.add_node("multi_hop_retrieve", nodes.multi_hop_retrieve)
    g.add_node("clarify", nodes.clarify)
    g.add_node("out_of_scope", nodes.out_of_scope)
    g.add_node("generate_answer", generate)

    g.set_entry_point("analyze_query")
    g.add_conditional_edges(
        "analyze_query",
        route_from_analysis,
        {
            "simple_retrieve": "simple_retrieve",
            "multi_hop_retrieve": "multi_hop_retrieve",
            "clarify": "clarify",
            "out_of_scope": "out_of_scope",
        },
    )
    g.add_edge("simple_retrieve", "generate_answer")
    g.add_edge("multi_hop_retrieve", "generate_answer")
    g.add_edge("generate_answer", END)
    g.add_edge("clarify", END)
    g.add_edge("out_of_scope", END)
    return g.compile()


def get_graph(*, use_spies: bool = False):
    global _GRAPH
    if use_spies:
        return build_graph(use_spies=True)
    if _GRAPH is None:
        _GRAPH = build_graph(use_spies=False)
    return _GRAPH


def run_turn(
    state: dict[str, Any] | GraphState,
    *,
    llm: Any | None = None,
    use_spies: bool = False,
) -> GraphState:
    """
    Execute one conversation turn.

    ``state`` must include ``messages`` (list of {role, content}).
    Optional: ``language``, ``session_id``.
    """
    # Allow injecting llm into analyze/generate via node defaults is limited;
    # when llm is provided, call nodes directly in a simple sequential path for tests.
    if llm is not None:
        return _run_turn_with_llm(dict(state), llm=llm)

    graph = get_graph(use_spies=use_spies)
    result = graph.invoke(dict(state))
    return result  # type: ignore[return-value]


def _run_turn_with_llm(state: dict[str, Any], *, llm: Any) -> GraphState:
    current: GraphState = dict(state)  # type: ignore[assignment]
    current = nodes.analyze_query(current, llm=llm)
    route = route_from_analysis(current)
    current["route"] = route
    if route == "clarify":
        return nodes.clarify(current)
    if route == "out_of_scope":
        return nodes.out_of_scope(current)
    if route == "multi_hop_retrieve":
        current = nodes.multi_hop_retrieve(current)
    else:
        current = nodes.simple_retrieve(current)
    return nodes.generate_answer(current, llm=llm)


def stream_tokens(state: dict[str, Any], *, llm: Any | None = None):
    """
    Yield answer tokens for SSE (M5).

    Runs retrieval routing first, then streams generation.
    """
    client = llm or __import__("app.llm.client", fromlist=["get_llm_client"]).get_llm_client()
    # Non-stream path for analyze + retrieve
    interim = nodes.analyze_query(dict(state), llm=client)  # type: ignore[arg-type]
    route = route_from_analysis(interim)  # type: ignore[arg-type]
    interim["route"] = route
    yield {"event": "meta", "data": {"language": interim.get("language"), "route": route}}

    if route == "clarify":
        out = nodes.clarify(interim)  # type: ignore[arg-type]
        for ch in out.get("answer") or "":
            yield {"event": "token", "data": {"text": ch}}
        yield {
            "event": "sources",
            "data": {"items": []},
        }
        yield {"event": "done", "data": {"finish_reason": "stop", "answer": out.get("answer")}}
        return
    if route == "out_of_scope":
        out = nodes.out_of_scope(interim)  # type: ignore[arg-type]
        for ch in out.get("answer") or "":
            yield {"event": "token", "data": {"text": ch}}
        yield {"event": "sources", "data": {"items": []}}
        yield {"event": "done", "data": {"finish_reason": "stop", "answer": out.get("answer")}}
        return

    if route == "multi_hop_retrieve":
        interim = nodes.multi_hop_retrieve(interim)  # type: ignore[arg-type]
    else:
        interim = nodes.simple_retrieve(interim)  # type: ignore[arg-type]

    docs = interim.get("documents") or []
    from app.utils.public_source import public_source_id

    def _pub_id(d: dict[str, Any]) -> str:
        return public_source_id(
            d.get("source") if isinstance(d.get("source"), str) else None,
            faq_id=d.get("faq_id") if isinstance(d.get("faq_id"), str) else None,
            language=d.get("language") if isinstance(d.get("language"), str) else None,
        )

    # Server-side debug: full internal paths (not sent to client)
    for d in docs:
        pub = _pub_id(d)
        logger.info(
            "sources_selected title=%r public_id=%s internal_source=%s score=%s",
            d.get("title"),
            pub,
            d.get("source"),
            d.get("score"),
        )
    yield {
        "event": "sources",
        "data": {
            "items": [
                {
                    "title": d.get("title"),
                    "score": d.get("score"),
                    # Language-unique public id (e.g. faq-ja-Q15) — not full data/ paths
                    "source": _pub_id(d),
                }
                for d in docs
            ]
        },
    }

    # Stream generation
    from app.graph.nodes import GENERATE_SYSTEM, _history_text, _latest_user
    from app.utils.language import detect_language

    messages = list(interim.get("messages") or [])
    history = _history_text(messages)
    latest = _latest_user(messages)
    lang = interim.get("language") or detect_language(latest)
    ctx = "\n\n".join(
        f"Source: {d.get('title') or d.get('source')}\n{d.get('text')}" for d in docs
    ) or "(no retrieved context)"
    system = GENERATE_SYSTEM.format(language=lang)
    user_prompt = (
        f"Conversation history:\n{history}\n\n"
        f"Retrieved context:\n{ctx}\n\n"
        "Answer the latest user message.\n"
        "Do not append bracketed numbers like [1] or [2] as citations."
    )
    answer_parts: list[str] = []
    for token in client.stream(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
    ):
        answer_parts.append(token)
        yield {"event": "token", "data": {"text": token}}
    answer = "".join(answer_parts)
    yield {"event": "done", "data": {"finish_reason": "stop", "answer": answer}}
