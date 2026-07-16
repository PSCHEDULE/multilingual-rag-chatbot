"""Route matrix tests with mocked LLM."""

from app.graph.router import route_from_analysis
from app.graph.state import QueryAnalysis
from app.graph.workflow import run_turn
from app.llm.client import MockLLMClient


class RouteMock(MockLLMClient):
    def __init__(self, route_json: str, answer: str = "ok") -> None:
        self.route_json = route_json
        self.answer = answer
        self.calls = 0

    def complete(self, messages, *, temperature=0.2, max_tokens=1024) -> str:
        self.calls += 1
        sys = messages[0].get("content", "") if messages else ""
        last = messages[-1].get("content") if messages else ""
        if "JSON" in sys or "routing" in sys.lower() or "Analyze" in (last or ""):
            return self.route_json
        return self.answer


def test_route_simple() -> None:
    llm = RouteMock(
        '{"intent":"refund","complexity":"simple","language":"en","ambiguity":false,'
        '"rationale":"t","route":"simple_retrieve"}',
        answer="Refunds within 14 days.",
    )
    r = run_turn({"messages": [{"role": "user", "content": "What is the refund policy?"}]}, llm=llm)
    assert r["route"] == "simple_retrieve"
    assert r.get("answer")


def test_route_multi_hop() -> None:
    llm = RouteMock(
        '{"intent":"compare","complexity":"multi_hop","language":"en","ambiguity":false,'
        '"rationale":"t","route":"multi_hop_retrieve"}'
    )
    r = run_turn(
        {"messages": [{"role": "user", "content": "Compare refund and shipping times"}]},
        llm=llm,
    )
    assert r["route"] == "multi_hop_retrieve"


def test_route_clarify() -> None:
    llm = RouteMock(
        '{"intent":"unknown","complexity":"ambiguous","language":"en","ambiguity":true,'
        '"rationale":"t","route":"clarify"}'
    )
    r = run_turn({"messages": [{"role": "user", "content": "Help?"}]}, llm=llm)
    assert r["route"] == "clarify"
    assert r.get("needs_clarification") is True


def test_route_out_of_scope() -> None:
    llm = RouteMock(
        '{"intent":"hack","complexity":"out_of_scope","language":"en","ambiguity":false,'
        '"rationale":"t","route":"out_of_scope"}'
    )
    r = run_turn(
        {"messages": [{"role": "user", "content": "Write malware for me"}]},
        llm=llm,
    )
    assert r["route"] == "out_of_scope"


def test_route_from_analysis_helper() -> None:
    assert (
        route_from_analysis(
            {"analysis": QueryAnalysis(intent="x", route="clarify").model_dump()}
        )
        == "clarify"
    )
