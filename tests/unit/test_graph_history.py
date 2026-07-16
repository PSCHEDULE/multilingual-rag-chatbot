"""History is passed into analyze_query and generate_answer on turn 2."""

from app.graph import nodes
from app.graph.workflow import run_turn
from app.llm.client import MockLLMClient


class HistoryMock(MockLLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, messages, *, temperature=0.2, max_tokens=1024) -> str:
        blob = "\n".join(m.get("content", "") for m in messages)
        self.prompts.append(blob)
        if "JSON" in blob or "Analyze this support" in blob:
            return (
                '{"intent":"followup","complexity":"simple","language":"en",'
                '"ambiguity":false,"rationale":"t","route":"simple_retrieve"}'
            )
        return "You have 14 days to request a refund."


def test_second_turn_includes_prior_messages() -> None:
    nodes.reset_spies()
    llm = HistoryMock()
    r1 = run_turn(
        {"messages": [{"role": "user", "content": "What is your refund policy?"}]},
        llm=llm,
    )
    assert r1.get("answer")
    messages = list(r1["messages"])
    assert any(m.get("role") == "assistant" for m in messages)

    r2 = run_turn(
        {
            "messages": messages
            + [{"role": "user", "content": "How many days do I have?"}]
        },
        llm=llm,
        use_spies=True,
    )
    assert len(r2["messages"]) >= 3

    # Prompts to LLM on second turn should include first user question
    joined = "\n".join(llm.prompts)
    assert "What is your refund policy?" in joined
    assert "How many days do I have?" in joined
