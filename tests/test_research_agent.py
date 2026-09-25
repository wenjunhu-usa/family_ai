from types import SimpleNamespace

from family_ai.agent_runtime import AgentRuntime
from family_ai.research_agent import ResearchAgent


def test_research_agent_uses_multiple_queries_deduplicates_and_survives_partial_failure():
    def search(query, **_kwargs):
        if query == "broken":
            raise ConnectionError("offline")
        return [{"url": "https://example.com/a", "title": query}]

    agent = ResearchAgent(
        AgentRuntime(sleeper=lambda _: None), lambda *_: ["first", "broken", "third"],
        search, lambda rows: str(rows), lambda _: SimpleNamespace(content="answer"),
    )
    result = agent.collect("request", "base", 3)
    assert result.status == "success" and result.verified
    assert len(result.value["results"]) == 1
    assert [x["status"] for x in result.value["attempts"]] == ["success", "failed", "success"]


def test_research_agent_returns_structured_failure_when_all_queries_fail():
    agent = ResearchAgent(
        AgentRuntime(max_attempts=1, sleeper=lambda _: None), lambda *_: ["one"],
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError("offline")),
        str, lambda _: SimpleNamespace(content="answer"),
    )
    result = agent.collect("request", "base")
    assert result.status == "needs_specialist"
