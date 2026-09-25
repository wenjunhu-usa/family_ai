from family_ai.agent_runtime import AgentRuntime
from family_ai.productivity_agents import MemoryAgent, TasksAgent
from family_ai.storage import FamilyStorage


def test_memory_agent_writes_then_verifies_scope(tmp_path):
    storage = FamilyStorage(tmp_path / "family.db")
    agent = MemoryAgent(storage, AgentRuntime(sleeper=lambda _: None))
    result = agent.add("member", "likes tea", "private")
    assert result.status == "success" and result.verified
    listed = agent.list("member")
    assert listed.value[0]["content"] == "likes tea"


def test_tasks_agent_adds_and_verifies_completion(tmp_path):
    storage = FamilyStorage(tmp_path / "family.db")
    agent = TasksAgent(storage, AgentRuntime(sleeper=lambda _: None))
    created = agent.add("member", "buy milk", "me")
    assert created.status == "success" and created.verified
    completed = agent.complete("member", created.value["id"][:8])
    assert completed.status == "success" and completed.verified


def test_tasks_agent_returns_structured_permanent_error_for_unknown_task(tmp_path):
    storage = FamilyStorage(tmp_path / "family.db")
    agent = TasksAgent(storage, AgentRuntime(sleeper=lambda _: None))
    outcome = agent.complete("member", "missing")
    assert outcome.status == "permanent_error"
    assert outcome.error_type == "ValueError"
