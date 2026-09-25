from family_ai.agent_runtime import AgentRuntime
from family_ai.registry_agent import RegistryAgent
from family_ai.storage import FamilyStorage


def test_registry_proposal_stays_pending_until_explicit_decision(tmp_path):
    agent = RegistryAgent(FamilyStorage(tmp_path / "family.db"),
                          AgentRuntime(sleeper=lambda _: None))
    proposed = agent.propose("member", "Calendar Agent", "read calendar", ["calendar"])
    assert proposed.status == "success"
    assert proposed.value["status"] == "pending"
    approved = agent.decide(proposed.value["id"][:8], True)
    assert approved.status == "success" and approved.verified


def test_registry_rejects_unknown_permission(tmp_path):
    agent = RegistryAgent(FamilyStorage(tmp_path / "family.db"),
                          AgentRuntime(sleeper=lambda _: None))
    try:
        agent.propose("member", "Unsafe", "unsafe", ["root"])
    except ValueError as exc:
        assert "Unsupported permissions" in str(exc)
    else:
        raise AssertionError("unknown permission was accepted")
