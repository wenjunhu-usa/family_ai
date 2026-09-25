from family_ai.agent_core import AgentDecision, AutonomousAgent, ToolSpec
from family_ai.agent_runtime import AgentOutcome


class ScriptedModel:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def invoke(self, _messages):
        return next(self.decisions)


class FailingModel:
    def invoke(self, _messages):
        raise ValueError("invalid structured output")


def outcome(status="success", value=None):
    return AgentOutcome(status=status, value=value, agent="test", operation="test",
                        verified=status == "success")


def test_agent_observes_failure_then_uses_diagnostic_and_finishes():
    calls = []
    model = ScriptedModel([
        AgentDecision(action="use_tool", tool="sync", arguments={}),
        AgentDecision(action="use_tool", tool="health", arguments={}),
        AgentDecision(action="use_tool", tool="sync", arguments={"retry": True}),
        AgentDecision(action="finish", answer="同步完成", completion_evidence=["sync verified"]),
    ])

    def sync(args):
        calls.append(("sync", args))
        return outcome("success", {"synced": 2}) if args else outcome("transient_error")

    agent = AutonomousAgent("Main Agent", model, [
        ToolSpec("sync", "sync accounts", sync),
        ToolSpec("health", "check connectivity", lambda _: outcome(value={"db": "ok"})),
    ])
    trace = agent.run("同步所有 Gmail")
    assert trace.status == "completed"
    assert [item["tool"] for item in trace.observations] == ["sync", "health", "sync"]
    assert calls[-1] == ("sync", {"retry": True})


def test_agent_rejects_unregistered_tool_and_can_recover():
    model = ScriptedModel([
        AgentDecision(action="use_tool", tool="shell", arguments={}),
        AgentDecision(action="clarify", answer="需要授权工具"),
    ])
    trace = AutonomousAgent("Main", model, []).run("delete something")
    assert trace.status == "needs_input"
    assert trace.observations[0]["status"] == "permanent_error"


def test_agent_requires_evidence_after_using_a_tool():
    model = ScriptedModel([
        AgentDecision(action="use_tool", tool="check", arguments={}),
        AgentDecision(action="finish", answer="done"),
        AgentDecision(action="finish", answer="done", completion_evidence=["check succeeded"]),
    ])
    trace = AutonomousAgent("Main", model, [
        ToolSpec("check", "check status", lambda _: outcome(value="ok")),
    ]).run("verify status")
    assert trace.status == "completed"
    assert any(item["status"] == "invalid_finish" for item in trace.observations)


def test_control_model_can_finish_with_evidence_without_writing_final_prose():
    model = ScriptedModel([
        AgentDecision(action="finish", completion_evidence=["database healthy"])
    ])
    trace = AutonomousAgent("Main", model, []).run(
        "check database", initial_observations=[{"tool": "database", "status": "success"}]
    )
    assert trace.status == "completed"
    assert trace.answer == ""


def test_agent_stops_at_bounded_step_limit():
    model = ScriptedModel([
        AgentDecision(action="use_tool", tool="check", arguments={}),
        AgentDecision(action="use_tool", tool="check", arguments={}),
    ])
    trace = AutonomousAgent("Main", model, [
        ToolSpec("check", "check status", lambda _: outcome(value="same")),
    ], max_steps=2).run("never complete")
    assert trace.status == "failed"
    assert trace.steps == 2


def test_decision_parse_errors_are_observed_not_raised():
    trace = AutonomousAgent("Main", FailingModel(), [], max_steps=8).run("status")
    assert trace.status == "failed"
    assert trace.steps == 2
    assert trace.observations[-1]["status"] == "decision_error"


def test_initial_specialist_observation_is_available_to_reasoner():
    model = ScriptedModel([
        AgentDecision(action="finish", answer="not running", completion_evidence=["status observation"])
    ])
    trace = AutonomousAgent("Main", model, []).run(
        "is it running", initial_observations=[{"tool": "status", "status": "success"}]
    )
    assert trace.status == "completed"
    assert trace.observations[0]["tool"] == "status"
