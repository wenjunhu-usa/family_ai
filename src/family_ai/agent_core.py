from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, Field

from .agent_runtime import AgentOutcome


class AgentDecision(BaseModel):
    """One externally observable decision in an agent's bounded reasoning loop."""

    action: Literal["use_tool", "finish", "clarify"]
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None
    completion_evidence: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[[dict[str, Any]], AgentOutcome]
    permission: str | None = None


@dataclass
class AgentTrace:
    goal: str
    status: Literal["running", "completed", "needs_input", "failed"] = "running"
    observations: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    steps: int = 0


class DecisionModel(Protocol):
    def invoke(self, messages: list[Any]) -> AgentDecision: ...


class AutonomousAgent:
    """Bounded observe-plan-act-verify loop shared by Main and specialist agents.

    The model never invokes Python directly. It chooses from an allowlisted registry,
    receives sanitized structured observations, and must cite completion evidence.
    """

    def __init__(self, name: str, decision_model: DecisionModel,
                 tools: list[ToolSpec], *, max_steps: int = 8):
        self.name = name
        self.decision_model = decision_model
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max(1, max_steps)

    def _prompt(self, trace: AgentTrace, context: str) -> list[dict[str, str]]:
        tools = [
            {"name": item.name, "description": item.description,
             "permission": item.permission or "none"}
            for item in self.tools.values()
        ]
        state = {
            "goal": trace.goal,
            "step": trace.steps + 1,
            "max_steps": self.max_steps,
            "tools": tools,
            "observations": trace.observations[-6:],
            "context": context,
        }
        return [{
            "role": "system",
            "content": (
                f"You are {self.name}, an autonomous but bounded private assistant. "
                "At every step, analyze the goal and observations, then choose exactly one "
                "action: use_tool, clarify, or finish. Use only listed tools. A failed tool "
                "is an observation: diagnose or choose another useful action instead of merely "
                "repeating it. Finish only when the user's goal is satisfied and include concrete "
                "completion_evidence. Never claim an action succeeded without a success observation. "
                "Do not reveal hidden reasoning. Return the AgentDecision schema.\n\nSTATE:\n"
                + json.dumps(state, ensure_ascii=False, default=str)
            ),
        }]

    @staticmethod
    def _observation(tool: str, outcome: AgentOutcome) -> dict[str, Any]:
        return {
            "tool": tool,
            "status": outcome.status,
            "operation": outcome.operation,
            "attempts": outcome.attempts,
            "verified": outcome.verified,
            "result": outcome.value if outcome.status == "success" else None,
            "error_type": outcome.error_type,
            "error_summary": outcome.error_summary,
            "diagnostics": outcome.diagnostics,
        }

    def run(self, goal: str, *, context: str = "",
            initial_observations: list[dict[str, Any]] | None = None) -> AgentTrace:
        trace = AgentTrace(goal=goal, observations=list(initial_observations or []))
        repeated_action: tuple[str, str] | None = None
        repeated_count = 0
        decision_errors = 0
        for _ in range(self.max_steps):
            trace.steps += 1
            try:
                decision = self.decision_model.invoke(self._prompt(trace, context))
            except Exception as exc:
                decision_errors += 1
                trace.observations.append({
                    "status": "decision_error", "error_type": type(exc).__name__,
                    "error_summary": str(exc)[:300],
                })
                if decision_errors >= 2:
                    trace.status = "failed"
                    trace.answer = "The agent could not produce a valid next action."
                    return trace
                continue
            if decision.action == "clarify":
                if not (decision.answer or "").strip():
                    trace.observations.append({
                        "status": "invalid_clarification",
                        "error_summary": "a specific user-facing question is required",
                    })
                    continue
                trace.status = "needs_input"
                trace.answer = decision.answer
                return trace
            if decision.action == "finish":
                if trace.observations and not decision.completion_evidence:
                    trace.observations.append({
                        "status": "invalid_finish",
                        "error_summary": "completion evidence is required after tool use",
                    })
                    continue
                trace.status = "completed"
                trace.answer = (decision.answer or "").strip()
                return trace
            tool = self.tools.get(decision.tool or "")
            if tool is None:
                trace.observations.append({
                    "tool": decision.tool, "status": "permanent_error",
                    "error_summary": "tool is not registered or permitted",
                })
                continue
            signature = (tool.name, json.dumps(decision.arguments, sort_keys=True, default=str))
            repeated_count = repeated_count + 1 if signature == repeated_action else 1
            repeated_action = signature
            if repeated_count > 2:
                trace.observations.append({
                    "tool": tool.name, "status": "permanent_error",
                    "error_summary": "identical action repeated without new evidence",
                })
                continue
            try:
                outcome = tool.handler(decision.arguments)
            except Exception as exc:
                outcome = AgentOutcome(
                    status="needs_specialist", agent=self.name,
                    operation=f"call {tool.name}", error_type=type(exc).__name__,
                    error_summary=str(exc)[:300], verified=False,
                )
            trace.observations.append(self._observation(tool.name, outcome))
        trace.status = "failed"
        trace.answer = "The agent reached its safe step limit before the goal was verified."
        return trace
