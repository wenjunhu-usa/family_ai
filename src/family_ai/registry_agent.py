from __future__ import annotations

from .agent_runtime import AgentOutcome, AgentRuntime


class RegistryAgent:
    """Least-privilege lifecycle manager; never self-approves proposals."""

    allowed_permissions = frozenset({
        "network_public", "family_memory_read", "database_read", "database_write",
        "files_read", "files_write", "email", "calendar", "remote_control",
    })

    def __init__(self, storage, runtime: AgentRuntime, diagnose=None):
        self.storage, self.runtime, self.diagnose = storage, runtime, diagnose

    def list(self) -> AgentOutcome:
        return self.runtime.execute(
            "Agent Registry", "list registered agents", self.storage.list_agents,
            diagnose=self.diagnose, verify=lambda value: isinstance(value, list),
        )

    def propose(self, member_id: str, name: str, purpose: str,
                permissions: list[str]) -> AgentOutcome:
        normalized = sorted(set(permissions))
        if not name.strip() or not purpose.strip():
            raise ValueError("Agent name and purpose are required")
        unknown = set(normalized) - self.allowed_permissions
        if unknown:
            raise ValueError(f"Unsupported permissions: {', '.join(sorted(unknown))}")

        def action():
            proposal_id = self.storage.propose_agent(
                member_id, name.strip(), purpose.strip(), normalized
            )
            return {"id": proposal_id, "name": name.strip(),
                    "permissions": normalized, "status": "pending"}

        def verify(value):
            return any(str(item["id"]) == value["id"] and item["status"] == "pending"
                       for item in self.storage.list_agents())

        return self.runtime.execute("Agent Registry", "create pending agent proposal",
                                    action, diagnose=self.diagnose, verify=verify)

    def decide(self, proposal_id: str, approved: bool) -> AgentOutcome:
        expected = "active" if approved else "rejected"

        def action():
            if not self.storage.decide_agent(proposal_id, approved):
                raise ValueError("No unique pending Agent proposal matched")
            return {"id_prefix": proposal_id, "status": expected}

        def verify(_value):
            matches = [item for item in self.storage.list_agents()
                       if str(item["id"]).startswith(proposal_id)]
            return len(matches) == 1 and matches[0]["status"] == expected

        return self.runtime.execute("Agent Registry", f"mark proposal {expected}",
                                    action, diagnose=self.diagnose, verify=verify)
