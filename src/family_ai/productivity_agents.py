from __future__ import annotations

from .agent_runtime import AgentOutcome, AgentRuntime


class MemoryAgent:
    permissions = frozenset({"family_memory_read", "database_write"})

    def __init__(self, storage, runtime: AgentRuntime, diagnose=None):
        self.storage, self.runtime, self.diagnose = storage, runtime, diagnose

    def add(self, member_id: str, content: str, scope: str) -> AgentOutcome:
        content = content.strip()
        if not content or scope not in {"private", "family"}:
            raise ValueError("Memory content and scope must be valid")

        def action():
            memory_id = self.storage.add_memory(member_id, content, scope)
            return {"id": memory_id, "content": content, "scope": scope}

        def verify(value):
            return any(
                str(item["id"]) == value["id"] and item["content"] == content
                for item in self.storage.list_memories(member_id)
            )

        return self.runtime.execute("Memory Agent", "save memory", action,
                                    diagnose=self.diagnose, verify=verify)

    def list(self, member_id: str) -> AgentOutcome:
        return self.runtime.execute(
            "Memory Agent", "list authorized memories",
            lambda: self.storage.list_memories(member_id), diagnose=self.diagnose,
            verify=lambda value: isinstance(value, list),
        )


class TasksAgent:
    permissions = frozenset({"database_read", "database_write"})

    def __init__(self, storage, runtime: AgentRuntime, diagnose=None):
        self.storage, self.runtime, self.diagnose = storage, runtime, diagnose

    def add(self, member_id: str, content: str, owner: str) -> AgentOutcome:
        content = content.strip()
        owner_id = member_id if owner == "me" else "family"
        if not content or owner not in {"me", "family"}:
            raise ValueError("Task content and owner must be valid")

        def action():
            task_id = self.storage.add_todo(owner_id, content)
            return {"id": task_id, "owner_id": owner_id, "content": content}

        def verify(value):
            return any(str(item["id"]) == value["id"] and not item["completed"]
                       for item in self.storage.list_todos(member_id))

        return self.runtime.execute("Tasks Agent", "add task", action,
                                    diagnose=self.diagnose, verify=verify)

    def list(self, member_id: str) -> AgentOutcome:
        return self.runtime.execute(
            "Tasks Agent", "list tasks", lambda: self.storage.list_todos(member_id),
            diagnose=self.diagnose, verify=lambda value: isinstance(value, list),
        )

    def complete(self, member_id: str, task_id: str) -> AgentOutcome:
        task_id = task_id.strip()
        if not task_id:
            raise ValueError("Task ID is required")

        def action():
            if not self.storage.complete_todo(member_id, task_id):
                raise ValueError("No matching authorized task was found")
            return {"id_prefix": task_id, "completed": True}

        def verify(_value):
            matches = [item for item in self.storage.list_todos(member_id)
                       if str(item["id"]).startswith(task_id)]
            return len(matches) == 1 and bool(matches[0]["completed"])

        return self.runtime.execute("Tasks Agent", "complete task", action,
                                    diagnose=self.diagnose, verify=verify)
