from family_ai.task_registry import TaskRegistry


def test_task_registry_persists_result_and_failure(tmp_path):
    registry = TaskRegistry(tmp_path / "tasks.sqlite")
    registry.create("one", "codex", "member", "analyze")
    registry.update("one", "completed", result="analysis result")
    row = registry.latest("member", "codex")
    assert row["status"] == "completed"
    assert row["result_summary"] == "analysis result"

    registry.create("two", "codex", "member", "fail")
    registry.update("two", "failed", ValueError("bad task"))
    assert registry.latest("member", "codex")["error_type"] == "ValueError"
    assert registry.get("two", "member")["status"] == "failed"
    assert registry.get("two", "someone-else") is None
