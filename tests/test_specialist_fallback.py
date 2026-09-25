from family_ai.graph import propose_specialist_repair
from family_ai.storage import FamilyStorage


def test_specialist_failure_creates_sanitized_codex_approval(tmp_path):
    storage = FamilyStorage(tmp_path / "family.db")
    reply = propose_specialist_repair(
        storage, "member", "Gmail Backup Agent", "backup", ValueError("private message body")
    )
    requests = storage.list_codex_requests("member")
    assert len(requests) == 1
    assert requests[0]["status"] == "pending"
    assert requests[0]["workspace"] == "family_project"
    assert "private message body" not in requests[0]["task"]
    assert "ValueError" in requests[0]["task"]
    assert requests[0]["id"][:8] in reply
