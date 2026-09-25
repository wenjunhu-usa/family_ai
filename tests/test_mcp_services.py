from family_ai.mcp_services import FamilyMCPServices
from family_ai.storage import FamilyStorage


def test_mcp_services_enforce_existing_member_boundaries(tmp_path):
    storage = FamilyStorage(tmp_path / "family.db")
    service = FamilyMCPServices(storage)
    service.memory_add("member", "member only", "private")
    service.memory_add("member_two", "shared", "family")
    assert [item["content"] for item in service.memory_list("member")] == ["shared", "member only"]
    assert [item["content"] for item in service.memory_list("member_two")] == ["shared"]


def test_mcp_task_tools_reuse_verified_task_agent(tmp_path):
    service = FamilyMCPServices(FamilyStorage(tmp_path / "family.db"))
    created = service.tasks_add("member", "buy milk", "me")
    assert service.tasks_complete("member", created["id"][:8])["completed"] is True
