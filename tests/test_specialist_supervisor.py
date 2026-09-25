import json
from unittest.mock import Mock

from family_ai.specialist_supervisor import SpecialistSupervisor
from family_ai.storage import FamilyStorage


def test_permissionless_specialist_is_activated_and_answers(tmp_path):
    storage = FamilyStorage(tmp_path / "family.db")
    codex = Mock()
    codex.run.return_value = json.dumps({"name": "Writing Specialist Agent",
        "purpose": "Create structured drafts", "permissions": [],
        "system_prompt": "Write clearly.", "final_answer": "Draft completed."})
    supervisor = SpecialistSupervisor(codex, tmp_path / "agents", storage)
    result = supervisor.create_and_execute("Draft a speech")
    assert result.created and result.answer == "Draft completed."
    assert any(x["name"] == result.name and x["status"] == "active" for x in storage.list_agents())
    assert result.name in {item["name"] for item in supervisor.active_manifests()}


def test_specialist_needing_permissions_stays_pending(tmp_path):
    storage = FamilyStorage(tmp_path / "family.db")
    codex = Mock()
    codex.run.return_value = json.dumps({"name": "Email Specialist Agent",
        "purpose": "Send email", "permissions": ["email"],
        "system_prompt": "Send approved mail only.", "final_answer": ""})
    result = SpecialistSupervisor(codex, tmp_path / "agents", storage).create_and_execute("Email recipient")
    assert not result.created and "email" in result.answer
    assert any(x["name"] == result.name and x["status"] == "pending" for x in storage.list_agents())
