from datetime import datetime

from family_ai.curator import CuratorAgent


def test_scheduled_run_skips_when_current_month_completed(tmp_path, monkeypatch):
    curator = CuratorAgent(tmp_path)
    curator._save(curator.report_path, {
        "status": "completed",
        "finished_at": datetime.now().astimezone().isoformat(),
    })
    monkeypatch.setattr(curator, "run", lambda apply: (_ for _ in ()).throw(AssertionError("should not run")))
    assert curator.run_if_due()["status"] == "not_due"


def test_scheduled_run_catches_up_without_report(tmp_path, monkeypatch):
    curator = CuratorAgent(tmp_path)
    monkeypatch.setattr(curator, "run", lambda apply: {"status": "completed", "apply": apply})
    assert curator.run_if_due() == {"status": "completed", "apply": True}
