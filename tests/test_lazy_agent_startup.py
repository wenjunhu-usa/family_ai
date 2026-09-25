from pathlib import Path
import asyncio

import family_ai.graph as graph_module
import family_ai.app as app_module


def test_build_agent_does_not_connect_database_or_initialize_rag(monkeypatch, tmp_path: Path):
    class ForbiddenRag:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("RAG must be lazy")

    monkeypatch.setattr(graph_module, "RagStore", ForbiddenRag)
    monkeypatch.setattr(graph_module.settings, "database_url",
                        "postgresql://invalid:invalid@127.0.0.1:1/family_ai")
    monkeypatch.setattr(graph_module.settings, "family_ai_data_dir", tmp_path)
    family_agent = graph_module.build_agent()
    assert family_agent.gmail_backup_worker is not None
    assert not family_agent.gmail_backup_worker.is_alive()


def test_approval_poll_does_not_initialize_main_agent(monkeypatch):
    class ApprovalStorage:
        def list_agents(self):
            return []

        def list_codex_requests(self, _member_id, _status):
            return []

    monkeypatch.setattr(app_module, "approval_storage", lambda: ApprovalStorage())
    monkeypatch.setattr(app_module, "agent", lambda: (_ for _ in ()).throw(
        AssertionError("approval polling must not initialize the main agent")
    ))
    result = app_module.pending_approvals("audit-member")
    assert result["items"] == []
    assert result["stale"] is False


def test_database_connection_timeout_is_retryable():
    ConnectionTimeout = type("ConnectionTimeout", (Exception,), {})
    assert app_module.database_connection_failed(ConnectionTimeout("slow"))


def test_startup_skips_gmail_scheduler_when_storage_is_unavailable(monkeypatch):
    class UnavailableStorage:
        def healthcheck(self):
            raise TimeoutError("database is offline")

    class Scheduler:
        started = False

        def start(self):
            self.started = True

    scheduler = Scheduler()
    current = type("Agent", (), {
        "storage": UnavailableStorage(),
        "gmail_daily_scheduler": scheduler,
    })()
    monkeypatch.setattr(app_module, "agent", lambda: current)
    monkeypatch.setattr(app_module.settings, "camera_monitor_enabled", False)
    asyncio.run(app_module.start_requested_background_services())
    assert scheduler.started is False
    assert app_module._database_reachable is False


def test_system_status_distinguishes_idle_agents_and_offline_postgres(monkeypatch):
    monkeypatch.setattr(app_module.settings, "database_url", "postgresql://offline")
    monkeypatch.setattr(app_module, "_database_reachable", False)
    app_module.agent.cache_clear()
    result = app_module.system_status()
    assert result["database"] == "postgresql-offline"
    assert result["rag"]["ready"] is False
    assert {item["status"] for item in result["agents"]} == {"idle"}
