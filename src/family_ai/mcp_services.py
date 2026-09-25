from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, is_dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from .agent_runtime import AgentOutcome, AgentRuntime
from .calendar_agent import CalendarAgent
from .camera_agent import CameraAgent
from .codex_cli import CodexCLI
from .config import settings
from .financial_video import FinancialVideoAgent
from .gmail_backup import GmailBackupAgent, GmailBackupWorker
from .graph import build_agent
from .productivity_agents import MemoryAgent, TasksAgent
from .rag import RagStore
from .storage import FamilyStorage, PostgresFamilyStorage
from .weather import get_forecast_range


def json_value(value: Any) -> Any:
    if isinstance(value, AgentOutcome):
        if value.status != "success":
            raise RuntimeError(value.error_summary or f"{value.agent} failed: {value.status}")
        return json_value(value.value)
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


class FamilyMCPServices:
    """Protocol-independent adapters over the existing Family AI business logic."""

    def __init__(self, storage=None):
        self.storage = storage or (
            PostgresFamilyStorage(settings.database_url, lazy=True) if settings.database_url
            else FamilyStorage(settings.family_ai_data_dir / "family.db")
        )
        self.runtime = AgentRuntime(max_attempts=3, base_delay=.3)
        self.memory = MemoryAgent(self.storage, self.runtime)
        self.tasks = TasksAgent(self.storage, self.runtime)

    @cached_property
    def gmail(self) -> GmailBackupAgent:
        return GmailBackupAgent(
            settings.google_oauth_client_id, settings.google_oauth_client_secret,
            settings.google_oauth_redirect_uri, settings.gmail_backup_root,
            secondary_archive=self.storage if settings.database_url else None,
            require_secondary=bool(settings.database_url), account_registry=self.storage,
        )

    @cached_property
    def gmail_worker(self) -> GmailBackupWorker:
        return GmailBackupWorker(self.gmail, self.storage)

    @cached_property
    def calendar(self) -> CalendarAgent:
        return CalendarAgent(self.gmail)

    @cached_property
    def family_agent(self):
        return build_agent()

    def system_status(self) -> dict:
        models = []
        try:
            with urllib.request.urlopen(settings.ollama_base_url.rstrip("/") + "/api/ps", timeout=3) as response:
                models = [item.get("name") for item in json.load(response).get("models", [])]
            ollama = "online"
        except Exception:
            ollama = "offline"
        try:
            database = self.storage.healthcheck()
        except Exception as exc:
            database = {"reachable": False, "error": type(exc).__name__}
        return {"status": "online", "ollama": ollama, "loaded_models": models,
                "configured_model": settings.ollama_model, "database": database}

    def agents_list(self) -> list[dict]:
        return json_value(self.storage.list_agents())

    def tasks_list(self, member_id: str):
        return json_value(self.tasks.list(member_id))

    def tasks_add(self, member_id: str, content: str, owner: str):
        return json_value(self.tasks.add(member_id, content, owner))

    def tasks_complete(self, member_id: str, task_id: str):
        return json_value(self.tasks.complete(member_id, task_id))

    def memory_list(self, member_id: str):
        return json_value(self.memory.list(member_id))

    def memory_add(self, member_id: str, content: str, scope: str):
        return json_value(self.memory.add(member_id, content, scope))

    def weather(self, location: str, days: int, language: str):
        return get_forecast_range(location, days=max(1, min(days, 7)), language=language)

    def calendar_upcoming(self, member_id: str, account: str | None, days: int):
        return self.calendar.upcoming(member_id, account, days)

    def gmail_status(self, member_id: str, account: str | None):
        return self.gmail.status(member_id, account)

    def gmail_backup_start(self, member_id: str, account: str | None):
        accounts = self.gmail.accounts(member_id)
        selected = accounts if account == "*" else [self.gmail._select_account(member_id, account)]
        return {"job_id": self.gmail_worker.submit(member_id, selected), "accounts": selected}

    def gmail_backup_status(self, member_id: str):
        return self.gmail_worker.status(member_id)

    def knowledge_search(self, member_id: str, query: str, limit: int):
        if not settings.database_url:
            raise RuntimeError("Knowledge search requires PostgreSQL")
        store = RagStore(settings.database_url, settings.ollama_base_url, settings.rag_embedding_model)
        return json_value(store.search(member_id, query, max(1, min(limit, 10))))

    def assistant_chat(self, member_id: str, message: str, thread_id: str):
        return {"reply": self.family_agent.invoke(member_id, thread_id, message)}

    def camera_inspect(self, provider: str, question: str, language: str):
        camera = CameraAgent(settings.camera_capture_helper, settings.ollama_base_url,
                             settings.ollama_model)
        return camera.inspect_current(provider, question, language)

    def video_create(self, request: str):
        codex = CodexCLI(settings.codex_cli_path, settings.family_ai_data_dir / "codex-workspace",
                         settings.family_ai_project_dir)
        return json_value(FinancialVideoAgent(settings.financial_video_workspace, codex).create(request))
