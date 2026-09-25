from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from .agent_runtime import AgentOutcome


class DiagnosticsAgent:
    permissions = frozenset({"database_read", "files_read"})

    def __init__(self, storage, backup_root: Path, ollama_url: str,
                 gmail_worker, gmail_scheduler, task_registry):
        self.storage = storage
        self.backup_root = backup_root
        self.ollama_url = ollama_url.rstrip("/")
        self.gmail_worker = gmail_worker
        self.gmail_scheduler = gmail_scheduler
        self.task_registry = task_registry

    @staticmethod
    def _safe(name, action):
        try:
            return {"status": "healthy", "details": action()}
        except Exception as exc:
            return {"status": "unhealthy", "error_type": type(exc).__name__,
                    "error_summary": str(exc)[:240]}

    def inspect(self, target: str, member_id: str) -> AgentOutcome:
        checks = {}
        if target in {"all", "database"}:
            checks["database"] = self._safe("database", self.storage.healthcheck)
        if target in {"all", "drive"}:
            volume = self.backup_root.parent.parent
            checks["drive"] = {
                "status": "healthy" if volume.is_dir() else "unhealthy",
                "mounted": volume.is_dir(), "volume": str(volume),
            }
        if target in {"all", "gmail"}:
            checks["gmail"] = self._safe("gmail", lambda: {
                "worker_alive": self.gmail_worker.is_alive(),
                "scheduler_alive": bool(
                    self.gmail_scheduler._thread and self.gmail_scheduler._thread.is_alive()
                ),
                "mode": "daily_postgresql_and_on_demand_disk_export",
                "accounts": self.gmail_worker.agent.accounts(member_id),
            })
        if target in {"all", "ollama"}:
            def ollama():
                with urllib.request.urlopen(self.ollama_url + "/api/tags", timeout=3) as response:
                    payload = json.loads(response.read())
                return {"reachable": True, "models": len(payload.get("models", []))}
            checks["ollama"] = self._safe("ollama", ollama)
        if target in {"all", "codex"}:
            checks["codex"] = {
                "status": "healthy",
                "running": self.task_registry.active_count("codex") > 0,
                "mode": "on_demand_after_approval",
                "latest_task": self.task_registry.latest(member_id, "codex"),
            }
        return AgentOutcome(
            status="success", value={"target": target, "checks": checks},
            operation=f"inspect {target}", agent="Diagnostics Agent",
            verified=bool(checks),
        )
