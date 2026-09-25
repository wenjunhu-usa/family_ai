import json
import re
from dataclasses import dataclass
from pathlib import Path

from .codex_cli import CodexCLI


@dataclass(frozen=True)
class SpecialistOutcome:
    name: str
    created: bool
    answer: str
    permissions: list[str]
    manifest_path: Path


class SpecialistSupervisor:
    """Have Codex define a missing least-privilege specialist and finish its task."""

    def __init__(self, codex: CodexCLI, manifest_dir: Path, storage):
        self.codex = codex
        self.manifest_dir = manifest_dir
        self.storage = storage
        manifest_dir.mkdir(parents=True, exist_ok=True)

    def active_manifests(self) -> list[dict]:
        active_names = {item["name"] for item in self.storage.list_agents() if item["status"] == "active"}
        manifests = []
        for path in self.manifest_dir.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if item.get("name") in active_names:
                manifests.append(item)
        return manifests

    @staticmethod
    def _parse_json(text: str) -> dict:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S | re.I)
        candidate = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
        if not candidate:
            raise RuntimeError("Codex did not return a specialist manifest")
        return json.loads(candidate)

    def create_and_execute(self, task: str) -> SpecialistOutcome:
        prompt = f"""No installed Family AI specialist can complete this task. Define the
smallest reusable specialist and complete the original task now. Return JSON only:
{{"name":"... Agent","purpose":"...","permissions":[],"system_prompt":"...","final_answer":"..."}}

Allowed permissions: network_public, family_memory_read, database_read,
database_write, files_read, files_write, email, calendar, remote_control.
Request only permissions actually required. Never claim an unavailable external
action was completed. Original task: {task}"""
        data = self._parse_json(self.codex.run(prompt, "isolated"))
        name = str(data["name"]).strip()
        permissions = [str(item) for item in data.get("permissions", [])]
        safe_name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "specialist"
        path = self.manifest_dir / f"{safe_name}.json"
        manifest = {"name": name, "purpose": str(data.get("purpose", "")),
                    "permissions": permissions, "system_prompt": str(data.get("system_prompt", ""))}
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        existing = next((item for item in self.storage.list_agents() if item["name"] == name), None)
        if existing:
            proposal_id = existing["id"]
            created = existing["status"] == "active"
        else:
            proposal_id = self.storage.propose_agent("main-agent", name, manifest["purpose"], permissions)
            created = not permissions
            if created:
                self.storage.decide_agent(proposal_id, True)
        answer = str(data.get("final_answer", "")).strip()
        if permissions:
            answer = (f"Codex 已生成 {name}，但它需要新权限：{', '.join(permissions)}。"
                      f"请先批准 Agent 提案 {proposal_id[:8]}，之后重新提交原任务。")
        return SpecialistOutcome(name, created, answer, permissions, path)
