"""Conservative monthly model and capability curator for Family AI."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .config import settings


TRUSTED_PUBLISHERS = {"*"}
PROTECTED_MODELS = {"local-main:latest", "local-main", "local-candidate:latest", "local-candidate", "local-rollback:latest", "local-rollback"}
MODEL_API = "https://huggingface.co/api/models"
TOOL_QUERIES = (
    "LangGraph release local agent tools GitHub",
    "Ollama release structured output tools",
    "local speech recognition open source",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_params_billions(model_id: str) -> float | None:
    hits = re.findall(r"(?i)(\d+(?:\.\d+)?)\s*[bB](?:-|_|\b)", model_id)
    return float(hits[-1]) if hits else None


class CuratorAgent:
    def __init__(self, data_dir: Path | None = None):
        self.root = (data_dir or settings.family_ai_data_dir) / "curator"
        self.root.mkdir(parents=True, exist_ok=True)
        self.report_path = self.root / "latest-report.json"
        self.state_path = self.root / "state.json"
        self.ollama = Path("/Applications/Ollama.app/Contents/Resources/ollama")
        self.env = os.environ.copy()
        self.env.setdefault("OLLAMA_HOST", settings.ollama_base_url)

    def _load(self, path: Path, default):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return default

    def _save(self, path: Path, value) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(path)

    def status(self) -> dict:
        state = self._load(self.state_path, {})
        return {
            "agent": "Model & Capability Curator Agent",
            "schedule": "monthly",
            "codex_policy": "only proposed for complex installation or migration; never called by monthly audit",
            "policy": {
                "parameter_range_b": [24, 64],
                "maximum_download_gib": 30,
                "trusted_publishers": sorted(TRUSTED_PUBLISHERS),
                "rollback_alias": "local-rollback",
                "retention_days": 30,
            },
            "latest_report": self._load(self.report_path, None),
            "pending_proposals": [x for x in state.get("proposals", []) if x.get("status") == "pending"],
        }

    def create_capability_proposal(self, state: dict, research: list[dict]) -> dict | None:
        official_hosts = ("github.com/", "docs.ollama.com/", "ollama.com/blog/", "langchain.com/")
        useful = [x for x in research if x.get("url") and x.get("title") and any(host in x["url"] for host in official_hosts)]
        if not useful or any(x.get("status") == "pending" for x in state.get("proposals", [])):
            return None
        top = useful[:5]
        proposal = {
            "id": str(uuid4()), "status": "pending", "created_at": utc_now(),
            "title": "Review current Family AI capability upgrades",
            "task": "Review these current tool/release findings, choose only a material and compatible improvement, implement it in Family AI, and test it:\n" + "\n".join(f"- {x['title']}: {x['url']}" for x in top),
        }
        state.setdefault("proposals", []).append(proposal)
        return proposal

    def approve_proposal(self, proposal_id: str) -> dict | None:
        state = self._load(self.state_path, {})
        matches = [x for x in state.get("proposals", []) if x.get("id", "").startswith(proposal_id) and x.get("status") == "pending"]
        if len(matches) != 1:
            return None
        matches[0]["status"] = "approved"
        matches[0]["approved_at"] = utc_now()
        self._save(self.state_path, state)
        return matches[0]

    def deny_proposal(self, proposal_id: str) -> bool:
        state = self._load(self.state_path, {})
        matches = [x for x in state.get("proposals", []) if x.get("id", "").startswith(proposal_id) and x.get("status") == "pending"]
        if len(matches) != 1:
            return False
        matches[0]["status"] = "denied"
        matches[0]["denied_at"] = utc_now()
        self._save(self.state_path, state)
        return True

    def _get_json(self, url: str):
        request = Request(url, headers={"User-Agent": "FamilyAI-Curator/1.0"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    def discover_models(self) -> list[dict]:
        query = urlencode({"filter": "gguf", "sort": "trendingScore", "direction": "-1", "limit": 100, "full": "true"})
        rows = self._get_json(f"{MODEL_API}?{query}")
        candidates = []
        for row in rows:
            model_id = row.get("id", "")
            if re.search(r"(?i)(?:^|[-_/])(MTP|BASE)(?:[-_/]|$)", model_id):
                continue
            publisher = model_id.split("/", 1)[0].casefold()
            params = model_params_billions(model_id)
            if publisher not in TRUSTED_PUBLISHERS or params is None or not 24 <= params <= 64:
                continue
            siblings = row.get("siblings") or []
            ggufs = []
            for item in siblings:
                name = item.get("rfilename", "")
                size = item.get("size") or (item.get("lfs") or {}).get("size")
                if name.lower().endswith(".gguf") and re.search(r"(?i)(UD-)?Q4_K_M|Q4_0", name):
                    ggufs.append({"file": name, "size": size})
            if not ggufs:
                continue
            ggufs.sort(key=lambda x: ("Q4_K_M" not in x["file"].upper(), x["size"] or 10**15))
            chosen = ggufs[0]
            if not chosen["size"]:
                try:
                    url = f"https://huggingface.co/{model_id}/resolve/main/{quote(chosen['file'])}"
                    request = Request(url, method="HEAD", headers={"User-Agent": "FamilyAI-Curator/1.0"})
                    with urlopen(request, timeout=30) as response:
                        chosen["size"] = int(response.headers.get("x-linked-size") or response.headers.get("content-length") or 0)
                except Exception:
                    chosen["size"] = 0
            if not chosen["size"] or chosen["size"] > 30 * 1024**3:
                continue
            quant = Path(chosen["file"]).stem.split("-")[-1]
            if "UD-Q4_K_M" in chosen["file"].upper():
                quant = "UD-Q4_K_M"
            elif "Q4_K_M" in chosen["file"].upper():
                quant = "Q4_K_M"
            elif "Q4_0" in chosen["file"].upper():
                quant = "Q4_0"
            candidates.append({
                "id": model_id, "params_b": params, "quant": quant,
                "ollama_ref": f"hf.co/{model_id}:{quant}",
                "file": chosen["file"], "size_bytes": chosen["size"],
                "downloads": row.get("downloads", 0), "likes": row.get("likes", 0),
                "last_modified": row.get("lastModified"),
            })
        return candidates[:12]

    def discover_tools(self) -> list[dict]:
        try:
            from ddgs import DDGS
            found = []
            for query in TOOL_QUERIES:
                for row in DDGS().text(query, max_results=3):
                    found.append({"query": query, "title": row.get("title"), "url": row.get("href"), "summary": row.get("body")})
            return found
        except Exception as exc:
            return [{"error": type(exc).__name__, "message": str(exc)[:200]}]

    def _ollama(self, *args: str, timeout: int = 3600) -> str:
        result = subprocess.run([str(self.ollama), *args], env=self.env, text=True, capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip())
        return result.stdout

    def installed_models(self) -> list[str]:
        data = self._get_json(settings.ollama_base_url.rstrip("/") + "/api/tags")
        return [row["name"] for row in data.get("models", [])]

    def current_source(self) -> str | None:
        try:
            text = self._ollama("show", "local-main", "--modelfile", timeout=60)
            match = re.search(r"(?m)^FROM\s+(.+)$", text)
            return match.group(1).strip() if match else None
        except Exception:
            return None

    def choose_candidate(self, models: list[dict], state: dict) -> dict | None:
        installed = {x.casefold() for x in self.installed_models()}
        evaluated = {x.casefold() for x in state.get("evaluated", [])}
        available = [m for m in models if m["ollama_ref"].casefold() not in installed | evaluated]
        if not available:
            return None
        available.sort(key=lambda m: (m.get("likes", 0), m.get("downloads", 0)), reverse=True)
        return available[0]

    def _chat_json(self, model: str, prompt: str) -> tuple[dict, float]:
        payload = json.dumps({"model": model, "stream": False, "format": "json", "messages": [{"role": "user", "content": prompt}], "options": {"temperature": 0, "num_predict": 160}}).encode()
        request = Request(settings.ollama_base_url.rstrip("/") + "/api/chat", data=payload, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
        content = json.loads(result["message"]["content"])
        speed = result.get("eval_count", 0) / max(result.get("eval_duration", 1) / 1e9, .001)
        return content, speed

    def benchmark(self, model: str) -> dict:
        cases = [
            ("Return JSON only: user asks '未来3天*天气'. Keys: tool, location, days.", lambda x: x.get("tool") == "weather" and str(x.get("location", "")) == "*" and x.get("days") == 3),
            ("Return JSON only: route 'Find four parks in *'. Keys: tool, query, count.", lambda x: x.get("tool") == "web_search" and x.get("count") == 4),
            ("Return JSON only with key answer: calculate 17*23.", lambda x: x.get("answer") == 391),
        ]
        passed, speeds, details = 0, [], []
        for prompt, check in cases:
            try:
                value, speed = self._chat_json(model, prompt)
                ok = bool(check(value)); passed += int(ok); speeds.append(speed)
                details.append({"passed": ok, "response": value})
            except Exception as exc:
                details.append({"passed": False, "error": type(exc).__name__})
        return {"score": passed / len(cases), "tokens_per_second": round(sum(speeds) / len(speeds), 2) if speeds else 0, "cases": details}

    def install_and_test(self, candidate: dict, state: dict) -> dict:
        self._ollama("pull", candidate["ollama_ref"])
        with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", dir=self.root, delete=False) as file:
            file.write(f"FROM {candidate['ollama_ref']}\nPARAMETER num_ctx 32768\nPARAMETER temperature 0.6\n")
            modelfile = file.name
        try:
            self._ollama("create", "local-candidate", "-f", modelfile)
        finally:
            Path(modelfile).unlink(missing_ok=True)
        current = self.benchmark("local-main")
        challenger = self.benchmark("local-candidate")
        # Speed alone is not an upgrade. The challenger must demonstrate a real
        # quality gain and remain usable on this machine.
        accepted = (
            challenger["score"] > current["score"]
            and challenger["score"] >= .8
            and challenger["tokens_per_second"] >= current["tokens_per_second"] * .5
        )
        outcome = {"candidate": candidate, "current_benchmark": current, "candidate_benchmark": challenger, "accepted": accepted}
        state.setdefault("evaluated", []).append(candidate["ollama_ref"])
        if not accepted:
            return outcome
        previous_source = self.current_source()
        self._ollama("create", "local-rollback", "-f", self._write_modelfile("FROM local-main\n"))
        self._ollama("create", "local-main", "-f", self._write_modelfile("FROM local-candidate\n"))
        state.setdefault("retired", []).append({"model": previous_source, "retired_at": utc_now()})
        state["current_source"] = candidate["ollama_ref"]
        outcome["switched"] = True
        return outcome

    def _write_modelfile(self, content: str) -> str:
        path = self.root / "switch.Modelfile"
        path.write_text(content)
        return str(path)

    def cleanup(self, state: dict) -> list[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        installed = set(self.installed_models())
        deleted = []
        for item in state.get("retired", []):
            name = item.get("model")
            try:
                old = datetime.fromisoformat(item["retired_at"]) < cutoff
            except Exception:
                old = False
            if name and old and name in installed and name not in PROTECTED_MODELS and name != state.get("current_source"):
                self._ollama("rm", name, timeout=300)
                deleted.append(name)
        state["retired"] = [x for x in state.get("retired", []) if x.get("model") not in deleted]
        return deleted

    def run(self, apply: bool = False) -> dict:
        started = time.monotonic()
        state = self._load(self.state_path, {"evaluated": [], "retired": []})
        report = {"started_at": utc_now(), "mode": "monthly" if apply else "audit", "hardware": {"machine": platform.machine(), "memory_gib": 48}, "codex_called": False}
        try:
            models = self.discover_models()
            report["model_candidates"] = models
            candidate = self.choose_candidate(models, state)
            report["selected_candidate"] = candidate
            report["capability_research"] = self.discover_tools()
            report["capability_proposal"] = self.create_capability_proposal(state, report["capability_research"])
            report["capability_policy"] = "Research only. Complex installation/migration becomes a Main Agent Codex proposal; nothing is installed automatically."
            if apply and candidate:
                report["upgrade"] = self.install_and_test(candidate, state)
                report["deleted_models"] = self.cleanup(state)
            else:
                report["upgrade"] = {"performed": False, "reason": "audit only" if not apply else "no unevaluated qualifying candidate"}
            report["status"] = "completed"
        except Exception as exc:
            report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        report["finished_at"] = utc_now()
        report["duration_seconds"] = round(time.monotonic() - started, 2)
        self._save(self.report_path, report)
        self._save(self.state_path, state)
        return report

    def run_if_due(self) -> dict:
        """Run once per calendar month, including after a powered-off missed date."""
        report = self._load(self.report_path, {})
        now = datetime.now().astimezone()
        try:
            finished = datetime.fromisoformat(report.get("finished_at", "")).astimezone()
            already_ran_this_month = (
                report.get("status") == "completed"
                and finished.year == now.year
                and finished.month == now.month
            )
        except (TypeError, ValueError):
            already_ran_this_month = False
        if already_ran_this_month:
            return {
                "status": "not_due",
                "checked_at": utc_now(),
                "reason": "monthly audit already completed this calendar month",
                "last_completed_at": report.get("finished_at"),
            }
        return self.run(apply=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "audit", "monthly", "scheduled"))
    args = parser.parse_args()
    agent = CuratorAgent()
    if args.command == "status":
        result = agent.status()
    elif args.command == "scheduled":
        result = agent.run_if_due()
    else:
        result = agent.run(apply=args.command == "monthly")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
