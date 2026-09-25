import re
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .codex_cli import CodexCLI


@dataclass(frozen=True)
class FinancialVideoResult:
    provider: str
    slug: str
    project_dir: Path
    video_path: Path


class FinancialVideoAgent:
    """Create review-draft financial videos in the dedicated workspace."""

    name = "Financial Video Agent"

    def __init__(self, workspace: Path, codex: CodexCLI):
        self.workspace = workspace
        self.projects = workspace / "projects"
        self.codex = codex

    def _next_slug(self, topic: str) -> str:
        cleaned = re.sub(r"(?i)(类似|参考|like|similar to)\s*(?:项目\s*)?\d{1,3}", " ", topic)
        words = re.findall(r"[A-Za-z]+|\d+[A-Za-z]+|[A-Za-z]+\d+", cleaned.lower())
        ignored = {"codex", "video", "financial", "create", "make", "please", "draft",
                   "use", "to", "a", "an", "the", "with", "using", "similar", "like"}
        base_words = [word for word in words if word not in ignored][:6]
        if re.search(r"401\s*k", cleaned, re.I) and re.search(r"什么|what|meaning|basics|介绍|了解", cleaned, re.I):
            base = "what-is-401k"
        elif re.search(r"401\s*k", cleaned, re.I):
            base = "401k-education"
        else:
            base = "-".join(base_words) or "financial-education"
        numbers = []
        if self.projects.exists():
            for folder in self.projects.iterdir():
                match = re.search(r"-(\d{3})$", folder.name) if folder.is_dir() else None
                if match:
                    numbers.append(int(match.group(1)))
        return f"{base}-{max(numbers, default=0) + 1:03d}"

    def create(self, request: str, provider: str = "ollama") -> FinancialVideoResult:
        if provider not in {"ollama", "codex"}:
            raise ValueError("provider must be ollama or codex")
        slug = self._next_slug(request)
        folder = self.projects / slug
        if provider == "codex":
            self._create_with_codex(request, slug)
        else:
            self._create_with_ollama(request, slug)
        video = folder / "output" / f"{slug}_draft.mp4"
        if not video.is_file():
            raise RuntimeError(f"Financial Video Agent did not produce {video}")
        return FinancialVideoResult(provider, slug, folder, video)

    def _create_with_ollama(self, request: str, slug: str) -> None:
        agent_dir = Path(__file__).resolve().parent / "agents" / "financial_video"
        cli = agent_dir / "financial_video_agent.py"
        folder = self.projects / slug
        env = {**os.environ, "FINANCIAL_VIDEO_WORKSPACE": str(self.workspace)}
        subprocess.run([sys.executable, str(cli), "new", slug], cwd=self.workspace, env=env, check=True)
        brief = folder / "brief.md"
        brief.write_text(
            "# Video Brief\n\n"
            f"Topic and user request:\n{request}\n\n"
            "Audience: *\n\n"
            "Core message: Create a clear educational draft based only on supported claims.\n\n"
            "Claims or products to avoid: Unsupported rates, guarantees, tax or legal conclusions, and personal recommendations.\n\n"
            "Desired CTA: Invite viewers to learn more and consult an appropriately licensed professional.\n",
            encoding="utf-8",
        )
        subprocess.run([sys.executable, str(cli), "draft", slug], cwd=self.workspace, env=env, check=True)
        renderer = agent_dir / "draft_renderer.py"
        subprocess.run([sys.executable, str(renderer), str(folder)], cwd=self.workspace, env=env, check=True)

    def _create_with_codex(self, request: str, slug: str) -> None:
        task = f"""Create the requested financial education video as project `{slug}`.
User request: {request}

Follow this workspace's AGENTS.md and use projects/kids-retirement-plan-001 as the
workflow and visual-quality reference. Create the complete production package and
render exactly one review MP4 at projects/{slug}/output/{slug}_draft.mp4. Do not use
Remotion, publish anything, or invent financial claims. Complete the video task,
not merely a plan or explanation."""
        self.codex.run_workspace_task(task, self.workspace)
