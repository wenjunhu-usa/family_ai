import subprocess
import tempfile
import shutil
from pathlib import Path


class CodexCLI:
    def __init__(self, executable: Path, isolated_workspace: Path, project_workspace: Path):
        self.executable = executable
        self.isolated_workspace = isolated_workspace
        self.project_workspace = project_workspace
        self.isolated_workspace.mkdir(parents=True, exist_ok=True)

    def resolved_executable(self) -> Path | None:
        if self.executable.is_file():
            return self.executable
        candidates = sorted(
            Path("*").glob(
                "openai.chatgpt-*/bin/macos-aarch64/codex"
            ),
            reverse=True,
        )
        return next((path for path in candidates if path.is_file()), None)

    def available(self) -> bool:
        return self.resolved_executable() is not None

    def run(self, task: str, workspace: str = "isolated") -> str:
        executable = self.resolved_executable()
        if executable is None:
            raise RuntimeError("Codex CLI is not installed at the configured path")
        temporary_project = None
        if workspace == "family_project":
            temporary_project = tempfile.TemporaryDirectory()
            cwd = Path(temporary_project.name) / "project"
            shutil.copytree(
                self.project_workspace,
                cwd,
                ignore=shutil.ignore_patterns(
                    ".env", ".git", ".venv", "data", "logs", "__pycache__", "*.db", "*.log"
                ),
            )
        else:
            cwd = self.isolated_workspace
        boundary = (
            "You are a specialist called by a local Family AI supervisor. "
            "Complete only the supplied task. The sandbox is read-only: do not modify files, "
            "send messages, access family memories, or request additional permissions. "
            "Return a self-contained final answer for the user.\n\nTask:\n" + task
        )
        try:
            with tempfile.NamedTemporaryFile(suffix=".txt") as output:
                completed = subprocess.run(
                    [
                        str(executable), "--ask-for-approval", "never", "exec",
                        "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
                        "--output-last-message", output.name, "-C", str(cwd), boundary,
                    ],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                answer = Path(output.name).read_text().strip()
            if completed.returncode != 0:
                detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown error"
                raise RuntimeError(f"Codex CLI failed: {detail}")
            if not answer:
                raise RuntimeError("Codex CLI returned no final answer")
            return answer
        finally:
            if temporary_project is not None:
                temporary_project.cleanup()

    def run_approved_upgrade(self, task: str) -> str:
        """Edit the real project only after an explicit Curator approval."""
        executable = self.resolved_executable()
        if executable is None:
            raise RuntimeError("Codex CLI is not installed at the configured path")
        boundary = (
            "You are performing an explicitly user-approved Family AI upgrade. "
            "Work only inside this repository. Inspect existing architecture first, preserve privacy and "
            "permission boundaries, implement the requested improvement, and run relevant tests. "
            "Do not access family data, credentials, cameras, databases, or other machines. "
            "Do not deploy, restart services, install system software, or delete models. "
            "Return a concise summary of files changed and tests.\n\nApproved task:\n" + task
        )
        with tempfile.NamedTemporaryFile(suffix=".txt") as output:
            completed = subprocess.run(
                [str(executable), "--ask-for-approval", "never", "exec", "--ephemeral",
                 "--sandbox", "workspace-write", "--skip-git-repo-check",
                 "--output-last-message", output.name, "-C", str(self.project_workspace), boundary],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800, check=False,
            )
            answer = Path(output.name).read_text().strip()
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown error"
            raise RuntimeError(f"Codex upgrade failed: {detail}")
        return answer or "Codex completed the approved upgrade."

    def run_workspace_task(self, task: str, workspace: Path) -> str:
        """Run an explicitly requested Codex task in one allowlisted workspace."""
        executable = self.resolved_executable()
        if executable is None:
            raise RuntimeError("Codex CLI is not installed at the configured path")
        workspace = workspace.resolve()
        boundary = (
            "The user explicitly selected Codex for this task. Work only inside the supplied "
            "workspace, follow its AGENTS.md, and complete the requested artifact. Do not access "
            "credentials, family memories, cameras, databases, or other workspaces. Do not publish, "
            "upload, or send messages.\n\nTask:\n" + task
        )
        with tempfile.NamedTemporaryFile(suffix=".txt") as output:
            completed = subprocess.run(
                [str(executable), "--ask-for-approval", "never", "exec", "--ephemeral",
                 "--sandbox", "workspace-write", "--skip-git-repo-check",
                 "--output-last-message", output.name, "-C", str(workspace), boundary],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800, check=False,
            )
            answer = Path(output.name).read_text().strip()
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown error"
            raise RuntimeError(f"Codex workspace task failed: {detail}")
        return answer or "Codex completed the workspace task."
