from pathlib import Path
from unittest.mock import Mock, patch

from family_ai.financial_video import FinancialVideoAgent
from family_ai.planner import ModelPlanner


def planner_without_init():
    return object.__new__(ModelPlanner)


def test_natural_video_request_defaults_to_ollama():
    plan = planner_without_init().plan("制作一个家庭应急基金视频")
    assert plan.tool == "financial_video_create"
    assert plan.provider == "ollama"


def test_explicit_codex_video_request_forces_codex():
    plan = planner_without_init().plan("请用 Codex 制作家庭应急基金视频")
    assert plan.tool == "financial_video_create"
    assert plan.provider == "codex"


def test_natural_codex_override_applies_to_non_video_tasks():
    plan = planner_without_init().plan("请用 Codex 分析这个复杂问题")
    assert plan.tool == "codex_run"
    assert "分析这个复杂问题" in plan.task


def test_codex_specialist_must_create_expected_mp4(tmp_path: Path):
    projects = tmp_path / "projects"
    (projects / "reference-001").mkdir(parents=True)
    codex = Mock()
    agent = FinancialVideoAgent(tmp_path, codex)

    def create_output(task, workspace):
        output = projects / "emergency-fund-002" / "output"
        output.mkdir(parents=True)
        (output / "emergency-fund-002_draft.mp4").write_bytes(b"draft")

    codex.run_workspace_task.side_effect = create_output
    result = agent.create("Use Codex to create an emergency fund video", "codex")
    assert result.slug == "emergency-fund-002"
    assert result.video_path.is_file()
    codex.run_workspace_task.assert_called_once()


def test_ollama_specialist_runs_hidden_pipeline(tmp_path: Path):
    (tmp_path / "projects" / "reference-001").mkdir(parents=True)
    (tmp_path / "agent").mkdir()
    codex = Mock()
    agent = FinancialVideoAgent(tmp_path, codex)

    def fake_run(command, **kwargs):
        slug = "family-budget-002"
        folder = tmp_path / "projects" / slug
        if command[2] == "new":
            (folder / "assets").mkdir(parents=True)
            (folder / "output").mkdir()
            (folder / "brief.md").write_text("", encoding="utf-8")
        elif command[1].endswith("draft_renderer.py"):
            (folder / "output" / f"{slug}_draft.mp4").write_bytes(b"draft")
        return Mock(returncode=0)

    with patch("family_ai.financial_video.subprocess.run", side_effect=fake_run):
        result = agent.create("制作 family budget 视频")
    assert result.provider == "ollama"
    assert result.video_path.is_file()


def test_slug_ignores_reference_project_number(tmp_path: Path):
    (tmp_path / "projects" / "kids-retirement-plan-001").mkdir(parents=True)
    agent = FinancialVideoAgent(tmp_path, Mock())
    assert agent._next_slug("做一个什么是401K的视频，类似001") == "what-is-401k-002"
