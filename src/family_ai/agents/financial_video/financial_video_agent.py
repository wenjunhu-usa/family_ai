#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
import re
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(os.environ.get("FINANCIAL_VIDEO_WORKSPACE", "*"))
JOBS = ROOT / "projects"
RENDERER = Path(__file__).resolve().with_name("draft_renderer.py")
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "local-main"


def job(slug: str) -> Path:
    if not slug.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("Project name may contain only letters, numbers, - and _.")
    return JOBS / slug


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ask_model(prompt: str) -> dict:
    scene = {"type": "object", "required": ["start", "duration", "title", "subtitle", "voiceover", "visual"], "properties": {
        "start": {"type": "number"}, "duration": {"type": "number"}, "title": {"type": "string"},
        "subtitle": {"type": "string"}, "voiceover": {"type": "string"}, "visual": {"type": "string"}}, "additionalProperties": False}
    schema = {"type": "object", "required": ["title", "language", "duration_seconds", "audience", "goal", "disclaimer", "cta", "music_prompt", "lyrics", "scenes"],
        "properties": {"title": {"type": "string"}, "language": {"type": "string", "enum": ["zh", "en"]},
        "duration_seconds": {"type": "number"}, "audience": {"type": "string"}, "goal": {"type": "string"},
        "disclaimer": {"type": "string"}, "cta": {"type": "string"}, "music_prompt": {"type": "string"},
        "lyrics": {"type": "string"}, "scenes": {"type": "array", "minItems": 4, "items": scene}}, "additionalProperties": False,
    }
    last_error = None
    for attempt in range(3):
        body = json.dumps({
            "model": MODEL, "stream": False, "format": schema, "think": False,
            "options": {"temperature": 0.25 if attempt else 0.4},
            "messages": [{"role": "system", "content": (
                "You are a production assistant for a *-licensed insurance agent. "
                "The human controls all financial claims and product decisions. Create a warm, clear vertical video production package. "
                "Do not invent rates, guarantees, laws, tax outcomes, product claims, or personal recommendations. Return valid JSON only."
            )}, {"role": "user", "content": prompt}],
        }).encode()
        try:
            request = Request(OLLAMA, data=body, headers={"content-type": "application/json"}, method="POST")
            with urlopen(request, timeout=300) as response:
                result = json.loads(response.read())
            content = str(result.get("message", {}).get("content", "")).strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
            data = json.loads(content)
            if not data.get("scenes"):
                raise ValueError("model returned no scenes")
            return data
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"Local model did not return a valid video package after 3 attempts: {last_error}")


def export_editable_files(folder: Path, data: dict):
    scenes = data.get("scenes", [])
    script = [f"# {data.get('title', '')}\n", f"Audience: {data.get('audience', '')}\n", f"Goal: {data.get('goal', '')}\n"]
    board = [f"# Storyboard: {data.get('title', '')}\n"]
    srt = []
    for index, scene in enumerate(scenes, 1):
        start = float(scene.get("start", 0)); end = start + float(scene.get("duration", 1))
        script += [f"## Scene {index}\n", f"On screen: {scene.get('title', '')} {scene.get('subtitle', '')}\n", f"Voiceover: {scene.get('voiceover', '')}\n"]
        board += [f"## Scene {index} ({start:.1f}s–{end:.1f}s)\n", f"Visual: {scene.get('visual', '')}\n", f"Text: {scene.get('title', '')} / {scene.get('subtitle', '')}\n"]
        def stamp(value):
            ms = int(value * 1000); return f"00:{ms//60000:02d}:{(ms//1000)%60:02d},{ms%1000:03d}"
        subtitle = scene.get("voiceover") or scene.get("subtitle") or scene.get("title", "")
        srt.append(f"{index}\n{stamp(start)} --> {stamp(end)}\n{subtitle}\n")
    script += ["## CTA\n", str(data.get("cta", "")) + "\n", "## Disclaimer\n", str(data.get("disclaimer", "")) + "\n"]
    (folder / "script.md").write_text("\n".join(script), encoding="utf-8")
    (folder / "storyboard.md").write_text("\n".join(board), encoding="utf-8")
    (folder / "subtitles.srt").write_text("\n".join(srt), encoding="utf-8")
    (folder / "music_prompt.md").write_text("# Music Prompt\n\n" + str(data.get("music_prompt", "")) + "\n\n# Lyrics\n\n" + str(data.get("lyrics", "")) + "\n", encoding="utf-8")


def cmd_new(args):
    folder = job(args.slug)
    (folder / "assets").mkdir(parents=True, exist_ok=False)
    (folder / "output").mkdir()
    (folder / "brief.md").write_text(
        "# Video Brief\n\nTopic:\n\nAudience: *\n\nCore message:\n\nPoints that must be included:\n\nClaims or products to avoid:\n\nDesired CTA:\n",
        encoding="utf-8",
    )
    print(f"Created {folder}\nEdit brief.md, then run: draft {args.slug}")


def cmd_draft(args):
    folder = job(args.slug); brief = (folder / "brief.md").read_text(encoding="utf-8")
    reference = JOBS / "kids-retirement-plan-001"
    examples = []
    for name in ("script.md", "storyboard.md", "canva_pages.md"):
        path = reference / name
        if path.is_file():
            examples.append(f"REFERENCE {name}:\n{path.read_text(encoding='utf-8')[:5000]}")
    data = ask_model(
        "Build a 9:16 short-video package from this human brief. Follow the reference's pacing, warmth, visual hierarchy, and production detail, but do not copy its claims or wording.\n\n"
        + brief + "\n\n" + "\n\n".join(examples)
    )
    (folder / "project.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    export_editable_files(folder, data)
    (folder / "approval.json").unlink(missing_ok=True)
    print(f"Draft generated in {folder}. Review project.json and the Markdown files.")


def cmd_revise(args):
    folder = job(args.slug); current = (folder / "project.json").read_text(encoding="utf-8")
    data = ask_model("Revise this existing video package. Preserve correct human-controlled claims.\n\nCURRENT:\n" + current + "\n\nINSTRUCTION:\n" + args.instruction)
    (folder / "project.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    export_editable_files(folder, data); (folder / "approval.json").unlink(missing_ok=True)
    print("Revision complete. Human approval was reset.")


def cmd_approve(args):
    folder = job(args.slug); project = folder / "project.json"
    record = {"approved_sha256": digest(project), "approved_by": args.reviewer, "note": args.note}
    (folder / "approval.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Approved. Add assets/music.mp3 if desired, then render the draft MP4.")


def cmd_render(args):
    folder = job(args.slug); project = folder / "project.json"; approval = folder / "approval.json"
    if not approval.exists() or json.loads(approval.read_text())["approved_sha256"] != digest(project):
        raise SystemExit("Current project.json is not approved. Run approve after your final edit.")
    subprocess.run([sys.executable, str(RENDERER), str(folder)], check=True)


def cmd_status(args):
    folder = job(args.slug); project = folder / "project.json"; approval = folder / "approval.json"
    approved = project.exists() and approval.exists() and json.loads(approval.read_text()).get("approved_sha256") == digest(project)
    print(json.dumps({"project": args.slug, "draft": project.exists(), "approved": approved,
                      "music": (folder / "assets/music.mp3").exists(),
                      "video": (folder / "output" / f"{args.slug}_draft.mp4").exists()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Human-approved Financial Video Agent")
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("new"); p.add_argument("slug"); p.set_defaults(func=cmd_new)
    p = commands.add_parser("draft"); p.add_argument("slug"); p.set_defaults(func=cmd_draft)
    p = commands.add_parser("revise"); p.add_argument("slug"); p.add_argument("instruction"); p.set_defaults(func=cmd_revise)
    p = commands.add_parser("approve"); p.add_argument("slug"); p.add_argument("--reviewer", default="*"); p.add_argument("--note", default="Human reviewed content and claims."); p.set_defaults(func=cmd_approve)
    p = commands.add_parser("render"); p.add_argument("slug"); p.set_defaults(func=cmd_render)
    p = commands.add_parser("status"); p.add_argument("slug"); p.set_defaults(func=cmd_status)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
