#!/usr/bin/env python3
"""Render a simple vertical draft MP4 from a project's project.json."""

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
BG = "#FAF8F3"
GREEN = "#2F5D50"
GOLD = "#D8B46A"
MUTED = "#6F8B7F"


def font(size: int, bold: bool = False):
    candidates = [
        (Path("/System/Library/Fonts/Hiragino Sans GB.ttc"), 2 if bold else 0),
        (Path("/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc"), 0),
        (Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"), 0),
    ]
    for path, index in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size, index=index)
    return ImageFont.load_default()


def centered(draw, y, value, color, size, bold=False, spacing=18):
    lines = str(value or "").splitlines() or [""]
    face = font(size, bold)
    boxes = [draw.textbbox((0, 0), line, font=face) for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    top = y - (sum(heights) + spacing * (len(lines) - 1)) / 2
    for line, box, height in zip(lines, boxes, heights):
        draw.text(((W - (box[2] - box[0])) / 2, top), line, fill=color, font=face)
        top += height + spacing


def render(project_folder: Path):
    project_path = project_folder / "project.json"
    data = json.loads(project_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes") or []
    if not scenes:
        raise SystemExit(f"No scenes found in {project_path}")

    output = project_folder / "output"
    frames = output / "draft_frames"
    output.mkdir(exist_ok=True)
    frames.mkdir(exist_ok=True)
    for old in frames.glob("scene_*.png"):
        old.unlink()

    frame_paths = []
    for index, scene in enumerate(scenes, 1):
        image = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((90, 85, W - 90, 180), radius=28, fill="#FFFFFF")
        centered(draw, 132, data.get("title", "Financial Education"), GREEN, 30, True)
        centered(draw, 500, scene.get("title", ""), GREEN, 72, True)
        centered(draw, 760, scene.get("subtitle", ""), GOLD, 48, True)
        draw.rounded_rectangle((120, 990, W - 120, 1450), radius=42, fill="#FFFFFF")
        centered(draw, 1180, scene.get("visual", "Draft visual"), MUTED, 40)
        centered(draw, 1370, scene.get("voiceover", ""), GREEN, 34)
        centered(draw, 1725, data.get("disclaimer", "Draft — for review only"), MUTED, 25)
        path = frames / f"scene_{index:02d}.png"
        image.save(path)
        frame_paths.append(path)

    concat = output / "draft_scenes.txt"
    lines = []
    for path, scene in zip(frame_paths, scenes):
        safe = path.as_posix().replace("'", "'\\''")
        lines.extend([f"file '{safe}'", f"duration {float(scene.get('duration', 1))}"])
    lines.append(f"file '{frame_paths[-1].as_posix()}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

    music = project_folder / "assets" / "music.mp3"
    result = output / f"{project_folder.name}_draft.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat)]
    if music.exists():
        command.extend(["-stream_loop", "-1", "-i", str(music)])
    command.extend(["-vf", f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    if music.exists():
        command.extend(["-filter:a", "volume=0.15", "-c:a", "aac", "-b:a", "192k", "-shortest"])
    command.append(str(result))
    subprocess.run(command, check=True)
    print(f"Created draft {result}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: draft_renderer.py PROJECT_FOLDER")
    render(Path(sys.argv[1]).resolve())
