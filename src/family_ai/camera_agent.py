import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field


class CameraInspection(BaseModel):
    """Schema-constrained result from the local vision model."""

    model_config = ConfigDict(extra="forbid")

    person_present: bool
    person_count: int = Field(ge=0)
    description: str
    confidence: float = Field(ge=0, le=1)


class CameraAgent:
    name = "Camera Agent"
    permissions = ("camera_window_read",)

    def __init__(self, capture_helper: Path, ollama_base_url: str, model: str):
        self.capture_helper = capture_helper
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model = model

    def _capture_frame(self, provider: str, directory: str, activate: bool) -> Path:
        if provider != "*":
            raise ValueError("provider must be configured locally")
        app_name = "*"
        if activate:
            subprocess.run(["/usr/bin/open", "-a", app_name], check=True, timeout=10)
            time.sleep(3)
        window_path = Path(directory) / f"{provider}-window.png"
        window_id = subprocess.run(
            [str(self.capture_helper), "id", provider], check=True,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        if not window_id.isdigit():
            raise RuntimeError("camera window id was not available")
        subprocess.run(
            ["/usr/sbin/screencapture", "-x", "-l", window_id, str(window_path)],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return window_path

    def detect_people_current(self, provider: str) -> dict:
        if not self.capture_helper.is_file():
            raise RuntimeError("camera capture helper is not installed")
        with tempfile.TemporaryDirectory(prefix="family-ai-camera-monitor-") as directory:
            frame_path = self._capture_frame(provider, directory, activate=False)
            output = subprocess.run(
                [str(self.capture_helper), "detect", str(frame_path)], check=True,
                capture_output=True, text=True, timeout=20,
            ).stdout
        return json.loads(output)

    def inspect_current(self, provider: str, question: str, language: str) -> dict:
        if provider != "*":
            raise ValueError("provider must be configured locally")
        if not self.capture_helper.is_file():
            raise RuntimeError("camera capture helper is not installed")
        with tempfile.TemporaryDirectory(prefix="family-ai-camera-") as directory:
            frame_path = self._capture_frame(provider, directory, activate=True)
            encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        prompt = f"""Analyze this current home security camera frame locally.
User request: {question}
Return JSON only with keys person_present (boolean), person_count (integer), description (short string), confidence (number 0 to 1).
Answer description in {'Chinese' if language == 'zh' else 'English'}.
Do not identify any person, guess identity, infer sensitive traits, or claim details that are not visible."""
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "format": CameraInspection.model_json_schema(),
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
        }).encode()
        request = Request(
            f"{self.ollama_base_url}/api/chat", data=body,
            headers={"content-type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=180) as response:
            content = json.loads(response.read())["message"]["content"]
        result = CameraInspection.model_validate_json(content)
        description = result.description.lower()
        if any(marker in description for marker in ("entirely black", "completely black", "全黑", "黑色画面")):
            raise RuntimeError("camera live view is black or not ready")
        return result.model_dump()
