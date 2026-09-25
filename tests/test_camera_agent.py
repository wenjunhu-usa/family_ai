import json
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

import family_ai.camera_agent as camera_module
from family_ai.camera_agent import CameraAgent


def _agent_with_frame(tmp_path, monkeypatch):
    helper = tmp_path / "capture-helper"
    helper.write_text("")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"local image bytes")
    agent = CameraAgent(helper, "http://127.0.0.1:11434", "local-vision")
    monkeypatch.setattr(agent, "_capture_frame", lambda *_args, **_kwargs: frame)
    return agent


def _mock_ollama(monkeypatch, model_content):
    captured = {}

    @contextmanager
    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout

        class Response:
            def read(self):
                return json.dumps({"message": {"content": model_content}}).encode()

        yield Response()

    monkeypatch.setattr(camera_module, "urlopen", fake_urlopen)
    return captured


def test_inspection_uses_json_schema_and_returns_validated_dict(tmp_path, monkeypatch):
    agent = _agent_with_frame(tmp_path, monkeypatch)
    content = json.dumps({
        "person_present": True,
        "person_count": 1,
        "description": "One person is visible.",
        "confidence": 0.92,
    })
    captured = _mock_ollama(monkeypatch, content)

    result = agent.inspect_current("*", "Is anyone visible?", "en")

    assert result["person_count"] == 1
    schema = captured["body"]["format"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "person_present", "person_count", "description", "confidence",
    }


@pytest.mark.parametrize("content", [
    '{"person_present":true,"person_count":1,"description":"Visible"}',
    '{"person_present":true,"person_count":-1,"description":"Visible","confidence":0.8}',
    '{"person_present":true,"person_count":1,"description":"Visible","confidence":1.2}',
    '{"person_present":true,"person_count":1,"description":"Visible","confidence":0.8,"identity":"Member"}',
])
def test_inspection_rejects_incomplete_or_out_of_policy_output(tmp_path, monkeypatch, content):
    agent = _agent_with_frame(tmp_path, monkeypatch)
    _mock_ollama(monkeypatch, content)

    with pytest.raises(ValidationError):
        agent.inspect_current("*", "Describe the view", "en")
