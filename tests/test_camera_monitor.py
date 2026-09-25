import time

from family_ai.camera_monitor import CameraMonitor


class FakeCameraAgent:
    def __init__(self):
        self.calls = {"*": 0}

    def detect_people_current(self, provider):
        self.calls[provider] += 1
        return {"person_count": 1, "confidence": .9, "nearly_black": False, "text": ""}


def test_configured_camera_monitor_records_health_and_alerts():
    agent = FakeCameraAgent()
    monitor = CameraMonitor(agent, interval_seconds=.2, cooldown_seconds=60, gate_interval_seconds=.04)
    monitor.start()
    time.sleep(.65)
    monitor.stop()
    assert agent.calls["*"] >= 3
    assert monitor.events_after(0)[0]["provider"] == "*"
    status = monitor.status()
    assert status["*"]["last_success_at"]
