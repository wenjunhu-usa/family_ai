from collections import deque
from datetime import datetime, timezone
from threading import Event, Lock, Thread
import time


class CameraMonitor:
    """Monitor the locally configured camera provider."""

    def __init__(self, camera_agent, interval_seconds: float = 6, cooldown_seconds: int = 120,
                 gate_interval_seconds: float = 2):
        self.camera_agent = camera_agent
        self.interval_seconds = interval_seconds
        self.gate_interval_seconds = gate_interval_seconds
        self.cooldown_seconds = cooldown_seconds
        self._events = deque(maxlen=50)
        self._lock = Lock()
        self._stop = Event()
        self._threads = {}
        self._next_id = 1
        self._last_alert = {}
        self._health = {"*": {"running": False, "last_success_at": None, "last_error": None}}

    def start(self):
        for provider in ("*",):
            thread = self._threads.get(provider)
            if thread and thread.is_alive():
                continue
            thread = Thread(target=self._run_provider, args=(provider,),
                            name=f"family-camera-{provider}", daemon=True)
            self._threads[provider] = thread
            thread.start()

    def stop(self):
        self._stop.set()
        for thread in self._threads.values():
            thread.join(timeout=3)

    def events_after(self, event_id: int) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._events if item["id"] > event_id]

    def status(self) -> dict:
        with self._lock:
            return {provider: {**details, "thread_alive": bool(
                self._threads.get(provider) and self._threads[provider].is_alive())}
                for provider, details in self._health.items()}

    def _set_health(self, provider: str, success: bool, error=None):
        with self._lock:
            health = self._health[provider]
            health["running"] = success
            if success:
                health["last_success_at"] = datetime.now(timezone.utc).isoformat()
                health["last_error"] = None
            else:
                health["last_error"] = f"{type(error).__name__}: {str(error)[:160]}"

    def _append_alert(self, provider: str, count: int, confidence: float):
        label = "*"
        with self._lock:
            self._events.append({
                "id": self._next_id, "provider": provider, "person_count": count,
                "confidence": confidence,
                "message": f"{label} camera detected {count} person{'s' if count != 1 else ''}.",
                "message_zh": f"{label} 摄像头检测到 {count} 个人。",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            self._next_id += 1

    def _run_provider(self, provider: str):
        interval = self.interval_seconds
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                result = self.camera_agent.detect_people_current(provider)
                self._set_health(provider, True)
                count = int(result.get("person_count", 0))
                confidence = float(result.get("confidence", 0))
                now = time.monotonic()
                if count > 0 and confidence >= .45 and now >= self._last_alert.get(provider, 0):
                    self._last_alert[provider] = now + self.cooldown_seconds
                    self._append_alert(provider, count, confidence)
            except Exception as exc:
                self._set_health(provider, False, exc)
            self._stop.wait(max(.2, interval - (time.monotonic() - started)))
