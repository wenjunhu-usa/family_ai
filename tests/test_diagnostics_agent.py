from pathlib import Path

from family_ai.diagnostics_agent import DiagnosticsAgent


class Storage:
    def healthcheck(self):
        return {"reachable": True, "latency_ms": 2}


class Worker:
    agent = None

    def is_alive(self):
        return False


class Registry:
    def active_count(self, agent):
        return 0

    def latest(self, member, agent):
        return None


def test_diagnostics_returns_structured_targeted_database_observation(tmp_path):
    agent = DiagnosticsAgent(Storage(), tmp_path / "USB" / "Gmail", "http://localhost",
                             Worker(), object(), Registry())
    result = agent.inspect("database", "member")
    assert result.status == "success" and result.verified
    assert result.value["checks"]["database"]["details"]["reachable"] is True
    assert "drive" not in result.value["checks"]


def test_diagnostics_isolates_component_failure(tmp_path):
    class BrokenStorage:
        def healthcheck(self):
            raise ConnectionError("offline")

    result = DiagnosticsAgent(BrokenStorage(), Path(tmp_path), "http://localhost",
                              Worker(), object(), Registry()).inspect("database", "member")
    assert result.status == "success"
    assert result.value["checks"]["database"]["status"] == "unhealthy"
