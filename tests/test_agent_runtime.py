from family_ai.agent_runtime import AgentRuntime, classify_error


class OperationalError(Exception):
    pass


def test_classifies_database_outage_as_transient():
    assert classify_error(OperationalError("Network is down")) == "transient_error"


def test_classifies_oauth_as_authorization():
    assert classify_error(RuntimeError("Google API returned HTTP 401")) == "needs_authorization"


def test_runtime_diagnoses_retries_and_verifies():
    calls = []

    def action():
        calls.append(1)
        if len(calls) == 1:
            raise OperationalError("temporary")
        return {"started": True}

    outcome = AgentRuntime(max_attempts=2, sleeper=lambda _delay: None).execute(
        "Gmail Backup Agent", "sync", action,
        diagnose=lambda _exc: {"postgresql": "reachable"},
        verify=lambda value: value["started"],
    )
    assert outcome.status == "success"
    assert outcome.attempts == 2
    assert outcome.verified
    assert outcome.diagnostics == {"postgresql": "reachable"}


def test_runtime_does_not_retry_permanent_failure():
    outcome = AgentRuntime(max_attempts=3, sleeper=lambda _delay: None).execute(
        "Camera Agent", "capture", lambda: (_ for _ in ()).throw(FileNotFoundError("helper"))
    )
    assert outcome.status == "permanent_error"
    assert outcome.attempts == 1
