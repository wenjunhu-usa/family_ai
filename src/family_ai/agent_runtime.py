from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


OutcomeStatus = Literal[
    "success", "transient_error", "permanent_error",
    "needs_authorization", "needs_specialist",
]


@dataclass(frozen=True)
class AgentOutcome:
    status: OutcomeStatus
    value: Any = None
    operation: str = ""
    agent: str = ""
    attempts: int = 1
    error_type: str | None = None
    error_summary: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    verified: bool = False


def classify_error(exc: BaseException) -> OutcomeStatus:
    name = type(exc).__name__
    detail = f"{name}: {exc}".casefold()
    if any(marker in detail for marker in (
        "oauth", "not authorized", "unauthorized", "http 401", "http 403",
        "permission denied", "access denied", "permission is missing", "reconnect this google account",
    )):
        return "needs_authorization"
    if name in {
        "TimeoutError", "ConnectionError", "RemoteDisconnected", "ConnectionResetError",
        "OperationalError", "InterfaceError", "AdminShutdown", "ConnectionException",
    } or any(marker in detail for marker in (
        "timed out", "timeout", "network is down", "connection is bad",
        "temporarily unavailable", "http 429", "http 500", "http 502", "http 503", "http 504",
    )):
        return "transient_error"
    if isinstance(exc, (ValueError, FileNotFoundError, PermissionError)) or any(
        marker in detail for marker in ("not configured", "not mounted", "invalid account")
    ):
        return "permanent_error"
    return "needs_specialist"


class AgentRuntime:
    """Bounded observe-act-verify loop shared by every specialist."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 0.25,
                 sleeper: Callable[[float], None] = time.sleep):
        self.max_attempts = max(1, max_attempts)
        self.base_delay = max(0, base_delay)
        self.sleeper = sleeper

    def execute(
        self,
        agent: str,
        operation: str,
        action: Callable[[], Any],
        *,
        diagnose: Callable[[BaseException], dict[str, Any]] | None = None,
        verify: Callable[[Any], bool] | None = None,
    ) -> AgentOutcome:
        observations: dict[str, Any] = {}
        for attempt in range(1, self.max_attempts + 1):
            try:
                value = action()
                verified = verify(value) if verify else True
                if not verified:
                    raise RuntimeError("post-operation verification failed")
                return AgentOutcome(
                    "success", value, operation, agent, attempt,
                    diagnostics=observations, verified=True,
                )
            except Exception as exc:
                status = classify_error(exc)
                if diagnose:
                    try:
                        observations.update(diagnose(exc))
                    except Exception as diagnostic_exc:
                        observations["diagnostic_error"] = type(diagnostic_exc).__name__
                if status == "transient_error" and attempt < self.max_attempts:
                    self.sleeper(self.base_delay * (2 ** (attempt - 1)))
                    continue
                return AgentOutcome(
                    status, None, operation, agent, attempt,
                    type(exc).__name__, str(exc)[:300], observations, False,
                )
        raise AssertionError("unreachable")
