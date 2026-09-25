from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Condition
from uuid import uuid4


@dataclass
class ModelTaskQueue:
    """FIFO gate: one complete model-driven workflow runs at a time."""

    _condition: Condition = field(default_factory=Condition)
    _waiting: list[str] = field(default_factory=list)
    _active: str | None = None

    def reserve(self) -> str:
        ticket = str(uuid4())
        with self._condition:
            self._waiting.append(ticket)
            self._condition.notify_all()
        return ticket

    def position(self, ticket: str) -> int:
        with self._condition:
            if self._active == ticket:
                return 0
            try:
                return self._waiting.index(ticket) + 1
            except ValueError:
                return -1

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "active": self._active is not None,
                "waiting": len(self._waiting),
            }

    @contextmanager
    def run(self, ticket: str):
        with self._condition:
            while self._active is not None or not self._waiting or self._waiting[0] != ticket:
                self._condition.wait()
            self._waiting.pop(0)
            self._active = ticket
        try:
            yield
        finally:
            with self._condition:
                if self._active == ticket:
                    self._active = None
                self._condition.notify_all()
