from collections import Counter
from contextlib import contextmanager
from threading import Lock

_lock = Lock()
_active: Counter[str] = Counter()

def mark_active(agent_id: str) -> None:
    with _lock:
        _active[agent_id] += 1

def mark_inactive(agent_id: str) -> None:
    with _lock:
        if _active[agent_id] <= 1:
            _active.pop(agent_id, None)
        else:
            _active[agent_id] -= 1

def active_agent_ids() -> set[str]:
    with _lock:
        return set(_active)

@contextmanager
def agent_activity(agent_id: str):
    mark_active(agent_id)
    try:
        yield
    finally:
        mark_inactive(agent_id)
