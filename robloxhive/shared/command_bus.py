from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Condition
from time import monotonic
from typing import Any


@dataclass(slots=True)
class Command:
    source: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


class CommandBus:
    """Thread-safe command queue with optional per-agent routing.

    Commands without an explicit agent_id remain backwards compatible and are
    consumed by agent-01 when using receive_for().
    """

    def __init__(self) -> None:
        self._items: deque[Command] = deque()
        self._cv = Condition()

    def publish(self, command: Command) -> None:
        with self._cv:
            self._items.append(command)
            self._cv.notify_all()

    def receive(self, timeout: float | None = None) -> Command | None:
        deadline = None if timeout is None else monotonic() + max(0.0, timeout)
        with self._cv:
            while not self._items:
                if deadline is None:
                    self._cv.wait()
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            return self._items.popleft()

    def receive_for(self, agent_id: str, timeout: float | None = None) -> Command | None:
        deadline = None if timeout is None else monotonic() + max(0.0, timeout)
        with self._cv:
            while True:
                for index, command in enumerate(self._items):
                    target = command.payload.get("agent_id")
                    if target == agent_id or (target is None and agent_id == "agent-01"):
                        selected = command
                        del self._items[index]
                        return selected

                if deadline is None:
                    self._cv.wait()
                    continue
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
