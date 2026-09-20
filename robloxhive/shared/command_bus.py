from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Any


@dataclass(slots=True)
class Command:
    source: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


class CommandBus:
    def __init__(self) -> None:
        self._queue: Queue[Command] = Queue()

    def publish(self, command: Command) -> None:
        self._queue.put(command)

    def receive(self, timeout: float | None = None) -> Command | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None
