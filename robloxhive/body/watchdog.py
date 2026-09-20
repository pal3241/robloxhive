from __future__ import annotations

from time import monotonic
from collections.abc import Callable


class BrainWatchdog:
    """Local failsafe: stop body input if brain heartbeat disappears."""

    def __init__(self, release_all: Callable[[], None], timeout_s: float = 5.0) -> None:
        self.release_all = release_all
        self.timeout_s = timeout_s
        self._last_heartbeat = monotonic()
        self.tripped = False

    def heartbeat(self) -> None:
        self._last_heartbeat = monotonic()
        self.tripped = False

    def tick(self) -> bool:
        if monotonic() - self._last_heartbeat <= self.timeout_s:
            return False
        if not self.tripped:
            self.release_all()
            self.tripped = True
        return True
