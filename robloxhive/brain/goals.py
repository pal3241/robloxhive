from __future__ import annotations

import heapq
import itertools
from robloxhive.shared.models import Goal


class GoalManager:
    def __init__(self) -> None:
        self._queue: list[tuple[int, int, Goal]] = []
        self._counter = itertools.count()
        self.active: Goal | None = None

    def submit(self, goal: Goal) -> None:
        heapq.heappush(self._queue, (-goal.priority, next(self._counter), goal))

    def next(self) -> Goal | None:
        if not self._queue:
            self.active = None
            return None
        self.active = heapq.heappop(self._queue)[2]
        return self.active

    def interrupt(self, goal: Goal) -> bool:
        if self.active and not self.active.interruptible:
            self.submit(goal)
            return False
        if self.active and self.active.persistent:
            self.submit(self.active)
        self.active = goal
        return True
