from __future__ import annotations

from abc import ABC, abstractmethod
from time import monotonic
from robloxhive.body.instances import InstanceManager
from robloxhive.shared.models import ActionResult, ActionStatus


class InputBackend(ABC):
    @abstractmethod
    def release_all(self, hwnd: int) -> None: ...

    @abstractmethod
    def move(self, hwnd: int, direction: str, seconds: float) -> None: ...


class BodyController:
    def __init__(self, instances: InstanceManager, backend: InputBackend) -> None:
        self.instances = instances
        self.backend = backend

    def move(self, agent_id: str, direction: str, seconds: float) -> ActionResult:
        started = monotonic()
        if direction not in {"forward", "back", "left", "right"}:
            return ActionResult(action="move", status=ActionStatus.BLOCKED, error="INVALID_DIRECTION")
        if seconds <= 0 or seconds > 10:
            return ActionResult(action="move", status=ActionStatus.BLOCKED, error="INVALID_DURATION")

        inst = self.instances.resolve_for_agent(agent_id)
        try:
            self.backend.move(inst.hwnd, direction, seconds)
            return ActionResult(
                action="move",
                status=ActionStatus.SUCCESS,
                duration_ms=int((monotonic() - started) * 1000),
                details={"pid": inst.pid, "hwnd": inst.hwnd, "direction": direction},
            )
        except Exception as exc:
            self.backend.release_all(inst.hwnd)
            return ActionResult(
                action="move",
                status=ActionStatus.FAILED,
                duration_ms=int((monotonic() - started) * 1000),
                error=type(exc).__name__,
            )

    def emergency_stop(self, agent_id: str) -> None:
        inst = self.instances.resolve_for_agent(agent_id)
        self.backend.release_all(inst.hwnd)
