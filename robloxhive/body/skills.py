from __future__ import annotations

from collections.abc import Callable
from typing import Any

from robloxhive.shared.models import ActionResult, ActionStatus


SkillHandler = Callable[[dict[str, Any]], ActionResult]


class SkillExecutor:
    """Body-side registry for semantic skills emitted by the Brain planner.

    Game adapters register handlers here. Unknown skills fail safely instead of
    falling through to arbitrary input.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, SkillHandler] = {}

    def register(self, name: str, handler: SkillHandler) -> None:
        self._handlers[name] = handler

    def execute(self, skill: str, payload: dict[str, Any]) -> ActionResult:
        handler = self._handlers.get(skill)
        if handler is None:
            return ActionResult(
                action=skill,
                status=ActionStatus.BLOCKED,
                error="UNSUPPORTED_SKILL",
                recoverable=False,
                details={"skill": skill},
            )
        try:
            return handler(payload)
        except Exception as exc:
            return ActionResult(
                action=skill,
                status=ActionStatus.FAILED,
                error=type(exc).__name__,
                recoverable=True,
                details={"message": str(exc)},
            )

    def available(self) -> list[str]:
        return sorted(self._handlers)
