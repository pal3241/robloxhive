from __future__ import annotations

from collections.abc import Callable
from typing import Any

from robloxhive.shared.models import ActionResult, ActionStatus


SkillHandler = Callable[[dict[str, Any]], ActionResult]


class SkillExecutor:
    """Body-side registry for semantic skills emitted by the Brain planner."""

    def __init__(self) -> None:
        self._handlers: dict[str, SkillHandler] = {}
        self.metadata: dict[str, Any] = {}
        self.perception: Any | None = None
        self.navigation: Any | None = None

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

    def attach_perception(self, perception: Any, metadata: dict[str, Any] | None = None) -> None:
        self.perception = perception
        if metadata:
            self.metadata.update(metadata)

    def attach_navigation(self, navigation: Any) -> None:
        self.navigation = navigation

    def navigation_probe(self) -> dict[str, Any]:
        if self.navigation is None:
            return {"available": False, "error": "NAVIGATION_NOT_ATTACHED"}
        diagnostics = getattr(self.navigation, "diagnostics", None)
        if not callable(diagnostics):
            return {"available": True, "adapter": type(self.navigation).__name__}
        return {"available": True, **diagnostics()}

    def probe(self, label: str) -> dict[str, Any]:
        if self.perception is None:
            return {"found": False, "error": "PERCEPTION_NOT_ATTACHED", "label": label}
        hit = self.perception.find(label)
        if hit is None:
            return {
                "found": False,
                "label": label,
                "diagnostics": self._perception_diagnostics(),
            }
        return {
            "found": True,
            "label": label,
            "detection": {
                "label": hit.label,
                "confidence": hit.confidence,
                "x": hit.x,
                "y": hit.y,
                "width": hit.width,
                "height": hit.height,
                "source": hit.source,
                "track_id": hit.track_id,
                "metadata": hit.metadata,
            },
            "diagnostics": self._perception_diagnostics(),
        }

    def _perception_diagnostics(self) -> dict[str, Any]:
        if self.perception is None:
            return {}
        fn = getattr(self.perception, "diagnostics", None)
        if callable(fn):
            try:
                return fn()
            except Exception as exc:
                return {"error": type(exc).__name__}
        return {"adapter": type(self.perception).__name__}

    def describe(self) -> dict[str, Any]:
        return {
            "skills": self.available(),
            "metadata": {
                **self.metadata,
                "perception": self._perception_diagnostics(),
                "navigation": self.navigation_probe(),
            },
        }
