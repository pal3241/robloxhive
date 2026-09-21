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
        self.game_adapter: Any | None = None
        self.input_backend: Any | None = None
        self.background_input_backend: Any | None = None

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

    def attach_game_adapter(self, adapter: Any) -> None:
        self.game_adapter = adapter

    def attach_input(self, input_backend: Any, background_input_backend: Any | None = None) -> None:
        self.input_backend = input_backend
        self.background_input_backend = background_input_backend

    def direct_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_mode = str(payload.get("mode") or "background").lower()
        backend = (
            self.background_input_backend
            if requested_mode == "background" and self.background_input_backend is not None
            else self.input_backend
        )
        if backend is None:
            return {"ok": False, "error": "INPUT_NOT_ATTACHED"}
        action = str(payload.get("action") or "").lower()
        seconds = max(0.01, min(float(payload.get("seconds") or 0.15), 3.0))
        try:
            if action in {"forward", "back", "left", "right"}:
                backend.move(action, seconds)
            elif action == "jump":
                backend.key("jump", min(seconds, 0.25))
            elif action == "interact":
                backend.interact()
            elif action == "release":
                # Release both paths so switching between background dashboard
                # control and reliable foreground automation cannot leave a key held.
                released = set()
                for candidate in (self.input_backend, self.background_input_backend):
                    if candidate is None or id(candidate) in released:
                        continue
                    released.add(id(candidate))
                    candidate.release_all()
            else:
                return {"ok": False, "error": "UNSUPPORTED_DIRECT_ACTION", "action": action}
            return {
                "ok": True,
                "action": action,
                "seconds": seconds,
                "mode": getattr(backend, "mode", requested_mode),
            }
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__, "message": str(exc), "action": action}

    def autonomy_tick(self) -> dict[str, Any]:
        if self.game_adapter is None:
            return {"adapter": None, "enabled": False}
        try:
            return self.game_adapter.tick()
        except Exception as exc:
            return {
                "adapter": type(self.game_adapter).__name__,
                "enabled": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def game_status(self) -> dict[str, Any]:
        if self.game_adapter is None:
            return {"adapter": None, "enabled": False}
        try:
            return self.game_adapter.status()
        except Exception as exc:
            return {"adapter": type(self.game_adapter).__name__, "error": type(exc).__name__}

    def game_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.game_adapter is None:
            return {"ok": False, "error": "NO_GAME_ADAPTER"}
        control = getattr(self.game_adapter, "control", None)
        if not callable(control):
            return {"ok": False, "error": "GAME_ADAPTER_NOT_CONTROLLABLE"}
        return control(payload)

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
                "game": self.game_status(),
                "input": {
                    "attached": self.input_backend is not None,
                    "backend": type(self.input_backend).__name__ if self.input_backend is not None else None,
                    "mode": getattr(self.input_backend, "mode", None) if self.input_backend is not None else None,
                    "background_available": self.background_input_backend is not None,
                    "background_backend": (
                        type(self.background_input_backend).__name__
                        if self.background_input_backend is not None
                        else None
                    ),
                },
            },
        }
