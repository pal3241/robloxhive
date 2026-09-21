from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from robloxhive.core.native_math import native_math
from robloxhive.shared.models import ActionResult, ActionStatus


@dataclass(slots=True)
class AimConfig:
    lead_seconds: float = 0.10
    max_lead_px: float = 110.0
    upper_torso_ratio: float = 0.36
    min_confidence: float = 0.45


class AdvancedGenericSkills:
    """Useful cross-game skills for UI, exploration and FPS-style combat."""

    def __init__(
        self,
        perception: Any,
        input_backend: Any,
        *,
        config: AimConfig | None = None,
    ) -> None:
        self.perception = perception
        self.input = input_backend
        self.config = config or AimConfig()

    @staticmethod
    def _target(payload: dict[str, Any]) -> str:
        for key in ("username", "target", "label", "enemy", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in ("username", "target", "enemy", "text"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def observe(self, payload: dict[str, Any]) -> ActionResult:
        return ActionResult(
            action="observe",
            status=ActionStatus.SUCCESS,
            details={"evidence": {"observation_requested": True}},
        )

    def wait(self, payload: dict[str, Any]) -> ActionResult:
        seconds = max(0.05, min(float(payload.get("seconds") or 0.35), 3.0))
        time.sleep(seconds)
        return ActionResult(
            action="wait",
            status=ActionStatus.SUCCESS,
            duration_ms=int(seconds * 1000),
            details={"seconds": seconds},
        )

    def press_key(self, payload: dict[str, Any]) -> ActionResult:
        key = str(payload.get("key") or "").strip()
        if not key:
            return ActionResult(
                action="press_key",
                status=ActionStatus.BLOCKED,
                error="KEY_REQUIRED",
                recoverable=False,
            )
        seconds = max(0.02, min(float(payload.get("seconds") or 0.08), 1.5))
        self.input.key(key, seconds)
        return ActionResult(
            action="press_key",
            status=ActionStatus.SUCCESS,
            details={"key": key, "seconds": seconds},
        )

    def click_ui(self, payload: dict[str, Any]) -> ActionResult:
        text = self._target(payload)
        if not text:
            return ActionResult(
                action="click_ui",
                status=ActionStatus.BLOCKED,
                error="UI_TEXT_REQUIRED",
                recoverable=False,
            )

        hit = self.perception.find(text)
        if hit is None:
            return ActionResult(
                action="click_ui",
                status=ActionStatus.FAILED,
                error="UI_TEXT_NOT_FOUND",
                recoverable=True,
                details={"text": text},
            )

        x, y = int(hit.center_x), int(hit.center_y)
        self.input.click_client(x, y, str(payload.get("button") or "left"))
        time.sleep(0.12)
        return ActionResult(
            action="click_ui",
            status=ActionStatus.SUCCESS,
            details={
                "text": text,
                "source": hit.source,
                "confidence": hit.confidence,
                "point": [x, y],
                "evidence": {"ui_clicked": True},
            },
        )

    def explore(self, payload: dict[str, Any]) -> ActionResult:
        seconds = max(0.08, min(float(payload.get("seconds") or 0.28), 1.5))
        pattern = str(payload.get("pattern") or "forward_right").lower()
        if pattern == "left":
            self.input.move("left", seconds)
        elif pattern == "right":
            self.input.move("right", seconds)
        elif pattern == "back":
            self.input.move("back", seconds)
        else:
            self.input.move("forward", seconds)
            self.input.move("right", min(0.12, seconds))
        return ActionResult(
            action="explore",
            status=ActionStatus.SUCCESS,
            details={"pattern": pattern, "seconds": seconds},
        )

    def _find_combat_target(self, payload: dict[str, Any]):
        target = self._target(payload)
        if not target:
            return "", None
        player_target = bool(payload.get("username")) or bool(payload.get("player"))
        query = f"player:{target}" if player_target and not target.lower().startswith("player:") else target
        return target, self.perception.find(query)

    def aim(self, payload: dict[str, Any]) -> ActionResult:
        target, hit = self._find_combat_target(payload)
        if not target:
            return ActionResult(
                action="aim",
                status=ActionStatus.BLOCKED,
                error="TARGET_REQUIRED",
                recoverable=False,
            )
        if hit is None:
            return ActionResult(
                action="aim",
                status=ActionStatus.FAILED,
                error="TARGET_NOT_VISIBLE",
                recoverable=True,
                details={"target": target},
            )
        if hit.confidence < self.config.min_confidence:
            return ActionResult(
                action="aim",
                status=ActionStatus.FAILED,
                error="TARGET_CONFIDENCE_TOO_LOW",
                recoverable=True,
                details={"target": target, "confidence": hit.confidence},
            )

        velocity = hit.metadata.get("velocity_px_s", [0.0, 0.0])
        try:
            vx, vy = float(velocity[0]), float(velocity[1])
        except (TypeError, ValueError, IndexError):
            vx, vy = 0.0, 0.0

        lead_s = max(0.0, min(float(payload.get("lead_seconds") or self.config.lead_seconds), 0.5))
        ratio = max(0.15, min(float(payload.get("vertical_ratio") or self.config.upper_torso_ratio), 0.85))
        x = int(native_math.predict_axis(hit.center_x, vx, lead_s, self.config.max_lead_px))
        y = int(native_math.predict_axis(hit.y + hit.height * ratio, vy, lead_s, self.config.max_lead_px))

        self.input.aim_client(x, y)
        return ActionResult(
            action="aim",
            status=ActionStatus.SUCCESS,
            details={
                "target": target,
                "point": [x, y],
                "confidence": hit.confidence,
                "track_id": hit.track_id,
                "velocity_px_s": [vx, vy],
                "lead_seconds": lead_s,
                "native_acceleration": native_math.enabled,
                "evidence": {"target_visible": True, "aimed": True},
            },
        )

    def combat(self, payload: dict[str, Any]) -> ActionResult:
        aimed = self.aim(payload)
        if aimed.status != ActionStatus.SUCCESS:
            return ActionResult(
                action="combat",
                status=aimed.status,
                error=aimed.error,
                recoverable=aimed.recoverable,
                details=aimed.details,
            )

        point = aimed.details.get("point") or [0, 0]
        button = str(payload.get("button") or "left")
        weapon_key = payload.get("weapon_key")
        if weapon_key:
            self.input.key(str(weapon_key), 0.05)
        self.input.click_client(int(point[0]), int(point[1]), button)
        return ActionResult(
            action="combat",
            status=ActionStatus.SUCCESS,
            details={
                **aimed.details,
                "button": button,
                "weapon_key": weapon_key,
                "evidence": {"target_visible": True, "aimed": True, "attack_sent": True},
            },
        )

    def register_into(self, executor: Any) -> None:
        executor.register("observe", self.observe)
        executor.register("wait", self.wait)
        executor.register("press_key", self.press_key)
        executor.register("click_ui", self.click_ui)
        executor.register("explore", self.explore)
        executor.register("aim", self.aim)
        executor.register("combat", self.combat)
