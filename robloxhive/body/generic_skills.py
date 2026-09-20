from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from robloxhive.body.perception import Detection, PerceptionAdapter
from robloxhive.shared.models import ActionResult, ActionStatus


class SkillInput(Protocol):
    def move(self, direction: str, seconds: float) -> None: ...
    def interact(self, key: str = "interact") -> None: ...
    def click_client(self, x: int, y: int, button: str = "left") -> None: ...
    def release_all(self) -> None: ...


@dataclass(slots=True)
class SkillConfig:
    center_tolerance: float = 0.14
    near_area_ratio: float = 0.075
    steering_seconds: float = 0.12
    forward_seconds: float = 0.18
    max_iterations: int = 35
    lost_target_limit: int = 5
    follow_near_area_ratio: float = 0.055
    follow_far_area_ratio: float = 0.018
    settle_seconds: float = 0.05
    interact_key: str = "interact"


class GenericVisualSkills:
    """Generic low-cost visual skills.

    They use normalized screen position and apparent target size. That makes
    them game-agnostic enough for a baseline while leaving true map/SLAM
    navigation to a future navigation subsystem.
    """

    def __init__(
        self,
        perception: PerceptionAdapter,
        input_backend: SkillInput,
        config: SkillConfig | None = None,
        navigator: Any | None = None,
    ) -> None:
        self.perception = perception
        self.input = input_backend
        self.config = config or SkillConfig()
        self.navigator = navigator

    @staticmethod
    def _target(payload: dict[str, Any]) -> str | None:
        for key in ("target", "label", "player", "item", "object"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("target")
            if isinstance(value, str) and value.strip():
                return value.strip()

        instruction = str(payload.get("instruction") or "").strip()
        quoted = re.findall(r'["\']([^"\']+)["\']', instruction)
        if quoted:
            return quoted[0].strip()

        patterns = (
            r"(?:follow_player|follow|ikuti)\s+(.+)$",
            r"(?:go to|travel to|reach|navigate to|pergi ke|menuju)\s+(?:the\s+)?(.+)$",
            r"(?:collect|loot|pick up|pickup|gather|ambil|kumpulkan)\s+(?:the\s+)?(.+)$",
            r"(?:interact with|open|use)\s+(?:the\s+)?(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, instruction, flags=re.IGNORECASE)
            if not match:
                continue
            target = re.sub(r"[.!?]+$", "", match.group(1)).strip()
            if target:
                return target
        return None

    def _frame_metrics(self, detection: Detection) -> tuple[float, float]:
        width, height = self.perception.frame_size()
        if width <= 0 or height <= 0:
            return 0.0, 0.0
        offset = (detection.center_x / width) - 0.5
        area_ratio = detection.area / float(width * height)
        return offset, area_ratio

    def _approach(
        self,
        label: str,
        near_area_ratio: float | None = None,
        max_iterations: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        cfg = self.config
        near = near_area_ratio if near_area_ratio is not None else cfg.near_area_ratio
        iterations = max_iterations or cfg.max_iterations

        if self.navigator is not None:
            return self.navigator.navigate_to(
                label,
                near_area_ratio=near,
                max_iterations=iterations,
            )
        lost = 0
        last: Detection | None = None

        for index in range(iterations):
            detection = self.perception.find(label)
            if detection is None:
                lost += 1
                if lost > cfg.lost_target_limit:
                    self.input.release_all()
                    return False, {
                        "reason": "TARGET_LOST",
                        "target": label,
                        "iterations": index + 1,
                    }
                # Small scan/steer step. This stays HWND-scoped.
                self.input.move("right", cfg.steering_seconds)
                time.sleep(cfg.settle_seconds)
                continue

            lost = 0
            last = detection
            offset, area_ratio = self._frame_metrics(detection)
            if abs(offset) > cfg.center_tolerance:
                self.input.move("right" if offset > 0 else "left", cfg.steering_seconds)
            elif area_ratio < near:
                self.input.move("forward", cfg.forward_seconds)
            else:
                self.input.release_all()
                return True, {
                    "target": label,
                    "confidence": detection.confidence,
                    "offset": round(offset, 4),
                    "area_ratio": round(area_ratio, 5),
                    "iterations": index + 1,
                }
            time.sleep(cfg.settle_seconds)

        self.input.release_all()
        details: dict[str, Any] = {"reason": "APPROACH_TIMEOUT", "target": label}
        if last:
            offset, area_ratio = self._frame_metrics(last)
            details.update(
                confidence=last.confidence,
                offset=round(offset, 4),
                area_ratio=round(area_ratio, 5),
            )
        return False, details

    def navigate(self, payload: dict[str, Any]) -> ActionResult:
        target = self._target(payload)
        if not target:
            return ActionResult(
                action="navigate",
                status=ActionStatus.BLOCKED,
                error="TARGET_REQUIRED",
                recoverable=False,
            )
        ok, details = self._approach(target)
        return ActionResult(
            action="navigate",
            status=ActionStatus.SUCCESS if ok else ActionStatus.FAILED,
            error=None if ok else details.get("reason", "NAVIGATION_FAILED"),
            recoverable=not ok,
            details={
                **details,
                "evidence": {
                    "target_visible": ok,
                    "target": target,
                },
            },
        )

    def interact(self, payload: dict[str, Any]) -> ActionResult:
        target = self._target(payload)
        if target:
            ok, details = self._approach(target)
            if not ok:
                return ActionResult(
                    action="interact",
                    status=ActionStatus.FAILED,
                    error=details.get("reason", "TARGET_UNREACHABLE"),
                    recoverable=True,
                    details=details,
                )
        else:
            details = {}

        before = self.perception.find(target) if target else None
        key = str(payload.get("key") or self.config.interact_key)
        self.input.interact(key)
        time.sleep(0.18)
        after = self.perception.find(target) if target else None

        changed = (
            target is None
            or (before is not None and after is None)
            or (
                before is not None
                and after is not None
                and abs(after.area - before.area) > max(before.area * 0.18, 1.0)
            )
        )
        verified = target is None or changed
        return ActionResult(
            action="interact",
            status=ActionStatus.SUCCESS if verified else ActionStatus.FAILED,
            error=None if verified else "INTERACTION_NOT_VERIFIED",
            recoverable=not verified,
            details={
                **details,
                "target": target,
                "key": key,
                "evidence": {
                    "interaction_sent": True,
                    "visual_change": changed,
                    "verified": changed if target else False,
                },
            },
        )

    def collect(self, payload: dict[str, Any]) -> ActionResult:
        target = self._target(payload)
        if not target:
            return ActionResult(
                action="collect",
                status=ActionStatus.BLOCKED,
                error="ITEM_REQUIRED",
                recoverable=False,
            )

        ok, details = self._approach(target)
        if not ok:
            return ActionResult(
                action="collect",
                status=ActionStatus.FAILED,
                error=details.get("reason", "ITEM_UNREACHABLE"),
                recoverable=True,
                details=details,
            )

        before = self.perception.find(target)
        self.input.interact(str(payload.get("key") or self.config.interact_key))
        time.sleep(0.2)
        after = self.perception.find(target)

        disappeared = before is not None and after is None
        changed = disappeared or (
            before is not None
            and after is not None
            and abs(after.area - before.area) > max(before.area * 0.25, 1.0)
        )
        return ActionResult(
            action="collect",
            status=ActionStatus.SUCCESS if changed else ActionStatus.FAILED,
            error=None if changed else "COLLECTION_NOT_VERIFIED",
            recoverable=not changed,
            details={
                **details,
                "target": target,
                "evidence": {
                    "verified": disappeared,
                    "target_disappeared": disappeared,
                    "visual_change": changed,
                },
            },
        )

    def follow_player(self, payload: dict[str, Any]) -> ActionResult:
        target = self._target(payload)
        if not target:
            return ActionResult(
                action="follow_player",
                status=ActionStatus.BLOCKED,
                error="PLAYER_TARGET_REQUIRED",
                recoverable=False,
            )

        cfg = self.config
        max_iterations = int(payload.get("iterations") or cfg.max_iterations)
        stable = 0
        lost = 0

        for index in range(max(1, min(max_iterations, 300))):
            detection = self.perception.find(f"player:{target}")
            if detection is None:
                lost += 1
                if lost > cfg.lost_target_limit:
                    self.input.release_all()
                    return ActionResult(
                        action="follow_player",
                        status=ActionStatus.FAILED,
                        error="PLAYER_LOST",
                        recoverable=True,
                        details={"target": target, "iterations": index + 1},
                    )
                self.input.move("right", cfg.steering_seconds)
                continue

            lost = 0
            offset, area_ratio = self._frame_metrics(detection)
            if abs(offset) > cfg.center_tolerance:
                self.input.move("right" if offset > 0 else "left", cfg.steering_seconds)
                stable = 0
            elif area_ratio < cfg.follow_far_area_ratio:
                self.input.move("forward", cfg.forward_seconds)
                stable = 0
            elif area_ratio > cfg.follow_near_area_ratio:
                self.input.move("back", cfg.forward_seconds)
                stable = 0
            else:
                self.input.release_all()
                stable += 1
                if stable >= 3:
                    return ActionResult(
                        action="follow_player",
                        status=ActionStatus.SUCCESS,
                        details={
                            "target": target,
                            "distance_band": "maintained",
                            "evidence": {
                                "target_visible": True,
                                "follow_distance_stable": True,
                            },
                        },
                    )
            time.sleep(cfg.settle_seconds)

        self.input.release_all()
        return ActionResult(
            action="follow_player",
            status=ActionStatus.TIMEOUT,
            error="FOLLOW_WINDOW_EXPIRED",
            recoverable=True,
            details={"target": target},
        )

    def register_into(self, executor: Any) -> None:
        executor.register("navigate", self.navigate)
        executor.register("collect", self.collect)
        executor.register("interact", self.interact)
        executor.register("follow_player", self.follow_player)
