from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from robloxhive.games.murder_mystery_2.models import MM2SceneSnapshot, PlayerObservation


@dataclass(slots=True)
class MM2CombatConfig:
    sheriff_fire_confidence: float = 0.93
    melee_area_ratio: float = 0.055
    throw_min_area_ratio: float = 0.006
    fire_cooldown_s: float = 1.15
    throw_cooldown_s: float = 1.15
    melee_cooldown_s: float = 0.22
    crowd_padding_px: float = 34.0
    sheriff_lead_s: float = 0.08
    knife_throw_lead_s: float = 0.20
    max_lead_px: float = 140.0


class MM2Combat:
    def __init__(self, input_backend: Any, config: MM2CombatConfig | None = None) -> None:
        self.input = input_backend
        self.config = config or MM2CombatConfig()
        self._last_fire = 0.0
        self._last_throw = 0.0
        self._last_melee = 0.0

    @staticmethod
    def _center(player: PlayerObservation) -> tuple[float, float]:
        d = player.detection
        return float(d.center_x), float(d.y + d.height * 0.38)

    def _lead_point(
        self,
        player: PlayerObservation,
        scene: MM2SceneSnapshot,
        lead_s: float,
    ) -> tuple[int, int]:
        x, y = self._center(player)
        velocity = player.detection.metadata.get("velocity_px_s", [0.0, 0.0])
        try:
            vx, vy = float(velocity[0]), float(velocity[1])
        except (TypeError, ValueError, IndexError):
            vx, vy = 0.0, 0.0

        lead_x = max(-self.config.max_lead_px, min(self.config.max_lead_px, vx * lead_s))
        lead_y = max(-self.config.max_lead_px, min(self.config.max_lead_px, vy * lead_s))
        w, h = scene.frame_size
        px = int(max(0, min(max(0, w - 1), x + lead_x))) if w > 0 else int(x + lead_x)
        py = int(max(0, min(max(0, h - 1), y + lead_y))) if h > 0 else int(y + lead_y)
        return px, py

    def _crowd_clear(
        self,
        target: PlayerObservation,
        scene: MM2SceneSnapshot,
        aim_point: tuple[int, int],
    ) -> bool:
        tx, ty = aim_point
        pad = self.config.crowd_padding_px
        for other in scene.players:
            if other.track_id == target.track_id:
                continue
            d = other.detection
            if d.x - pad <= tx <= d.x + d.width + pad and d.y - pad <= ty <= d.y + d.height + pad:
                return False
        return True

    def sheriff_fire(self, target: PlayerObservation, scene: MM2SceneSnapshot) -> tuple[bool, str]:
        now = time.monotonic()
        if target.murderer_confidence < self.config.sheriff_fire_confidence:
            return False, "MURDERER_CONFIDENCE_TOO_LOW"
        aim_point = self._lead_point(target, scene, self.config.sheriff_lead_s)
        if not self._crowd_clear(target, scene, aim_point):
            return False, "FRIENDLY_FIRE_RISK"
        if now - self._last_fire < self.config.fire_cooldown_s:
            return False, "GUN_COOLDOWN"

        x, y = aim_point
        self.input.key("1", 0.05)
        self.input.aim_client(x, y)
        self.input.click_client(x, y, "left")
        self._last_fire = now
        return True, "SHOT_FIRED"

    def murderer_attack(
        self,
        target: PlayerObservation,
        scene: MM2SceneSnapshot,
    ) -> tuple[bool, str, str]:
        w, h = scene.frame_size
        area_ratio = target.area_ratio((w, h))
        now = time.monotonic()
        center_x, center_y = self._center(target)

        self.input.key("1", 0.05)

        if area_ratio >= self.config.melee_area_ratio:
            x, y = int(center_x), int(center_y)
            self.input.aim_client(x, y)
            if now - self._last_melee < self.config.melee_cooldown_s:
                return False, "MELEE_COOLDOWN", "melee"
            self.input.click_client(x, y, "left")
            self._last_melee = now
            return True, "KNIFE_MELEE", "melee"

        if area_ratio >= self.config.throw_min_area_ratio:
            if now - self._last_throw < self.config.throw_cooldown_s:
                return False, "THROW_COOLDOWN", "throw"
            x, y = self._lead_point(target, scene, self.config.knife_throw_lead_s)
            self.input.aim_client(x, y)
            self.input.click_client(x, y, "right")
            self._last_throw = now
            return True, "KNIFE_THROW", "throw"

        return False, "TARGET_TOO_FAR", "approach"
