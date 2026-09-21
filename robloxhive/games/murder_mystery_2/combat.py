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


class MM2Combat:
    def __init__(self, input_backend: Any, config: MM2CombatConfig | None = None) -> None:
        self.input = input_backend
        self.config = config or MM2CombatConfig()
        self._last_fire = 0.0
        self._last_throw = 0.0
        self._last_melee = 0.0

    @staticmethod
    def _center(player: PlayerObservation) -> tuple[int, int]:
        d = player.detection
        # Aim upper torso rather than feet.
        return int(d.center_x), int(d.y + d.height * 0.38)

    def _crowd_clear(self, target: PlayerObservation, scene: MM2SceneSnapshot) -> bool:
        tx, ty = self._center(target)
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
        if not self._crowd_clear(target, scene):
            return False, "FRIENDLY_FIRE_RISK"
        if now - self._last_fire < self.config.fire_cooldown_s:
            return False, "GUN_COOLDOWN"

        x, y = self._center(target)
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
        x, y = self._center(target)

        self.input.key("1", 0.05)
        self.input.aim_client(x, y)

        if area_ratio >= self.config.melee_area_ratio:
            if now - self._last_melee < self.config.melee_cooldown_s:
                return False, "MELEE_COOLDOWN", "melee"
            self.input.click_client(x, y, "left")
            self._last_melee = now
            return True, "KNIFE_MELEE", "melee"

        if area_ratio >= self.config.throw_min_area_ratio:
            if now - self._last_throw < self.config.throw_cooldown_s:
                return False, "THROW_COOLDOWN", "throw"
            self.input.click_client(x, y, "right")
            self._last_throw = now
            return True, "KNIFE_THROW", "throw"

        return False, "TARGET_TOO_FAR", "approach"
