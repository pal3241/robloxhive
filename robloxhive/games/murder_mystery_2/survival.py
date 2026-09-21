from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robloxhive.games.murder_mystery_2.models import MM2SceneSnapshot, PlayerObservation


@dataclass(slots=True)
class SurvivalConfig:
    danger_area_ratio: float = 0.028
    panic_area_ratio: float = 0.060
    strafe_seconds: float = 0.16
    retreat_seconds: float = 0.18


class MM2Survival:
    def __init__(self, input_backend: Any, config: SurvivalConfig | None = None) -> None:
        self.input = input_backend
        self.config = config or SurvivalConfig()

    def evade(
        self,
        threat: PlayerObservation,
        scene: MM2SceneSnapshot,
    ) -> tuple[bool, str]:
        w, h = scene.frame_size
        if w <= 0 or h <= 0:
            return False, "NO_FRAME_SIZE"

        area = threat.area_ratio(scene.frame_size)
        offset = threat.detection.center_x / w - 0.5
        if area < self.config.danger_area_ratio:
            return False, "THREAT_NOT_CLOSE"

        # Strafe away from the threat first; panic adds a backwards hop.
        direction = "left" if offset > 0 else "right"
        self.input.move(direction, self.config.strafe_seconds)
        if area >= self.config.panic_area_ratio:
            self.input.move("back", self.config.retreat_seconds)
            try:
                self.input.key("jump", 0.04)
            except Exception:
                pass
            return True, "PANIC_EVADE"
        return True, "EVADE"
