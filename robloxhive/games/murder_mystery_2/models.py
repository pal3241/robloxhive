from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from robloxhive.body.perception import Detection


class MM2Role(str, Enum):
    UNKNOWN = "unknown"
    INNOCENT = "innocent"
    SHERIFF = "sheriff"
    HERO = "hero"
    MURDERER = "murderer"


class RoundPhase(str, Enum):
    LOBBY = "lobby"
    ROLE_REVEAL = "role_reveal"
    ROUND = "round"
    ROUND_END = "round_end"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class PlayerObservation:
    track_id: int
    detection: Detection
    has_knife: bool = False
    has_gun: bool = False
    alive: bool = True
    is_self: bool = False
    murderer_confidence: float = 0.0
    sheriff_confidence: float = 0.0

    def area_ratio(self, frame_size: tuple[int, int]) -> float:
        w, h = frame_size
        return self.detection.area / float(w * h) if w > 0 and h > 0 else 0.0


@dataclass(slots=True)
class MM2SceneSnapshot:
    frame_size: tuple[int, int]
    players: list[PlayerObservation] = field(default_factory=list)
    knives: list[Detection] = field(default_factory=list)
    guns: list[Detection] = field(default_factory=list)
    dropped_guns: list[Detection] = field(default_factory=list)
    bodies: list[Detection] = field(default_factory=list)
    ui_text: list[str] = field(default_factory=list)
    raw_detections: list[Detection] = field(default_factory=list)

    def player(self, track_id: int | None) -> PlayerObservation | None:
        if track_id is None:
            return None
        return next((p for p in self.players if p.track_id == track_id), None)

    def nearest_player_to_screen_center(self) -> PlayerObservation | None:
        w, h = self.frame_size
        if not self.players or w <= 0 or h <= 0:
            return None
        cx, cy = w / 2.0, h / 2.0
        return min(
            self.players,
            key=lambda p: (p.detection.center_x - cx) ** 2 + (p.detection.center_y - cy) ** 2,
        )


@dataclass(slots=True)
class MM2State:
    enabled: bool = True
    role: MM2Role = MM2Role.UNKNOWN
    role_confidence: float = 0.0
    role_override: MM2Role | None = None
    phase: RoundPhase = RoundPhase.UNKNOWN
    murderer_track_id: int | None = None
    murderer_confidence: float = 0.0
    sheriff_track_id: int | None = None
    target_track_id: int | None = None
    mode: str = "observe"
    last_action: str = "idle"
    last_reason: str | None = None
    shots: int = 0
    knife_swings: int = 0
    knife_throws: int = 0
    survival_moves: int = 0
    ticks: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
