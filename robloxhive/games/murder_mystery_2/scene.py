from __future__ import annotations

import math
from typing import Any

from robloxhive.body.perception import Detection
from robloxhive.games.murder_mystery_2.models import MM2SceneSnapshot, PlayerObservation


PLAYER_LABELS = {"player", "person", "avatar"}
KNIFE_LABELS = {"knife", "knife_held", "murderer_knife"}
GUN_LABELS = {"gun", "revolver", "sheriff_gun"}
DROPPED_GUN_LABELS = {"dropped_gun", "gun_drop"}
BODY_LABELS = {"body", "dead_player", "corpse"}


def _distance(a: Detection, b: Detection) -> float:
    return math.hypot(a.center_x - b.center_x, a.center_y - b.center_y)


class MM2SceneReader:
    def __init__(self, perception: Any) -> None:
        self.perception = perception
        self._player_tracker = None
        self._detector = None
        self._ocr = None
        for adapter in getattr(perception, "adapters", []):
            name = type(adapter).__name__
            if name == "PlayerTracker":
                self._player_tracker = adapter
            elif name == "OnnxYoloDetector":
                self._detector = adapter
            elif name == "UiTextDetector":
                self._ocr = adapter

    def _detections(self) -> list[Detection]:
        if self._player_tracker is not None:
            try:
                return list(self._player_tracker.tracker.update())
            except Exception:
                pass
        if self._detector is not None:
            try:
                return list(self._detector.detect())
            except Exception:
                pass
        return []

    def _texts(self) -> list[str]:
        if self._ocr is None:
            return []
        try:
            return [d.label for d in self._ocr.scan()]
        except Exception:
            return []

    @staticmethod
    def _associate_weapon(
        players: list[PlayerObservation],
        weapon: Detection,
        frame_size: tuple[int, int],
    ) -> PlayerObservation | None:
        if not players:
            return None
        w, h = frame_size
        max_distance = max(45.0, math.hypot(w, h) * 0.12)
        nearest = min(players, key=lambda p: _distance(p.detection, weapon))
        return nearest if _distance(nearest.detection, weapon) <= max_distance else None

    def observe(self) -> MM2SceneSnapshot:
        detections = self._detections()
        frame_size = self.perception.frame_size()
        players: list[PlayerObservation] = []

        for d in detections:
            if d.label.lower() in PLAYER_LABELS:
                players.append(
                    PlayerObservation(
                        track_id=int(d.track_id or -1),
                        detection=d,
                    )
                )

        w, h = frame_size
        if w > 0 and h > 0 and players:
            self_candidates = [
                p for p in players
                if abs(p.detection.center_x / w - 0.5) <= 0.22
                and (p.detection.y + p.detection.height) / h >= 0.62
            ]
            if self_candidates:
                own = max(self_candidates, key=lambda p: p.detection.area)
                own.is_self = True

        knives = [d for d in detections if d.label.lower() in KNIFE_LABELS]
        guns = [d for d in detections if d.label.lower() in GUN_LABELS]
        dropped_guns = [d for d in detections if d.label.lower() in DROPPED_GUN_LABELS]
        bodies = [d for d in detections if d.label.lower() in BODY_LABELS]

        for knife in knives:
            owner = self._associate_weapon(players, knife, frame_size)
            if owner:
                owner.has_knife = True
                owner.murderer_confidence = max(owner.murderer_confidence, 0.97)

        for gun in guns:
            owner = self._associate_weapon(players, gun, frame_size)
            if owner:
                owner.has_gun = True
                owner.sheriff_confidence = max(owner.sheriff_confidence, 0.90)

        return MM2SceneSnapshot(
            frame_size=frame_size,
            players=players,
            knives=knives,
            guns=guns,
            dropped_guns=dropped_guns,
            bodies=bodies,
            ui_text=self._texts(),
            raw_detections=detections,
        )
