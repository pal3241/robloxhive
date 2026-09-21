from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WorldSnapshot:
    ts: float = field(default_factory=time.time)
    game_id: int = 0
    frame_size: tuple[int, int] = (0, 0)
    ui_text: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    players: list[dict[str, Any]] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)
    teams: dict[str, list[str]] = field(default_factory=dict)
    enemies: list[str] = field(default_factory=list)
    allies: list[str] = field(default_factory=list)
    navigation: dict[str, Any] = field(default_factory=dict)
    game: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def compact(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "frame_size": list(self.frame_size),
            "ui_text": self.ui_text[:40],
            "entities": self.entities[:60],
            "players": self.players[:30],
            "roles": self.roles,
            "teams": self.teams,
            "enemies": self.enemies[:20],
            "allies": self.allies[:20],
            "navigation": self.navigation,
            "game": self.game,
            "notes": self.notes[:20],
        }


class WorldModel:
    """Build a compact semantic state from perception adapters.

    It deliberately separates observation from interpretation. Detector/OCR
    facts are marked as observed; relations such as ally/enemy/role may be
    populated later by game adapters or the LLM and should carry confidence.
    """

    def __init__(self, executor: Any, game_id: int = 0) -> None:
        self.executor = executor
        self.game_id = int(game_id or 0)
        self.last = WorldSnapshot(game_id=self.game_id)

    @staticmethod
    def _det_to_dict(det: Any) -> dict[str, Any]:
        return {
            "label": str(getattr(det, "label", "unknown")),
            "confidence": round(float(getattr(det, "confidence", 0.0)), 3),
            "box": [
                round(float(getattr(det, "x", 0.0)), 1),
                round(float(getattr(det, "y", 0.0)), 1),
                round(float(getattr(det, "width", 0.0)), 1),
                round(float(getattr(det, "height", 0.0)), 1),
            ],
            "track_id": getattr(det, "track_id", None),
            "source": str(getattr(det, "source", "unknown")),
            "metadata": dict(getattr(det, "metadata", {}) or {}),
        }

    def observe(self) -> WorldSnapshot:
        perception = getattr(self.executor, "perception", None)
        snapshot = WorldSnapshot(game_id=self.game_id)
        if perception is None:
            snapshot.notes.append("perception unavailable")
            self.last = snapshot
            return snapshot

        try:
            snapshot.frame_size = tuple(perception.frame_size())
        except Exception:
            pass

        seen_entity_keys: set[tuple[Any, ...]] = set()
        tracker_supplied_detector = False
        for adapter in getattr(perception, "adapters", []) or []:
            name = type(adapter).__name__
            try:
                if name == "PlayerTracker":
                    detections = list(adapter.tracker.update())
                    tracker_supplied_detector = True
                    labels = set(getattr(adapter, "player_labels", ("player", "person", "avatar")))
                    for det in detections:
                        row = self._det_to_dict(det)
                        is_player = str(row["label"]).lower() in labels
                        row["kind"] = "player" if is_player else "entity"
                        key = ("track", row.get("track_id"), row["label"])
                        if key in seen_entity_keys:
                            continue
                        if is_player:
                            snapshot.players.append(row)
                        snapshot.entities.append(row)
                        seen_entity_keys.add(key)
                elif name == "OnnxYoloDetector":
                    # DetectionTracker already invoked this detector above.
                    # Do not run ONNX twice in one cognitive observation.
                    if tracker_supplied_detector:
                        continue
                    detections = list(adapter.detect())
                    for det in detections:
                        row = self._det_to_dict(det)
                        key = (
                            row["label"],
                            round(row["box"][0] / 10),
                            round(row["box"][1] / 10),
                        )
                        if key not in seen_entity_keys:
                            snapshot.entities.append(row)
                            seen_entity_keys.add(key)
                elif name == "UiTextDetector":
                    detections = list(adapter.scan())
                    for det in detections:
                        text = str(det.label).strip()
                        if text and text not in snapshot.ui_text:
                            snapshot.ui_text.append(text)
            except Exception as exc:
                snapshot.notes.append(f"{name}:{type(exc).__name__}")

        try:
            snapshot.navigation = self.executor.navigation_probe()
        except Exception:
            snapshot.navigation = {}

        try:
            snapshot.game = self.executor.game_status()
        except Exception:
            snapshot.game = {}

        joined_ui = " | ".join(snapshot.ui_text)
        role_match = re.search(
            r"(?i)\byou\s+are(?:\s+the)?\s+([a-z][a-z0-9 _-]{1,28})",
            joined_ui,
        )
        if role_match:
            role = role_match.group(1).strip(" .!:-").lower()
            if role:
                snapshot.roles["self"] = role

        team_patterns = (
            r"(?i)\bteam\s*[:=-]\s*([a-z0-9 _-]{2,24})",
            r"(?i)\byou\s+are\s+on\s+(?:the\s+)?([a-z0-9 _-]{2,24})\s+team\b",
        )
        for pattern in team_patterns:
            match = re.search(pattern, joined_ui)
            if match:
                team = match.group(1).strip(" .!:-").lower()
                if team:
                    snapshot.teams.setdefault(team, []).append("self")
                    break

        for entity in snapshot.entities:
            label = str(entity.get("label") or "").lower()
            track_id = entity.get("track_id")
            ref = f"track:{track_id}" if track_id is not None else label
            if any(token in label for token in ("enemy", "hostile", "opponent")) and ref:
                if ref not in snapshot.enemies:
                    snapshot.enemies.append(ref)
            if any(token in label for token in ("ally", "teammate", "friendly")) and ref:
                if ref not in snapshot.allies:
                    snapshot.allies.append(ref)

        game = snapshot.game or {}
        if game.get("adapter") == "murder_mystery_2":
            role = game.get("role")
            if role:
                snapshot.roles["self"] = str(role)
            murderer_track = game.get("murderer_track_id")
            sheriff_track = game.get("sheriff_track_id")
            if murderer_track is not None:
                snapshot.enemies.append(f"track:{murderer_track}")
            if sheriff_track is not None and role == "murderer":
                snapshot.enemies.append(f"track:{sheriff_track}")

        self.last = snapshot
        return snapshot

    def apply_interpretation(self, interpretation: dict[str, Any]) -> None:
        roles = interpretation.get("roles")
        if isinstance(roles, dict):
            self.last.roles.update({str(k): str(v) for k, v in roles.items()})
        teams = interpretation.get("teams")
        if isinstance(teams, dict):
            self.last.teams.update(
                {
                    str(k): [str(x) for x in v]
                    for k, v in teams.items()
                    if isinstance(v, list)
                }
            )
        for key, target in (("enemies", self.last.enemies), ("allies", self.last.allies)):
            values = interpretation.get(key)
            if isinstance(values, list):
                for value in values:
                    text = str(value)
                    if text not in target:
                        target.append(text)

    def json(self) -> str:
        return json.dumps(self.last.compact(), ensure_ascii=False, separators=(",", ":"))
