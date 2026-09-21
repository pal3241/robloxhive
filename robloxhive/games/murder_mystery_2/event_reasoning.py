from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from robloxhive.body.perception import Detection
from robloxhive.games.murder_mystery_2.models import MM2SceneSnapshot
from robloxhive.games.murder_mystery_2.threat import MM2ThreatModel


@dataclass(slots=True)
class KillEvent:
    body_x: float
    body_y: float
    suspect_track_id: int | None
    confidence: float
    reason: str


def _center_distance(a: Detection, b: Detection) -> float:
    return math.hypot(a.center_x - b.center_x, a.center_y - b.center_y)


class WitnessedKillReasoner:
    """Infer extra Murderer evidence from newly appeared bodies.

    A body event is only used as supporting evidence; knife possession remains
    the strongest signal. This avoids turning proximity alone into certainty.
    """

    def __init__(self) -> None:
        self._previous_bodies: list[Detection] = []
        self._initialized = False
        self.events: deque[KillEvent] = deque(maxlen=24)

    def _is_new_body(self, body: Detection) -> bool:
        if not self._previous_bodies:
            return True
        return all(_center_distance(body, old) > max(body.width, body.height, 45.0) * 0.65 for old in self._previous_bodies)

    def update(self, scene: MM2SceneSnapshot, threats: MM2ThreatModel) -> list[KillEvent]:
        if not self._initialized:
            self._previous_bodies = list(scene.bodies)
            self._initialized = True
            return []

        new_events: list[KillEvent] = []
        w, h = scene.frame_size
        max_distance = max(80.0, math.hypot(w, h) * 0.18) if w > 0 and h > 0 else 160.0

        for body in scene.bodies:
            if not self._is_new_body(body):
                continue

            nearby = [
                player for player in scene.players
                if not player.is_self and _center_distance(player.detection, body) <= max_distance
            ]
            if not nearby:
                event = KillEvent(body.center_x, body.center_y, None, 0.0, "body_appeared_no_visible_suspect")
                self.events.append(event)
                new_events.append(event)
                continue

            # Knife holder is compelling. Otherwise only reinforce an already-suspicious nearby player.
            knife_holders = [p for p in nearby if p.has_knife]
            if knife_holders:
                suspect = min(knife_holders, key=lambda p: _center_distance(p.detection, body))
                confidence = max(0.985, suspect.murderer_confidence)
                reason = "new_body_near_visible_knife_holder"
                threats.add_murderer_evidence(
                    suspect.track_id,
                    confidence,
                    reason,
                    {"body": [round(body.center_x, 1), round(body.center_y, 1)]},
                )
            else:
                suspect = max(
                    nearby,
                    key=lambda p: threats.murderer_score.get(p.track_id, 0.0),
                )
                prior = threats.murderer_score.get(suspect.track_id, 0.0)
                if prior >= 0.45:
                    confidence = min(0.92, prior + 0.12)
                    reason = "new_body_reinforces_existing_suspect"
                    threats.add_murderer_evidence(
                        suspect.track_id,
                        confidence,
                        reason,
                        {"prior": round(prior, 3)},
                    )
                else:
                    suspect = None
                    confidence = 0.0
                    reason = "body_appeared_proximity_insufficient"

            event = KillEvent(
                body.center_x,
                body.center_y,
                suspect.track_id if suspect else None,
                confidence,
                reason,
            )
            self.events.append(event)
            new_events.append(event)

        self._previous_bodies = list(scene.bodies)
        return new_events

    def recent(self) -> list[dict]:
        return [
            {
                "body": [round(event.body_x, 1), round(event.body_y, 1)],
                "suspect_track_id": event.suspect_track_id,
                "confidence": round(event.confidence, 3),
                "reason": event.reason,
            }
            for event in list(self.events)[-10:]
        ]
