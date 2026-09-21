from __future__ import annotations

from collections import defaultdict, deque

from robloxhive.games.murder_mystery_2.models import MM2SceneSnapshot, PlayerObservation


class MM2ThreatModel:
    def __init__(self) -> None:
        self.murderer_score: dict[int, float] = defaultdict(float)
        self.sheriff_score: dict[int, float] = defaultdict(float)
        self.evidence_log: deque[dict] = deque(maxlen=40)

    def update(self, scene: MM2SceneSnapshot) -> None:
        visible = {p.track_id for p in scene.players}
        for track_id in list(self.murderer_score):
            if track_id not in visible:
                self.murderer_score[track_id] *= 0.985
        for track_id in list(self.sheriff_score):
            if track_id not in visible:
                self.sheriff_score[track_id] *= 0.985

        for player in scene.players:
            if player.has_knife:
                self.murderer_score[player.track_id] = max(
                    self.murderer_score[player.track_id], 0.97
                )
            else:
                self.murderer_score[player.track_id] *= 0.995

            if player.has_gun:
                self.sheriff_score[player.track_id] = max(
                    self.sheriff_score[player.track_id], 0.90
                )
            else:
                self.sheriff_score[player.track_id] *= 0.995

            player.murderer_confidence = max(
                player.murderer_confidence,
                self.murderer_score[player.track_id],
            )
            player.sheriff_confidence = max(
                player.sheriff_confidence,
                self.sheriff_score[player.track_id],
            )

    def add_murderer_evidence(
        self,
        track_id: int,
        confidence: float,
        reason: str,
        metadata: dict | None = None,
    ) -> None:
        if track_id < 0:
            return
        confidence = max(0.0, min(1.0, confidence))
        self.murderer_score[track_id] = max(self.murderer_score[track_id], confidence)
        self.evidence_log.append(
            {
                "track_id": track_id,
                "confidence": round(confidence, 3),
                "reason": reason,
                "metadata": metadata or {},
            }
        )

    def murderer(self, scene: MM2SceneSnapshot) -> PlayerObservation | None:
        candidates = [
            p for p in scene.players
            if not p.is_self and self.murderer_score.get(p.track_id, 0.0) > 0.20
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: self.murderer_score[p.track_id])

    def sheriff(self, scene: MM2SceneSnapshot) -> PlayerObservation | None:
        candidates = [
            p for p in scene.players
            if not p.is_self and self.sheriff_score.get(p.track_id, 0.0) > 0.20
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: self.sheriff_score[p.track_id])
