from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from robloxhive.body.perception import Detection


def _iou(a: Detection, b: Detection) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass(slots=True)
class Track:
    id: int
    detection: Detection
    age: int = 1
    misses: int = 0
    last_time: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class DetectionTracker:
    """Small IoU tracker with no ML dependency."""

    def __init__(self, detector: Callable[[], list[Detection]], iou_threshold: float = 0.25, max_misses: int = 5) -> None:
        self.detector = detector
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self) -> list[Detection]:
        detections = self.detector()
        now = time.monotonic()
        unmatched_tracks = set(self._tracks)
        assigned: list[Detection] = []

        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            candidates = [
                (track_id, _iou(track.detection, det))
                for track_id, track in self._tracks.items()
                if track_id in unmatched_tracks and track.detection.label == det.label
            ]
            track_id = None
            if candidates:
                best_id, best_iou = max(candidates, key=lambda row: row[1])
                if best_iou >= self.iou_threshold:
                    track_id = best_id

            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(track_id, det, last_time=now)
            else:
                track = self._tracks[track_id]
                dt = max(1e-3, now - track.last_time)
                vx = (det.center_x - track.detection.center_x) / dt
                vy = (det.center_y - track.detection.center_y) / dt
                alpha = 0.40
                track.velocity_x = track.velocity_x * (1.0 - alpha) + vx * alpha
                track.velocity_y = track.velocity_y * (1.0 - alpha) + vy * alpha
                track.detection = det
                track.last_time = now
                track.age += 1
                track.misses = 0
                unmatched_tracks.discard(track_id)

            det.track_id = track_id
            track = self._tracks[track_id]
            det.metadata["track_age"] = track.age
            det.metadata["velocity_px_s"] = [round(track.velocity_x, 3), round(track.velocity_y, 3)]
            det.metadata["speed_px_s"] = round((track.velocity_x ** 2 + track.velocity_y ** 2) ** 0.5, 3)
            assigned.append(det)

        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            track.misses += 1
            if track.misses > self.max_misses:
                del self._tracks[track_id]

        return assigned


class PlayerTracker:
    """Track the most stable player/person detection.

    If a custom Roblox model labels avatars as 'player', use that. Generic
    person models may use 'person'. Username-specific tracking can later be
    combined with UI/nameplate OCR.
    """

    def __init__(
        self,
        detection_tracker: DetectionTracker,
        player_labels: tuple[str, ...] = ("player", "person", "avatar"),
    ) -> None:
        self.tracker = detection_tracker
        self.player_labels = tuple(x.lower() for x in player_labels)
        self._selected_track: int | None = None
        self._shape = (0, 0)

    def frame_size(self) -> tuple[int, int]:
        detector_obj = getattr(self.tracker.detector, "__self__", None)
        if detector_obj and hasattr(detector_obj, "frame_size"):
            return detector_obj.frame_size()
        return self._shape

    def find(self, label: str) -> Detection | None:
        requested = label.strip()
        lowered = requested.lower()
        allowed = lowered in self.player_labels or lowered.startswith("player:")
        if not allowed:
            return None

        detections = self.tracker.update()
        players = [d for d in detections if d.label.lower() in self.player_labels]
        if not players:
            return None

        selected = next((d for d in players if d.track_id == self._selected_track), None)
        if selected is None:
            selected = max(players, key=lambda d: (int(d.metadata.get("track_age", 1)), d.confidence, d.area))
            self._selected_track = selected.track_id

        selected.metadata["requested_player"] = requested.removeprefix("player:").strip()
        selected.metadata["tracking_mode"] = "visual-avatar"
        selected.source = "player_tracker"
        return selected
