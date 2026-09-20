from __future__ import annotations

from typing import Any

from robloxhive.body.perception import Detection, PerceptionAdapter


class CompositePerception:
    """Ordered perception fusion with early-exit for realtime performance.

    Preferred order is player tracker -> ONNX -> OCR -> template. A strong hit
    exits early so the Windows Body does not run every expensive backend on
    every realtime skill iteration.
    """

    def __init__(
        self,
        adapters: list[PerceptionAdapter],
        early_exit_confidence: float = 0.72,
    ) -> None:
        self.adapters = adapters
        self.early_exit_confidence = early_exit_confidence
        self._shape = (0, 0)
        self.last_source: str | None = None
        self.last_attempts: list[str] = []

    def frame_size(self) -> tuple[int, int]:
        for adapter in self.adapters:
            shape = adapter.frame_size()
            if shape[0] > 0 and shape[1] > 0:
                self._shape = shape
                return shape
        return self._shape

    def find(self, label: str) -> Detection | None:
        self.last_attempts = []
        hits: list[Detection] = []
        player_target = label.strip().lower().startswith("player:")

        for adapter in self.adapters:
            name = type(adapter).__name__
            self.last_attempts.append(name)
            try:
                hit = adapter.find(label)
            except Exception:
                continue
            if hit is None:
                continue

            hits.append(hit)

            # PlayerTracker is the dedicated route for follow_player.
            if player_target and hit.source == "player_tracker":
                self.last_source = hit.source
                return hit

            # A strong ONNX/OCR hit is enough; avoid template work afterward.
            if hit.confidence >= self.early_exit_confidence and hit.source in {"onnx", "ocr"}:
                self.last_source = hit.source
                return hit

        if not hits:
            self.last_source = None
            return None

        source_priority = {
            "player_tracker": 0.10,
            "onnx": 0.08,
            "ocr": 0.04,
            "template": 0.0,
        }
        best = max(
            hits,
            key=lambda d: d.confidence + source_priority.get(d.source, 0.0),
        )
        self.last_source = best.source
        return best

    def diagnostics(self) -> dict[str, Any]:
        return {
            "adapters": [type(adapter).__name__ for adapter in self.adapters],
            "last_source": self.last_source,
            "last_attempts": list(self.last_attempts),
            "frame_size": self.frame_size(),
            "early_exit_confidence": self.early_exit_confidence,
        }
