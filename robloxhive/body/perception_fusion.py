from __future__ import annotations

from typing import Any

from robloxhive.body.perception import Detection, PerceptionAdapter


class CompositePerception:
    """Try multiple perception sources and return the highest-confidence hit."""

    def __init__(self, adapters: list[PerceptionAdapter]) -> None:
        self.adapters = adapters
        self._shape = (0, 0)
        self.last_source: str | None = None

    def frame_size(self) -> tuple[int, int]:
        for adapter in self.adapters:
            shape = adapter.frame_size()
            if shape[0] > 0 and shape[1] > 0:
                self._shape = shape
                return shape
        return self._shape

    def find(self, label: str) -> Detection | None:
        hits: list[Detection] = []
        for adapter in self.adapters:
            try:
                hit = adapter.find(label)
            except Exception:
                continue
            if hit is not None:
                hits.append(hit)
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
            "frame_size": self.frame_size(),
        }
