from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MotionConfig:
    min_change_ratio: float = 0.018
    pixel_threshold: int = 18
    roi_top_ratio: float = 0.25
    roi_bottom_ratio: float = 0.90


class MotionVerifier:
    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()
        self.last_score = 0.0

    def score(self, before, after) -> float:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Motion verification requires opencv-python") from exc

        if before is None or after is None or before.shape != after.shape:
            self.last_score = 0.0
            return 0.0

        h = before.shape[0]
        y0 = int(h * self.config.roi_top_ratio)
        y1 = int(h * self.config.roi_bottom_ratio)
        a = before[y0:y1]
        b = after[y0:y1]
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
        diff = cv2.absdiff(ga, gb)
        ratio = float((diff >= self.config.pixel_threshold).mean())
        self.last_score = ratio
        return ratio

    def moved(self, before, after) -> bool:
        return self.score(before, after) >= self.config.min_change_ratio
