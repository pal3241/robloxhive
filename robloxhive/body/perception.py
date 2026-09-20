from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.width * self.height)


class FrameSource(Protocol):
    def capture(self) -> np.ndarray: ...


class PerceptionAdapter(Protocol):
    def find(self, label: str) -> Detection | None: ...

    def frame_size(self) -> tuple[int, int]: ...


class TemplateVision:
    """Lightweight template-matching perception for generic Roblox skills.

    Templates live under data/templates/<game_id>/<label>/*.png. This is not
    intended to replace a future detector/SLAM stack; it provides a practical
    low-cost baseline that works without an LLM in the realtime loop.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        template_root: str | Path,
        threshold: float = 0.72,
    ) -> None:
        self.frame_source = frame_source
        self.template_root = Path(template_root)
        self.threshold = threshold
        self._shape: tuple[int, int] = (0, 0)

    def frame_size(self) -> tuple[int, int]:
        return self._shape

    def find(self, label: str) -> Detection | None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("TemplateVision requires opencv-python") from exc

        frame = self.frame_source.capture()
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        self._shape = (w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        directory = self.template_root / label
        candidates = []
        if directory.is_file():
            candidates = [directory]
        elif directory.exists():
            candidates = sorted(
                p for p in directory.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
        else:
            direct = self.template_root / f"{label}.png"
            if direct.exists():
                candidates = [direct]

        best: Detection | None = None
        for path in candidates:
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            th, tw = template.shape[:2]
            if th > h or tw > w:
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if score < self.threshold:
                continue
            detection = Detection(
                label=label,
                confidence=float(score),
                x=float(loc[0]),
                y=float(loc[1]),
                width=float(tw),
                height=float(th),
            )
            if best is None or detection.confidence > best.confidence:
                best = detection
        return best
