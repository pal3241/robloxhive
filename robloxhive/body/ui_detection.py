from __future__ import annotations

import re
from typing import Any

from robloxhive.body.perception import Detection, FrameSource


class UiTextDetector:
    """OCR-backed detector for visible Roblox UI text/nameplates.

    Uses pytesseract when installed. It is optional; CompositePerception simply
    continues to ONNX/template sources when OCR is unavailable.
    """

    def __init__(self, frame_source: FrameSource, confidence: float = 45.0) -> None:
        self.frame_source = frame_source
        self.confidence = confidence
        self._shape = (0, 0)
        self.available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            import pytesseract
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def frame_size(self) -> tuple[int, int]:
        return self._shape

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def scan(self) -> list[Detection]:
        if not self.available:
            return []
        import cv2
        import pytesseract
        from pytesseract import Output

        frame = self.frame_source.capture()
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        self._shape = (w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        data: dict[str, Any] = pytesseract.image_to_data(gray, output_type=Output.DICT)

        out: list[Detection] = []
        for i, text in enumerate(data.get("text", [])):
            text = str(text).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                continue
            if conf < self.confidence:
                continue
            out.append(
                Detection(
                    label=text,
                    confidence=min(1.0, conf / 100.0),
                    x=float(data["left"][i]),
                    y=float(data["top"][i]),
                    width=float(data["width"][i]),
                    height=float(data["height"][i]),
                    source="ocr",
                    metadata={"text": text},
                )
            )
        return out

    def find_exact(self, text: str) -> Detection | None:
        wanted = re.sub(r"[^a-z0-9_]", "", text.strip().lower().lstrip("@"))
        if not wanted:
            return None
        matches = []
        for det in self.scan():
            candidate = re.sub(r"[^a-z0-9_]", "", str(det.label).strip().lower().lstrip("@"))
            if candidate == wanted:
                matches.append(det)
        if not matches:
            return None
        hit = max(matches, key=lambda d: d.confidence)
        hit.metadata["username_verified"] = True
        hit.metadata["tracking_mode"] = "nameplate_ocr_only"
        return hit

    def find(self, label: str) -> Detection | None:
        wanted = self._norm(label)
        if not wanted:
            return None
        matches = []
        for det in self.scan():
            text = self._norm(det.label)
            if wanted == text or wanted in text or text in wanted:
                matches.append(det)
        return max(matches, key=lambda d: d.confidence) if matches else None
