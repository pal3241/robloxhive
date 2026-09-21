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
        lines: dict[tuple[int, int, int], list[tuple[str, float, float, float, float, float]]] = {}
        for i, text in enumerate(data.get("text", [])):
            text = str(text).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
                left = float(data["left"][i])
                top = float(data["top"][i])
                width = float(data["width"][i])
                height = float(data["height"][i])
            except (TypeError, ValueError, IndexError):
                continue
            if conf < self.confidence:
                continue
            word = Detection(
                label=text,
                confidence=min(1.0, conf / 100.0),
                x=left,
                y=top,
                width=width,
                height=height,
                source="ocr",
                metadata={"text": text, "ocr_level": "word"},
            )
            out.append(word)
            try:
                line_key = (
                    int(data.get("block_num", [0])[i]),
                    int(data.get("par_num", [0])[i]),
                    int(data.get("line_num", [0])[i]),
                )
            except (TypeError, ValueError, IndexError):
                line_key = (0, 0, i)
            lines.setdefault(line_key, []).append((text, conf, left, top, width, height))

        for words in lines.values():
            if len(words) < 2:
                continue
            words.sort(key=lambda row: row[2])
            text = " ".join(row[0] for row in words).strip()
            x1 = min(row[2] for row in words)
            y1 = min(row[3] for row in words)
            x2 = max(row[2] + row[4] for row in words)
            y2 = max(row[3] + row[5] for row in words)
            avg_conf = sum(row[1] for row in words) / len(words)
            out.append(
                Detection(
                    label=text,
                    confidence=min(1.0, avg_conf / 100.0),
                    x=x1,
                    y=y1,
                    width=max(1.0, x2 - x1),
                    height=max(1.0, y2 - y1),
                    source="ocr",
                    metadata={"text": text, "ocr_level": "line"},
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
