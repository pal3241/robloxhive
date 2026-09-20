from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robloxhive.body.perception import Detection, FrameSource


class OnnxYoloDetector:
    """YOLOv8-style ONNX detector running locally on the Windows Body.

    Expected output is a common YOLO tensor shaped [1, N, 4+C] or
    [1, 4+C, N]. Labels come from a JSON/TXT file or a supplied list.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        model_path: str | Path,
        labels: list[str] | None = None,
        labels_path: str | Path | None = None,
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX detector requires onnxruntime") from exc

        self.frame_source = frame_source
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.labels = labels or self._load_labels(labels_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self._shape = (0, 0)
        self._last: list[Detection] = []

    @staticmethod
    def _load_labels(path: str | Path | None) -> list[str]:
        if path is None:
            return []
        p = Path(path)
        if not p.exists():
            return []
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                pairs = sorted(
                    ((int(k), str(v)) for k, v in data.items()),
                    key=lambda row: row[0],
                )
                return [value for _, value in pairs]
            if isinstance(data, list):
                return [str(x) for x in data]
        return [
            line.strip()
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def frame_size(self) -> tuple[int, int]:
        return self._shape

    @staticmethod
    def _letterbox(frame: Any, size: int):
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.astype("float32") / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]
        return blob, scale, dx, dy

    def detect(self) -> list[Detection]:
        import cv2
        import numpy as np

        frame = self.frame_source.capture()
        if frame is None or frame.size == 0:
            self._last = []
            return []
        h, w = frame.shape[:2]
        self._shape = (w, h)

        blob, scale, dx, dy = self._letterbox(frame, self.input_size)
        raw = self.session.run(None, {self.input_name: blob})[0]
        pred = np.asarray(raw)
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            self._last = []
            return []
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 256:
            pred = pred.T
        if pred.shape[1] < 5:
            self._last = []
            return []

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        for row in pred:
            cx, cy, bw, bh = map(float, row[:4])
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.confidence:
                continue

            x1 = (cx - bw / 2 - dx) / scale
            y1 = (cy - bh / 2 - dy) / scale
            ww = bw / scale
            hh = bh / scale
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            ww = max(1.0, min(float(w) - x1, ww))
            hh = max(1.0, min(float(h) - y1, hh))
            boxes.append([int(x1), int(y1), int(ww), int(hh)])
            scores.append(score)
            class_ids.append(class_id)

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.iou_threshold)
        if len(keep) == 0:
            self._last = []
            return []

        indices = [int(x) for x in np.array(keep).reshape(-1)]
        detections: list[Detection] = []
        for index in indices:
            x, y, bw, bh = boxes[index]
            cid = class_ids[index]
            label = self.labels[cid] if cid < len(self.labels) else f"class_{cid}"
            detections.append(
                Detection(
                    label=label,
                    confidence=scores[index],
                    x=float(x),
                    y=float(y),
                    width=float(bw),
                    height=float(bh),
                    source="onnx",
                    metadata={"class_id": cid},
                )
            )
        self._last = detections
        return detections

    def find(self, label: str) -> Detection | None:
        wanted = label.strip().lower()
        detections = self.detect()
        exact = [d for d in detections if d.label.lower() == wanted]
        if exact:
            return max(exact, key=lambda d: d.confidence)
        partial = [
            d for d in detections
            if wanted in d.label.lower() or d.label.lower() in wanted
        ]
        return max(partial, key=lambda d: d.confidence) if partial else None

    def last_detections(self) -> list[Detection]:
        return list(self._last)
