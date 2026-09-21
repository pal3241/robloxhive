from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CLASSES = ["player", "knife", "gun", "dropped_gun", "dead_player"]


@dataclass(slots=True)
class DatasetSample:
    id: str
    image_path: str
    annotation_path: str
    width: int
    height: int
    boxes: int
    approved_boxes: int


class MM2DatasetRecorder:
    """Capture MM2 frames with detector proposals for later human review."""

    def __init__(
        self,
        frame_source: Any,
        root: str | Path = "data/datasets/mm2",
        classes: list[str] | None = None,
    ) -> None:
        self.frame_source = frame_source
        self.root = Path(root)
        self.classes = classes or list(DEFAULT_CLASSES)
        self.raw_images = self.root / "raw" / "images"
        self.raw_annotations = self.root / "raw" / "annotations"
        self.raw_images.mkdir(parents=True, exist_ok=True)
        self.raw_annotations.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        detections: list[Any] | None = None,
        note: str | None = None,
        auto_approve_confidence: float | None = None,
    ) -> DatasetSample:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Dataset capture requires opencv-python") from exc

        frame = self.frame_source.capture()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise RuntimeError("DATASET_CAPTURE_EMPTY_FRAME")
        h, w = frame.shape[:2]
        sample_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        image_path = self.raw_images / f"{sample_id}.png"
        annotation_path = self.raw_annotations / f"{sample_id}.json"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError("DATASET_IMAGE_WRITE_FAILED")

        boxes: list[dict[str, Any]] = []
        for det in detections or []:
            label = str(getattr(det, "label", "")).lower().strip()
            if label not in self.classes:
                continue
            confidence = float(getattr(det, "confidence", 0.0))
            approved = (
                auto_approve_confidence is not None
                and confidence >= auto_approve_confidence
            )
            boxes.append(
                {
                    "label": label,
                    "x": float(getattr(det, "x", 0.0)),
                    "y": float(getattr(det, "y", 0.0)),
                    "width": float(getattr(det, "width", 0.0)),
                    "height": float(getattr(det, "height", 0.0)),
                    "confidence": confidence,
                    "source": str(getattr(det, "source", "unknown")),
                    "approved": approved,
                }
            )

        payload = {
            "id": sample_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image": str(image_path),
            "width": w,
            "height": h,
            "classes": self.classes,
            "note": note,
            "boxes": boxes,
        }
        annotation_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return DatasetSample(
            id=sample_id,
            image_path=str(image_path),
            annotation_path=str(annotation_path),
            width=w,
            height=h,
            boxes=len(boxes),
            approved_boxes=sum(1 for box in boxes if box["approved"]),
        )

    def list_samples(self, limit: int = 100) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in sorted(self.raw_annotations.glob("*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            boxes = data.get("boxes", [])
            samples.append(
                {
                    "id": data.get("id", path.stem),
                    "image": data.get("image"),
                    "annotation": str(path),
                    "width": data.get("width"),
                    "height": data.get("height"),
                    "boxes": len(boxes),
                    "approved_boxes": sum(1 for b in boxes if b.get("approved")),
                    "note": data.get("note"),
                }
            )
        return samples

    def review(
        self,
        sample_id: str,
        boxes: list[dict[str, Any]],
        note: str | None = None,
    ) -> dict[str, Any]:
        path = self.raw_annotations / f"{sample_id}.json"
        if not path.exists():
            raise FileNotFoundError(sample_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        clean: list[dict[str, Any]] = []
        for box in boxes:
            label = str(box.get("label", "")).lower().strip()
            if label not in self.classes:
                continue
            clean.append(
                {
                    "label": label,
                    "x": max(0.0, float(box.get("x", 0.0))),
                    "y": max(0.0, float(box.get("y", 0.0))),
                    "width": max(1.0, float(box.get("width", 1.0))),
                    "height": max(1.0, float(box.get("height", 1.0))),
                    "confidence": float(box.get("confidence", 1.0)),
                    "source": str(box.get("source", "manual")),
                    "approved": bool(box.get("approved", True)),
                }
            )
        data["boxes"] = clean
        if note is not None:
            data["note"] = note
        data["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data

    def export_yolo(
        self,
        validation_ratio: float = 0.2,
        seed: int = 42,
    ) -> dict[str, Any]:
        yolo_root = self.root / "yolo"
        if yolo_root.exists():
            shutil.rmtree(yolo_root)
        for split in ("train", "val"):
            (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        eligible: list[dict[str, Any]] = []
        for ann_path in sorted(self.raw_annotations.glob("*.json")):
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            approved = [box for box in data.get("boxes", []) if box.get("approved")]
            if not approved:
                continue
            image_path = Path(data["image"])
            if not image_path.exists():
                continue
            data["_approved"] = approved
            eligible.append(data)

        rng = random.Random(seed)
        rng.shuffle(eligible)
        val_count = int(round(len(eligible) * max(0.0, min(0.5, validation_ratio))))
        if len(eligible) >= 5:
            val_count = max(1, val_count)
        val_ids = {item["id"] for item in eligible[:val_count]}

        counts = {"train": 0, "val": 0, "boxes": 0}
        class_to_id = {name: idx for idx, name in enumerate(self.classes)}

        for data in eligible:
            split = "val" if data["id"] in val_ids else "train"
            src = Path(data["image"])
            dst = yolo_root / "images" / split / src.name
            shutil.copy2(src, dst)

            width = float(data["width"])
            height = float(data["height"])
            lines: list[str] = []
            for box in data["_approved"]:
                class_id = class_to_id[box["label"]]
                cx = (float(box["x"]) + float(box["width"]) / 2.0) / width
                cy = (float(box["y"]) + float(box["height"]) / 2.0) / height
                bw = float(box["width"]) / width
                bh = float(box["height"]) / height
                cx, cy = min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))
                bw, bh = min(1.0, max(1e-6, bw)), min(1.0, max(1e-6, bh))
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counts["boxes"] += 1

            label_path = yolo_root / "labels" / split / f"{src.stem}.txt"
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts[split] += 1

        yaml_path = yolo_root / "dataset.yaml"
        names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(self.classes))
        yaml_path.write_text(
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            f"{names_yaml}\n",
            encoding="utf-8",
        )
        return {
            "root": str(yolo_root),
            "dataset_yaml": str(yaml_path),
            **counts,
        }

    def status(self) -> dict[str, Any]:
        samples = self.list_samples(limit=10000)
        return {
            "root": str(self.root),
            "classes": self.classes,
            "samples": len(samples),
            "approved_samples": sum(1 for s in samples if s["approved_boxes"] > 0),
            "approved_boxes": sum(int(s["approved_boxes"]) for s in samples),
        }
