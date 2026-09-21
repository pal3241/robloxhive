from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def train_mm2_detector(
    dataset_yaml: str | Path = "data/datasets/mm2/yolo/dataset.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    device: str | int | None = None,
    output_dir: str | Path = "runs/mm2",
    model_output_dir: str | Path = "data/models/142823291",
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Training requires the optional package: pip install -e '.[training]'"
        ) from exc

    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        raise FileNotFoundError(dataset_yaml)

    model = YOLO(base_model)
    args: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": max(1, epochs),
        "imgsz": imgsz,
        "batch": batch,
        "project": str(output_dir),
        "name": "detector",
        "exist_ok": True,
    }
    if device is not None:
        args["device"] = device

    result = model.train(**args)
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        best = Path(result.save_dir) / "weights" / "last.pt"

    export_model = YOLO(str(best))
    exported = export_model.export(format="onnx", imgsz=imgsz, simplify=True)
    exported_path = Path(str(exported))

    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    final_model = model_output_dir / "detector.onnx"
    shutil.copy2(exported_path, final_model)

    raw = dataset_yaml.read_text(encoding="utf-8")
    names: list[str] = []
    inside = False
    for line in raw.splitlines():
        if line.strip() == "names:":
            inside = True
            continue
        if inside:
            stripped = line.strip()
            if not stripped or ":" not in stripped:
                continue
            _, value = stripped.split(":", 1)
            names.append(value.strip())
    labels = model_output_dir / "labels.txt"
    labels.write_text("\n".join(names) + "\n", encoding="utf-8")

    summary = {
        "best_weights": str(best),
        "exported_onnx": str(final_model),
        "labels": str(labels),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
    }
    (model_output_dir / "training.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
