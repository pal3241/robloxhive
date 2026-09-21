from __future__ import annotations

from pathlib import Path

from robloxhive.body.capture import Win32WindowCapture
from robloxhive.body.generic_skills import GenericVisualSkills, SkillConfig
from robloxhive.body.input_win32 import Win32MessageInput, create_input_backend
from robloxhive.body.navigation.navigator import LocalNavigator
from robloxhive.body.perception import TemplateVision
from robloxhive.body.perception_fusion import CompositePerception
from robloxhive.body.skills import SkillExecutor
from robloxhive.body.ui_detection import UiTextDetector


def create_generic_skill_executor(
    hwnd: int,
    game_id: int | str,
    template_root: str | Path = "data/templates",
    model_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    enable_ocr: bool = True,
    config: SkillConfig | None = None,
    input_mode: str = "auto",
) -> SkillExecutor:
    capture = Win32WindowCapture(hwnd)
    adapters = []
    metadata: dict[str, object] = {
        "game_id": int(game_id) if str(game_id).isdigit() else str(game_id),
        "template_fallback": True,
        "onnx_enabled": False,
        "ocr_requested": enable_ocr,
    }

    resolved_model = Path(model_path) if model_path else Path("data/models") / str(game_id) / "detector.onnx"
    if resolved_model.exists():
        from robloxhive.body.detector_onnx import OnnxYoloDetector
        from robloxhive.body.tracking import DetectionTracker, PlayerTracker

        resolved_labels: Path | None
        if labels_path:
            resolved_labels = Path(labels_path)
        else:
            json_labels = resolved_model.with_name("labels.json")
            txt_labels = resolved_model.with_name("labels.txt")
            resolved_labels = json_labels if json_labels.exists() else (txt_labels if txt_labels.exists() else None)

        detector = OnnxYoloDetector(
            capture,
            resolved_model,
            labels_path=resolved_labels,
        )
        tracker = DetectionTracker(detector.detect)
        player_tracker = PlayerTracker(tracker)
        adapters.extend([player_tracker, detector])
        metadata.update(
            onnx_enabled=True,
            model_path=str(resolved_model),
            labels_path=str(resolved_labels) if resolved_labels else None,
        )

    if enable_ocr:
        ocr = UiTextDetector(capture)
        if ocr.available:
            adapters.append(ocr)
            metadata["ocr_available"] = True
        else:
            metadata["ocr_available"] = False

    templates = TemplateVision(capture, Path(template_root) / str(game_id))
    adapters.append(templates)

    perception = CompositePerception(adapters)
    input_backend = create_input_backend(hwnd, input_mode)
    background_input_backend = Win32MessageInput(hwnd)
    navigator = LocalNavigator(capture, perception, input_backend)
    skills = GenericVisualSkills(perception, input_backend, config, navigator=navigator)

    executor = SkillExecutor()
    skills.register_into(executor)
    executor.attach_perception(perception, metadata)
    executor.attach_navigation(navigator)
    executor.attach_input(input_backend, background_input_backend)
    executor.metadata["input_mode"] = getattr(input_backend, "mode", input_mode)

    try:
        numeric_game_id = int(game_id)
    except (TypeError, ValueError):
        numeric_game_id = -1

    from robloxhive.games.murder_mystery_2 import OFFICIAL_PLACE_ID, MM2Autonomy
    if numeric_game_id == OFFICIAL_PLACE_ID:
        executor.attach_game_adapter(
            MM2Autonomy(
                perception=perception,
                input_backend=input_backend,
                navigator=navigator,
                frame_source=capture,
            )
        )
        executor.metadata["game_adapter"] = "murder_mystery_2"
        executor.metadata["game_name"] = "Murder Mystery 2"

    return executor
