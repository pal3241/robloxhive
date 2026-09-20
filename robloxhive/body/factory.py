from __future__ import annotations

from pathlib import Path

from robloxhive.body.capture import Win32WindowCapture
from robloxhive.body.generic_skills import GenericVisualSkills, SkillConfig
from robloxhive.body.input_win32 import Win32MessageInput
from robloxhive.body.perception import TemplateVision
from robloxhive.body.skills import SkillExecutor


def create_generic_skill_executor(
    hwnd: int,
    game_id: int | str,
    template_root: str | Path = "data/templates",
    config: SkillConfig | None = None,
) -> SkillExecutor:
    capture = Win32WindowCapture(hwnd)
    perception = TemplateVision(capture, Path(template_root) / str(game_id))
    input_backend = Win32MessageInput(hwnd)
    skills = GenericVisualSkills(perception, input_backend, config)
    executor = SkillExecutor()
    skills.register_into(executor)
    return executor
