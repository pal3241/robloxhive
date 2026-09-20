from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class InstanceRole(str, Enum):
    PLAYER = "player"
    BOT = "bot"
    UNASSIGNED = "unassigned"


class ActionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"


class ActionResult(BaseModel):
    action: str
    status: ActionStatus
    attempts: int = 1
    duration_ms: int = 0
    error: str | None = None
    recoverable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ActionStatus.SUCCESS


class Goal(BaseModel):
    type: str
    target: str | None = None
    priority: int = Field(default=50, ge=0, le=100)
    persistent: bool = False
    interruptible: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorldState(BaseModel):
    game_id: int | None = None
    place_id: int | None = None
    health: float | None = None
    state: str = "unknown"
    visible_objects: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
