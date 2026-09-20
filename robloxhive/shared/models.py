from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

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


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class PlanStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


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


class PlanStep(BaseModel):
    index: int
    skill: str
    instruction: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    success_condition: str
    knowledge_refs: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    max_attempts: int = Field(default=3, ge=1, le=10)
    last_error: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    game_id: int
    goal: Goal
    status: PlanStatus = PlanStatus.READY
    current_step_index: int = 0
    knowledge_synthesizer: str | None = None
    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def current_step(self) -> PlanStep | None:
        if self.current_step_index < 0 or self.current_step_index >= len(self.steps):
            return None
        return self.steps[self.current_step_index]


class WorldState(BaseModel):
    game_id: int | None = None
    place_id: int | None = None
    health: float | None = None
    state: str = "unknown"
    visible_objects: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
