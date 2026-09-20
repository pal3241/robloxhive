from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from robloxhive.body.discovery import discover_roblox_windows
from robloxhive.body.instances import InstanceManager, ProtectedInstanceError
from robloxhive.brain.learning import LearningManager
from robloxhive.brain.memory import GameMemory
from robloxhive.brain.runtime import AgentRuntime
from robloxhive.shared.models import ActionResult, Goal


class RoleRequest(BaseModel):
    role: Literal["player", "bot", "unassigned"]
    agent_id: str | None = None


class LearnRequest(BaseModel):
    game_id: int = Field(gt=0)
    game_name: str = Field(min_length=1, max_length=120)
    objective: str = Field(
        default="learn from beginner to completion, including progression, tips and tricks",
        max_length=500,
    )


class GoalRequest(BaseModel):
    game_id: int = Field(gt=0)
    type: str = Field(default="autonomous_progress", min_length=1, max_length=80)
    target: str | None = Field(default=None, max_length=160)
    instruction: str = Field(min_length=1, max_length=600)
    priority: int = Field(default=70, ge=0, le=100)
    persistent: bool = True


class BodyResultRequest(BaseModel):
    plan_id: str
    step_index: int = Field(ge=0)
    result: ActionResult
    evidence: dict[str, Any] = Field(default_factory=dict)


def create_app(data_root: str | Path = "data/games") -> FastAPI:
    app = FastAPI(title="RobloxHive Dashboard", version="0.4.0")
    memory = GameMemory(data_root)
    learning = LearningManager(memory)
    runtime = AgentRuntime(memory)
    instances = InstanceManager()
    static_index = Path(__file__).parent / "static" / "index.html"

    def scan_instances() -> list[dict]:
        discovered = discover_roblox_windows()
        known = {item.pid: item for item in instances.list()}

        discovered_pids = {item.pid for item in discovered}
        for item in discovered:
            previous = known.get(item.pid)
            if previous:
                item.role = previous.role
                item.agent_id = previous.agent_id
                item.protected = previous.protected
            instances.register(item)

        for pid in known:
            if pid not in discovered_pids:
                instances.mark_dead(pid)

        return [
            {
                "pid": item.pid,
                "hwnd": item.hwnd,
                "title": item.title,
                "role": item.role.value,
                "agent_id": item.agent_id,
                "protected": item.protected,
                "alive": item.alive,
            }
            for item in instances.list()
        ]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return static_index.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> dict:
        active = runtime.get_plan()
        return {
            "ok": True,
            "version": "0.4.0",
            "synthesizer": getattr(learning.synthesizer, "name", "unknown"),
            "active_plan": active.id if active else None,
        }

    @app.get("/api/instances")
    def get_instances() -> list[dict]:
        return scan_instances()

    @app.post("/api/instances/{pid}/role")
    def set_instance_role(pid: int, request: RoleRequest) -> dict:
        scan_instances()
        match = next((item for item in instances.list() if item.pid == pid and item.alive), None)
        if not match:
            raise HTTPException(status_code=404, detail="Roblox instance not found")

        try:
            if request.role == "player":
                instances.mark_player(pid)
            elif request.role == "bot":
                if not request.agent_id:
                    raise HTTPException(status_code=400, detail="agent_id is required for bot")
                match.protected = False
                instances.assign_bot(pid, request.agent_id)
            else:
                match.protected = False
                match.agent_id = None
                from robloxhive.shared.models import InstanceRole
                match.role = InstanceRole.UNASSIGNED
        except ProtectedInstanceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return {"ok": True}

    @app.get("/api/games")
    def games() -> list[dict]:
        return memory.list_games()

    @app.get("/api/games/{game_id}/memory")
    def game_memory(game_id: int) -> dict:
        return memory.load_profile(game_id)

    @app.get("/api/games/{game_id}/knowledge")
    def game_knowledge(game_id: int) -> dict:
        knowledge = memory.load_knowledge(game_id)
        if not knowledge:
            raise HTTPException(status_code=404, detail="Knowledge has not been synthesized yet")
        return knowledge

    @app.post("/api/learn")
    def start_learning(request: LearnRequest) -> dict:
        job = learning.start(request.game_id, request.game_name, request.objective)
        return job.to_dict()

    @app.get("/api/learn/jobs")
    def learning_jobs() -> list[dict]:
        return learning.list()

    @app.get("/api/learn/jobs/{job_id}")
    def learning_job(job_id: str) -> dict:
        job = learning.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Learning job not found")
        return job.to_dict()

    @app.post("/api/agent/goals")
    def start_goal(request: GoalRequest) -> dict:
        knowledge = memory.load_knowledge(request.game_id)
        if not knowledge:
            raise HTTPException(
                status_code=409,
                detail="Game knowledge is empty. Learn/synthesize this game first.",
            )
        goal = Goal(
            type=request.type,
            target=request.target,
            priority=request.priority,
            persistent=request.persistent,
            metadata={"instruction": request.instruction},
        )
        plan = runtime.start_goal(request.game_id, goal)
        return plan.model_dump(mode="json")

    @app.get("/api/agent/active-plan")
    def active_plan() -> dict | None:
        plan = runtime.get_plan()
        return plan.model_dump(mode="json") if plan else None

    @app.get("/api/agent/plans/{plan_id}")
    def get_plan(plan_id: str) -> dict:
        plan = runtime.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        return plan.model_dump(mode="json")

    @app.get("/api/body/commands/next")
    def next_body_command(agent_id: str = "agent-01", timeout: float = 0.0) -> dict | None:
        # agent_id is reserved for routing when multi-agent support arrives.
        _ = agent_id
        command = runtime.next_command(timeout=max(0.0, min(timeout, 5.0)))
        if command is None:
            return None
        return {
            "source": command.source,
            "type": command.type,
            "payload": command.payload,
        }

    @app.post("/api/body/results")
    def body_result(request: BodyResultRequest) -> dict:
        try:
            plan = runtime.record_result(
                request.plan_id,
                request.step_index,
                request.result,
                request.evidence,
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return plan.model_dump(mode="json")

    return app
