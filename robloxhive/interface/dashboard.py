from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from robloxhive.body.discovery import discover_roblox_windows
from robloxhive.body.instances import InstanceManager, ProtectedInstanceError
from robloxhive.brain.learning import LearningManager
from robloxhive.brain.memory import GameMemory
from robloxhive.brain.runtime import AgentRuntime
from robloxhive.shared.command_bus import Command
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


class BodyRegistration(BaseModel):
    agent_id: str = Field(min_length=1, max_length=80)
    skills: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BodyResultRequest(BaseModel):
    agent_id: str = "agent-01"
    plan_id: str
    step_index: int = Field(ge=0)
    result: ActionResult
    evidence: dict[str, Any] = Field(default_factory=dict)


class ManualSkillRequest(BaseModel):
    agent_id: str = "agent-01"
    skill: Literal["navigate", "collect", "interact", "follow_player"]
    target: str = Field(min_length=1, max_length=160)
    game_id: int | None = Field(default=None, gt=0)
    instruction: str | None = Field(default=None, max_length=500)
    iterations: int | None = Field(default=None, ge=1, le=300)
    key: str | None = Field(default=None, max_length=32)


class ManualSkillResult(BaseModel):
    agent_id: str
    test_id: str | None = None
    skill: str | None = None
    result: ActionResult
    evidence: dict[str, Any] = Field(default_factory=dict)


class PerceptionProbeRequest(BaseModel):
    agent_id: str = "agent-01"
    label: str = Field(min_length=1, max_length=160)


class PerceptionProbeResult(BaseModel):
    agent_id: str
    probe_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class NavigationProbeRequest(BaseModel):
    agent_id: str = "agent-01"


class NavigationProbeResult(BaseModel):
    agent_id: str
    probe_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


def create_app(data_root: str | Path = "data/games") -> FastAPI:
    app = FastAPI(title="RobloxHive Dashboard", version="0.7.0")
    memory = GameMemory(data_root)
    learning = LearningManager(memory)
    runtime = AgentRuntime(memory)
    instances = InstanceManager()
    body_nodes: dict[str, dict[str, Any]] = {}
    manual_tests: dict[str, dict[str, Any]] = {}
    perception_probes: dict[str, dict[str, Any]] = {}
    navigation_probes: dict[str, dict[str, Any]] = {}
    static_index = Path(__file__).parent / "static" / "index.html"

    def body_snapshot() -> list[dict[str, Any]]:
        now = time.time()
        items = []
        for node in body_nodes.values():
            copy = dict(node)
            copy["online"] = now - float(copy.get("last_seen", 0)) <= 15.0
            items.append(copy)
        return sorted(items, key=lambda item: item.get("agent_id", ""))

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
        online_bodies = sum(1 for body in body_snapshot() if body["online"])
        return {
            "ok": True,
            "version": "0.7.0",
            "synthesizer": getattr(learning.synthesizer, "name", "unknown"),
            "active_plan": active.id if active else None,
            "online_bodies": online_bodies,
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

    @app.post("/api/body/register")
    def register_body(request: BodyRegistration) -> dict:
        previous = body_nodes.get(request.agent_id, {})
        body_nodes[request.agent_id] = {
            **previous,
            "agent_id": request.agent_id,
            "skills": sorted(set(request.skills)),
            "metadata": request.metadata,
            "last_seen": time.time(),
        }
        return {"ok": True, "agent_id": request.agent_id}

    @app.get("/api/body/nodes")
    def get_body_nodes() -> list[dict[str, Any]]:
        return body_snapshot()

    @app.post("/api/body/skills/test")
    def test_body_skill(request: ManualSkillRequest) -> dict:
        node = body_nodes.get(request.agent_id)
        if not node or time.time() - float(node.get("last_seen", 0)) > 15.0:
            raise HTTPException(status_code=409, detail="Selected Windows Body is offline")
        if request.skill not in node.get("skills", []):
            raise HTTPException(
                status_code=409,
                detail=f"Body does not advertise skill: {request.skill}",
            )

        test_id = uuid4().hex[:12]
        payload: dict[str, Any] = {
            "test_id": test_id,
            "agent_id": request.agent_id,
            "skill": request.skill,
            "target": request.target,
            "instruction": request.instruction or f"{request.skill} {request.target}",
        }
        if request.game_id is not None:
            payload["game_id"] = request.game_id
        if request.iterations is not None:
            payload["iterations"] = request.iterations
        if request.key:
            payload["key"] = request.key

        manual_tests[test_id] = {
            "test_id": test_id,
            "agent_id": request.agent_id,
            "skill": request.skill,
            "target": request.target,
            "status": "queued",
            "created_at": time.time(),
        }
        runtime.commands.publish(Command(source="dashboard", type="EXECUTE_SKILL", payload=payload))
        return manual_tests[test_id]

    @app.get("/api/body/skill-tests")
    def get_skill_tests() -> list[dict[str, Any]]:
        return sorted(
            manual_tests.values(),
            key=lambda item: item.get("created_at", 0),
            reverse=True,
        )[:30]

    @app.post("/api/body/perception/probe")
    def probe_perception(request: PerceptionProbeRequest) -> dict:
        node = body_nodes.get(request.agent_id)
        if not node or time.time() - float(node.get("last_seen", 0)) > 15.0:
            raise HTTPException(status_code=409, detail="Selected Windows Body is offline")

        probe_id = uuid4().hex[:12]
        perception_probes[probe_id] = {
            "probe_id": probe_id,
            "agent_id": request.agent_id,
            "label": request.label,
            "status": "queued",
            "created_at": time.time(),
        }
        runtime.commands.publish(
            Command(
                source="dashboard",
                type="PERCEPTION_PROBE",
                payload={
                    "probe_id": probe_id,
                    "agent_id": request.agent_id,
                    "label": request.label,
                },
            )
        )
        return perception_probes[probe_id]

    @app.get("/api/body/perception/probes")
    def get_perception_probes() -> list[dict[str, Any]]:
        return sorted(
            perception_probes.values(),
            key=lambda item: item.get("created_at", 0),
            reverse=True,
        )[:30]

    @app.post("/api/body/perception-results")
    def perception_result(request: PerceptionProbeResult) -> dict:
        node = body_nodes.get(request.agent_id)
        if node:
            node["last_seen"] = time.time()
            node["last_perception"] = request.result
        if request.probe_id and request.probe_id in perception_probes:
            perception_probes[request.probe_id].update(
                status="complete",
                result=request.result,
                finished_at=time.time(),
            )
            return perception_probes[request.probe_id]
        return {
            "status": "orphan_result",
            "agent_id": request.agent_id,
            "result": request.result,
        }

    @app.post("/api/body/navigation/probe")
    def probe_navigation(request: NavigationProbeRequest) -> dict:
        node = body_nodes.get(request.agent_id)
        if not node or time.time() - float(node.get("last_seen", 0)) > 15.0:
            raise HTTPException(status_code=409, detail="Selected Windows Body is offline")

        probe_id = uuid4().hex[:12]
        navigation_probes[probe_id] = {
            "probe_id": probe_id,
            "agent_id": request.agent_id,
            "status": "queued",
            "created_at": time.time(),
        }
        runtime.commands.publish(
            Command(
                source="dashboard",
                type="NAVIGATION_PROBE",
                payload={
                    "probe_id": probe_id,
                    "agent_id": request.agent_id,
                },
            )
        )
        return navigation_probes[probe_id]

    @app.get("/api/body/navigation/probes")
    def get_navigation_probes() -> list[dict[str, Any]]:
        return sorted(
            navigation_probes.values(),
            key=lambda item: item.get("created_at", 0),
            reverse=True,
        )[:30]

    @app.post("/api/body/navigation-results")
    def navigation_result(request: NavigationProbeResult) -> dict:
        node = body_nodes.get(request.agent_id)
        if node:
            node["last_seen"] = time.time()
            node["last_navigation"] = request.result
        if request.probe_id and request.probe_id in navigation_probes:
            navigation_probes[request.probe_id].update(
                status="complete",
                result=request.result,
                finished_at=time.time(),
            )
            return navigation_probes[request.probe_id]
        return {
            "status": "orphan_result",
            "agent_id": request.agent_id,
            "result": request.result,
        }

    @app.get("/api/body/commands/next")
    def next_body_command(agent_id: str = "agent-01", timeout: float = 0.0) -> dict | None:
        node = body_nodes.get(agent_id)
        if node:
            node["last_seen"] = time.time()
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
        node = body_nodes.get(request.agent_id)
        if node:
            node["last_seen"] = time.time()
            node["last_result"] = request.result.model_dump(mode="json")
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

    @app.post("/api/body/manual-results")
    def manual_result(request: ManualSkillResult) -> dict:
        node = body_nodes.get(request.agent_id)
        if node:
            node["last_seen"] = time.time()
            node["last_result"] = request.result.model_dump(mode="json")
        if request.test_id and request.test_id in manual_tests:
            manual_tests[request.test_id].update(
                status="complete",
                result=request.result.model_dump(mode="json"),
                evidence=request.evidence,
                finished_at=time.time(),
            )
            return manual_tests[request.test_id]
        return {
            "status": "orphan_result",
            "agent_id": request.agent_id,
            "result": request.result.model_dump(mode="json"),
        }

    return app
