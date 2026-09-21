from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from robloxhive.body.discovery import discover_roblox_windows
from robloxhive.body.instances import InstanceManager, ProtectedInstanceError, RobloxInstance
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
    agent_id: str = Field(default="agent-01", min_length=1, max_length=80)
    game_id: int | None = Field(default=None, gt=0)
    type: str = Field(default="autonomous_progress", min_length=1, max_length=80)
    target: str | None = Field(default=None, max_length=160)
    instruction: str = Field(min_length=1, max_length=600)
    priority: int = Field(default=70, ge=0, le=100)
    persistent: bool = True


class BodyRegistration(BaseModel):
    agent_id: str = Field(min_length=1, max_length=80)
    skills: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JoinGameRequest(BaseModel):
    agent_id: str = "agent-01"
    place_id: int = Field(gt=0)


class JoinGameResult(BaseModel):
    agent_id: str
    join_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class GameContextRequest(BaseModel):
    agent_id: str = "agent-01"
    place_id: int = Field(gt=0)


class GameContextResult(BaseModel):
    agent_id: str
    context_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class DirectControlRequest(BaseModel):
    agent_id: str = "agent-01"
    action: Literal["forward", "back", "left", "right", "jump", "interact", "release"]
    seconds: float = Field(default=0.15, ge=0.01, le=3.0)
    mode: Literal["background", "reliable"] = "background"


class DirectControlResult(BaseModel):
    agent_id: str
    control_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class BindInstanceResult(BaseModel):
    agent_id: str
    bind_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


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


class GameControlRequest(BaseModel):
    agent_id: str = "agent-01"
    action: Literal["enable", "disable", "role_override", "clear_role"]
    role: Literal["auto", "innocent", "sheriff", "hero", "murderer"] | None = None


class GameControlResult(BaseModel):
    agent_id: str
    control_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class MM2DatasetRequest(BaseModel):
    agent_id: str = "agent-01"
    action: Literal["capture", "list", "preview", "approve", "reject", "review", "export", "status"]
    sample_id: str | None = None
    indices: list[int] | None = None
    note: str | None = Field(default=None, max_length=300)
    auto_approve_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_ratio: float = Field(default=0.2, ge=0.0, le=0.5)
    limit: int = Field(default=30, ge=1, le=200)
    boxes: list[dict[str, Any]] | None = None


def create_app(data_root: str | Path = "data/games") -> FastAPI:
    app = FastAPI(title="RobloxHive Dashboard", version="1.0.2")
    memory = GameMemory(data_root)
    learning = LearningManager(memory)
    runtime = AgentRuntime(memory)
    instances = InstanceManager()
    body_nodes: dict[str, dict[str, Any]] = {}
    manual_tests: dict[str, dict[str, Any]] = {}
    perception_probes: dict[str, dict[str, Any]] = {}
    navigation_probes: dict[str, dict[str, Any]] = {}
    game_controls: dict[str, dict[str, Any]] = {}
    join_requests: dict[str, dict[str, Any]] = {}
    game_context_requests: dict[str, dict[str, Any]] = {}
    direct_controls: dict[str, dict[str, Any]] = {}
    bind_requests: dict[str, dict[str, Any]] = {}
    static_index = Path(__file__).parent / "static" / "index.html"

    def body_snapshot() -> list[dict[str, Any]]:
        now = time.time()
        items = []
        for node in body_nodes.values():
            copy = dict(node)
            copy["online"] = now - float(copy.get("last_seen", 0)) <= 30.0
            items.append(copy)
        return sorted(items, key=lambda item: item.get("agent_id", ""))

    def scan_instances() -> list[dict]:
        # Brain may run on Android/Linux, so Windows discovery must come from
        # connected Body nodes instead of scanning the Brain host.
        discovered = []
        if os.name == "nt":
            discovered.extend(discover_roblox_windows())
        for node in body_nodes.values():
            metadata = node.get("metadata") or {}
            for row in metadata.get("windows") or []:
                try:
                    discovered.append(
                        RobloxInstance(
                            pid=int(row["pid"]),
                            hwnd=int(row["hwnd"]),
                            title=str(row.get("title") or "Roblox"),
                            alive=bool(row.get("alive", True)),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue

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
                "pid": item.pid, "hwnd": item.hwnd, "title": item.title,
                "role": item.role.value, "agent_id": item.agent_id,
                "protected": item.protected, "alive": item.alive,
            }
            for item in instances.list() if item.alive
        ]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return static_index.read_text(encoding="utf-8")

    @app.websocket("/ws/dashboard")
    async def dashboard_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json({
                    "type": "snapshot",
                    "server_time": time.time(),
                    "bodies": body_snapshot(),
                    "instances": scan_instances(),
                })
                await asyncio.sleep(1.5)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.get("/api/health")
    def health() -> dict:
        active = runtime.get_plan()
        online_bodies = sum(1 for body in body_snapshot() if body["online"])
        return {
            "ok": True,
            "version": "1.0.2",
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
                for node in body_nodes.values():
                    metadata = node.get("metadata") or {}
                    if int(metadata.get("pid") or 0) == pid:
                        runtime.commands.publish(Command(
                            source="dashboard",
                            type="DISARM",
                            payload={
                                "agent_id": node.get("agent_id"),
                                "pid": pid,
                            },
                        ))
            elif request.role == "bot":
                if not request.agent_id:
                    raise HTTPException(status_code=400, detail="agent_id is required for bot")
                match.protected = False
                instances.assign_bot(pid, request.agent_id)
                bind_id = uuid4().hex[:12]
                bind_requests[bind_id] = {
                    "bind_id": bind_id,
                    "agent_id": request.agent_id,
                    "pid": pid,
                    "status": "queued",
                    "created_at": time.time(),
                }
                runtime.commands.publish(Command(
                    source="dashboard",
                    type="BIND_INSTANCE",
                    payload={
                        "bind_id": bind_id,
                        "agent_id": request.agent_id,
                        "pid": pid,
                    },
                ))
            else:
                previous_agent = match.agent_id
                match.protected = False
                match.agent_id = None
                from robloxhive.shared.models import InstanceRole
                match.role = InstanceRole.UNASSIGNED
                if previous_agent:
                    runtime.commands.publish(Command(
                        source="dashboard",
                        type="DISARM",
                        payload={"agent_id": previous_agent, "pid": pid},
                    ))
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
        node = body_nodes.get(request.agent_id)
        if not node or time.time() - float(node.get("last_seen", 0)) > 30.0:
            raise HTTPException(status_code=409, detail="Selected Windows Body is offline")

        metadata = node.get("metadata") or {}
        if not metadata.get("armed"):
            raise HTTPException(
                status_code=409,
                detail="Selected Body is not armed. Assign a bot window in Instances first.",
            )

        game_id = int(
            request.game_id
            or metadata.get("place_id")
            or metadata.get("game_id")
            or 0
        )
        if game_id <= 0:
            raise HTTPException(
                status_code=409,
                detail="Current game is unknown. Set the Place ID or attach the current game first.",
            )

        goal_metadata = {
            "instruction": request.instruction,
            "agent_id": request.agent_id,
        }
        if request.type == "follow_player":
            if not request.target or not request.target.strip():
                raise HTTPException(status_code=400, detail="Username is required for follow_player")
            goal_metadata["username"] = request.target.strip()

        goal = Goal(
            type=request.type,
            target=request.target.strip() if request.target else None,
            priority=request.priority,
            persistent=request.persistent,
            metadata=goal_metadata,
        )
        # Empty knowledge is allowed: the planner falls back to observation and
        # direct goal skills instead of making the Agent tab unusable.
        plan = runtime.start_goal(game_id, goal)
        return {
            **plan.model_dump(mode="json"),
            "agent_id": request.agent_id,
            "knowledge_available": bool(memory.load_knowledge(game_id)),
        }

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

    @app.post("/api/body/join")
    def join_game(request: JoinGameRequest) -> dict:
        node = body_nodes.get(request.agent_id)
        if not node or time.time() - float(node.get("last_seen", 0)) > 30.0:
            raise HTTPException(status_code=409, detail="Selected Windows Body is offline")
        join_id = uuid4().hex[:12]
        join_requests[join_id] = {
            "join_id": join_id, "agent_id": request.agent_id,
            "place_id": request.place_id, "status": "queued",
            "created_at": time.time(),
        }
        runtime.commands.publish(Command(
            source="dashboard", type="JOIN_GAME",
            payload={"join_id": join_id, "agent_id": request.agent_id, "place_id": request.place_id},
        ))
        return join_requests[join_id]

    @app.post("/api/body/game-context")
    def set_game_context(request: GameContextRequest) -> dict:
        node = body_nodes.get(request.agent_id)
        if not node or time.time() - float(node.get("last_seen", 0)) > 30.0:
            raise HTTPException(status_code=409, detail="Selected Windows Body is offline")
        context_id = uuid4().hex[:12]
        game_context_requests[context_id] = {
            "context_id": context_id,
            "agent_id": request.agent_id,
            "place_id": request.place_id,
            "status": "queued",
            "created_at": time.time(),
        }
        runtime.commands.publish(Command(
            source="dashboard",
            type="SET_GAME_CONTEXT",
            payload={
                "context_id": context_id,
                "agent_id": request.agent_id,
                "place_id": request.place_id,
            },
        ))
        return game_context_requests[context_id]

    @app.get("/api/body/game-contexts")
    def get_game_context_requests() -> list[dict[str, Any]]:
        return sorted(
            game_context_requests.values(),
            key=lambda item: item.get("created_at", 0),
            reverse=True,
        )[:30]

    @app.post("/api/body/game-context-results")
    def game_context_result(request: GameContextResult) -> dict:
        node = body_nodes.get(request.agent_id)
        if node:
            node["last_seen"] = time.time()
            if request.result.get("ok"):
                node["metadata"] = {
                    **(node.get("metadata") or {}),
                    "game_id": request.result.get("game_id"),
                    "place_id": request.result.get("place_id"),
                    "game_context_source": "dashboard",
                }
        if request.context_id and request.context_id in game_context_requests:
            game_context_requests[request.context_id].update(
                status="complete" if request.result.get("ok") else "failed",
                result=request.result,
                finished_at=time.time(),
            )
            return game_context_requests[request.context_id]
        return {"status": "orphan_result", "result": request.result}

    @app.get("/api/body/joins")
    def get_join_requests() -> list[dict[str, Any]]:
        return sorted(join_requests.values(), key=lambda x: x.get("created_at", 0), reverse=True)[:30]

    @app.post("/api/body/join-results")
    def join_game_result(request: JoinGameResult) -> dict:
        node = body_nodes.get(request.agent_id)
        if node:
            node["last_seen"] = time.time()
        if request.join_id and request.join_id in join_requests:
            join_requests[request.join_id].update(
                status="complete" if request.result.get("ok") else "failed",
                result=request.result, finished_at=time.time(),
            )
            return join_requests[request.join_id]
        return {"status": "orphan_result", "result": request.result}

    @app.get("/api/body/binds")
    def get_bind_requests() -> list[dict[str, Any]]:
        return sorted(
            bind_requests.values(),
            key=lambda item: item.get("created_at", 0),
            reverse=True,
        )[:30]

    @app.post("/api/body/bind-results")
    def bind_instance_result(request: BindInstanceResult) -> dict:
        node = body_nodes.get(request.agent_id)
        if node:
            node["last_seen"] = time.time()
            if request.result.get("ok"):
                node["metadata"] = {
                    **(node.get("metadata") or {}),
                    "pid": request.result.get("pid"),
                    "hwnd": request.result.get("hwnd"),
                    "title": request.result.get("title"),
                }
        if request.bind_id and request.bind_id in bind_requests:
            bind_requests[request.bind_id].update(
                status="complete" if request.result.get("ok") else "failed",
                result=request.result,
                finished_at=time.time(),
            )
            return bind_requests[request.bind_id]
        return {"status": "orphan_result", "result": request.result}

    @app.post("/api/body/control")
    def direct_control(request: DirectControlRequest) -> dict:
        node = body_nodes.get(request.agent_id)
        if not node or time.time() - float(node.get("last_seen", 0)) > 30.0:
            raise HTTPException(status_code=409, detail="Selected Windows Body is offline")
        control_id = uuid4().hex[:12]
        direct_controls[control_id] = {
            "control_id": control_id,
            "agent_id": request.agent_id,
            "action": request.action,
            "seconds": request.seconds,
            "mode": request.mode,
            "status": "queued",
            "created_at": time.time(),
        }
        runtime.commands.publish(Command(
            source="dashboard",
            type="DIRECT_INPUT",
            payload={
                "control_id": control_id,
                "agent_id": request.agent_id,
                "action": request.action,
                "seconds": request.seconds,
                "mode": request.mode,
            },
        ))
        return direct_controls[control_id]

    @app.get("/api/body/controls")
    def get_direct_controls() -> list[dict[str, Any]]:
        return sorted(
            direct_controls.values(),
            key=lambda item: item.get("created_at", 0),
            reverse=True,
        )[:30]

    @app.post("/api/body/control-results")
    def direct_control_result(request: DirectControlResult) -> dict:
        node = body_nodes.get(request.agent_id)
        if node:
            node["last_seen"] = time.time()
            node["last_control"] = request.result
        if request.control_id and request.control_id in direct_controls:
            direct_controls[request.control_id].update(
                status="complete" if request.result.get("ok") else "failed",
                result=request.result,
                finished_at=time.time(),
            )
            return direct_controls[request.control_id]
        return {"status": "orphan_result", "result": request.result}

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

    @app.get("/api/games/mm2/status")
    def mm2_status() -> list[dict[str, Any]]:
        from robloxhive.games.murder_mystery_2 import OFFICIAL_PLACE_ID

        rows = []
        for node in body_snapshot():
            metadata = node.get("metadata", {})
            game = metadata.get("game", {})
            game_id = int(metadata.get("place_id") or metadata.get("game_id") or 0)
            adapter_ready = game.get("adapter") == "murder_mystery_2"
            if adapter_ready or game_id == OFFICIAL_PLACE_ID:
                rows.append({
                    "agent_id": node.get("agent_id"),
                    "online": node.get("online", False),
                    "armed": bool(metadata.get("armed")),
                    "pid": metadata.get("pid"),
                    "hwnd": metadata.get("hwnd"),
                    "game_id": game_id,
                    "adapter_ready": adapter_ready,
                    "state": game if adapter_ready else {
                        "adapter": None,
                        "enabled": False,
                        "role": "unknown",
                        "phase": "unknown",
                        "diagnostics": {"waiting_for_adapter": True},
                    },
                })
        return rows

    @app.post("/api/games/mm2/control")
    def mm2_control(request: GameControlRequest) -> dict:
        node = body_nodes.get(request.agent_id)
        if not node or time.time() - float(node.get("last_seen", 0)) > 15.0:
            raise HTTPException(status_code=409, detail="Selected Windows Body is offline")
        game = node.get("metadata", {}).get("game", {})
        if game.get("adapter") != "murder_mystery_2":
            raise HTTPException(status_code=409, detail="Selected Body is not running the MM2 adapter")

        control_id = uuid4().hex[:12]
        payload = {
            "control_id": control_id,
            "agent_id": request.agent_id,
            "action": request.action,
        }
        if request.role is not None:
            payload["role"] = request.role

        game_controls[control_id] = {
            "control_id": control_id,
            "agent_id": request.agent_id,
            "action": request.action,
            "role": request.role,
            "status": "queued",
            "created_at": time.time(),
        }
        runtime.commands.publish(Command(source="dashboard", type="GAME_CONTROL", payload=payload))
        return game_controls[control_id]

    @app.post("/api/games/mm2/dataset")
    def mm2_dataset(request: MM2DatasetRequest) -> dict:
        node = body_nodes.get(request.agent_id)
        if not node or time.time() - float(node.get("last_seen", 0)) > 15.0:
            raise HTTPException(status_code=409, detail="Selected Windows Body is offline")
        game = node.get("metadata", {}).get("game", {})
        if game.get("adapter") != "murder_mystery_2":
            raise HTTPException(status_code=409, detail="Selected Body is not running the MM2 adapter")

        control_id = uuid4().hex[:12]
        action_map = {
            "capture": "dataset_capture",
            "list": "dataset_list",
            "preview": "dataset_preview",
            "approve": "dataset_approve",
            "reject": "dataset_approve",
            "review": "dataset_review",
            "export": "dataset_export",
            "status": "dataset_status",
        }
        payload: dict[str, Any] = {
            "control_id": control_id,
            "agent_id": request.agent_id,
            "action": action_map[request.action],
            "limit": request.limit,
            "validation_ratio": request.validation_ratio,
        }
        if request.sample_id:
            payload["sample_id"] = request.sample_id
        if request.indices is not None:
            payload["indices"] = request.indices
        if request.note:
            payload["note"] = request.note
        if request.auto_approve_confidence is not None:
            payload["auto_approve_confidence"] = request.auto_approve_confidence
        if request.boxes is not None:
            payload["boxes"] = request.boxes
        if request.action in {"approve", "reject"}:
            payload["approved"] = request.action == "approve"

        game_controls[control_id] = {
            "control_id": control_id,
            "agent_id": request.agent_id,
            "action": payload["action"],
            "sample_id": request.sample_id,
            "status": "queued",
            "created_at": time.time(),
        }
        runtime.commands.publish(Command(source="dashboard", type="GAME_CONTROL", payload=payload))
        return game_controls[control_id]

    @app.get("/api/games/mm2/controls")
    def mm2_controls() -> list[dict[str, Any]]:
        return sorted(
            game_controls.values(),
            key=lambda item: item.get("created_at", 0),
            reverse=True,
        )[:30]

    @app.post("/api/body/game-control-results")
    def game_control_result(request: GameControlResult) -> dict:
        node = body_nodes.get(request.agent_id)
        if node:
            node["last_seen"] = time.time()
        if request.control_id and request.control_id in game_controls:
            game_controls[request.control_id].update(
                status="complete",
                result=request.result,
                finished_at=time.time(),
            )
            return game_controls[request.control_id]
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
        command = runtime.next_command(
            timeout=max(0.0, min(timeout, 5.0)),
            agent_id=agent_id,
        )
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
