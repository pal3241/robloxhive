from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from robloxhive.brain.learning import LearningManager
from robloxhive.brain.memory import GameMemory
from robloxhive.brain.synthesis import HeuristicKnowledgeSynthesizer, OllamaKnowledgeSynthesizer
from robloxhive.single.runtime import SingleBotRuntime


class GoalRequest(BaseModel):
    type: str = Field(default="custom", min_length=1, max_length=80)
    instruction: str = Field(default="", max_length=1200)
    target: str | None = Field(default=None, max_length=160)
    start: bool = True


class ToggleRequest(BaseModel):
    enabled: bool


class ControlRequest(BaseModel):
    action: Literal["forward", "back", "left", "right", "jump", "interact", "release"]
    seconds: float = Field(default=0.18, ge=0.01, le=3.0)
    mode: Literal["background", "reliable"] = "background"


class OllamaRequest(BaseModel):
    url: str | None = Field(default=None, max_length=300)
    model: str | None = Field(default=None, max_length=120)
    vision: bool | None = None


class ModelPullRequest(BaseModel):
    model: str = Field(min_length=1, max_length=120)


class GameRequest(BaseModel):
    game_id: int = Field(ge=0)


class LearnRequest(BaseModel):
    game_id: int = Field(gt=0)
    game_name: str = Field(min_length=1, max_length=120)
    objective: str = Field(
        default="learn the game from beginner to completion, including mechanics, progression, UI, map, roles, team/enemy behavior, combat, and common mistakes",
        max_length=900,
    )


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    kinds: list[str] | None = None
    limit: int = Field(default=20, ge=1, le=64)


class RuntimeKnowledgeSynthesizer:
    """Use the same Ollama endpoint/model currently configured in the dashboard."""

    def __init__(self, runtime: SingleBotRuntime) -> None:
        self.runtime = runtime
        self.fallback = HeuristicKnowledgeSynthesizer()

    @property
    def name(self) -> str:
        cfg = self.runtime.actor.config
        return f"ollama:{cfg.model}"

    def synthesize(self, game_name: str, bundle: dict[str, Any]) -> dict[str, Any]:
        cfg = self.runtime.actor.config
        try:
            return OllamaKnowledgeSynthesizer(
                cfg.model,
                cfg.base_url,
                timeout=max(90, int(cfg.timeout_s * 3)),
            ).synthesize(game_name, bundle)
        except RuntimeError as exc:
            result = self.fallback.synthesize(game_name, bundle)
            result["synthesizer"] = "heuristic-fallback"
            result["synthesis_warning"] = str(exc)
            return result


def create_single_app(runtime: SingleBotRuntime) -> FastAPI:
    app = FastAPI(title="RobloxHive 1.0.0", version="1.0.0")
    static_index = Path(__file__).parent / "static" / "single.html"
    learning_memory = GameMemory("data/games")
    learning = LearningManager(
        learning_memory,
        synthesizer=RuntimeKnowledgeSynthesizer(runtime),
    )
    pull_state: dict[str, Any] = {
        "running": False,
        "model": None,
        "status": "idle",
        "error": None,
        "finished_at": None,
    }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return static_index.read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return runtime.status()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json({"type": "status", "data": runtime.status()})
                await asyncio.sleep(1.0)
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.post("/api/agent/goal")
    def set_goal(request: GoalRequest) -> dict[str, Any]:
        instruction = request.instruction.strip()
        if not instruction:
            if request.type == "follow_player":
                instruction = "Follow the exact requested Roblox username and maintain a useful distance."
            else:
                instruction = f"Execute the {request.type} goal using the current game state."
        try:
            goal = runtime.set_goal(
                instruction,
                goal_type=request.type,
                target=request.target,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.start:
            runtime.set_enabled(True)
        return {"ok": True, "goal": goal, "enabled": runtime.enabled}

    @app.post("/api/agent/toggle")
    def toggle(request: ToggleRequest) -> dict[str, Any]:
        runtime.set_enabled(request.enabled)
        return {"ok": True, "enabled": runtime.enabled, "paused_reason": runtime.paused_reason}

    @app.post("/api/control")
    def control(request: ControlRequest) -> dict[str, Any]:
        return runtime.executor.direct_control(request.model_dump())

    @app.get("/api/ollama")
    def ollama_health() -> dict[str, Any]:
        return runtime.actor.health()

    @app.post("/api/ollama")
    def ollama_config(request: OllamaRequest) -> dict[str, Any]:
        return runtime.configure_ollama(
            url=request.url,
            model=request.model,
            vision=request.vision,
        )

    @app.get("/api/ollama/pull")
    def pull_status() -> dict[str, Any]:
        return dict(pull_state)

    @app.post("/api/ollama/pull")
    def pull_model(request: ModelPullRequest) -> dict[str, Any]:
        if pull_state.get("running"):
            raise HTTPException(status_code=409, detail="A model pull is already running")

        pull_state.update(
            running=True,
            model=request.model,
            status="pulling",
            error=None,
            finished_at=None,
        )

        def worker() -> None:
            try:
                runtime.actor._request(
                    "/api/pull",
                    {"name": request.model, "stream": False},
                )
                pull_state.update(
                    running=False,
                    status="complete",
                    error=None,
                    finished_at=time.time(),
                )
            except Exception as exc:
                pull_state.update(
                    running=False,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=time.time(),
                )

        threading.Thread(target=worker, name="robloxhive-ollama-pull", daemon=True).start()
        return dict(pull_state)

    @app.post("/api/game")
    def set_game(request: GameRequest) -> dict[str, Any]:
        runtime.set_game_id(request.game_id)
        return {"ok": True, "game_id": runtime.game_id}

    @app.post("/api/learn")
    def learn(request: LearnRequest) -> dict[str, Any]:
        job = learning.start(request.game_id, request.game_name, request.objective)
        return job.to_dict()

    @app.get("/api/learn")
    def learn_jobs() -> list[dict[str, Any]]:
        return learning.list()

    @app.get("/api/memory")
    def memory_stats() -> dict[str, Any]:
        return runtime.memory.stats()

    @app.post("/api/memory/search")
    def memory_search(request: MemorySearchRequest) -> list[dict[str, Any]]:
        return runtime.memory.retrieve(
            request.query,
            game_id=runtime.game_id,
            kinds=request.kinds,
            limit=request.limit,
        )

    @app.get("/api/map")
    def map_status() -> dict[str, Any]:
        return runtime.semantic_map.summary()

    @app.get("/api/world")
    def world() -> dict[str, Any]:
        snapshot = runtime.world.observe()
        runtime.last_world = snapshot.compact()
        return runtime.last_world

    @app.post("/api/perception/probe")
    def probe(payload: dict[str, Any]) -> dict[str, Any]:
        label = str(payload.get("label") or "").strip()
        if not label:
            raise HTTPException(status_code=400, detail="label is required")
        return runtime.executor.probe(label)

    @app.post("/api/ui/click")
    def click_ui(payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        result = runtime.executor.execute("click_ui", {"text": text})
        return result.model_dump(mode="json")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "version": "1.0.0",
            "running": runtime.running,
            "enabled": runtime.enabled,
            "game_id": runtime.game_id,
            "time": time.time(),
        }

    return app
