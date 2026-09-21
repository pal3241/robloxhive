from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path
from collections import deque
from dataclasses import asdict
from typing import Any

from robloxhive.brain.ollama_actor import OllamaActor, OllamaConfig
from robloxhive.core.memory import CognitiveMemory
from robloxhive.core.semantic_map import SemanticMap
from robloxhive.core.world_model import WorldModel
from robloxhive.shared.models import ActionStatus


class SingleBotRuntime:
    """Local-first one-bot runtime.

    Vision, controls, safety, memory and dashboard live on the Windows laptop.
    Ollama can be localhost or any reachable URL (for example a phone).
    """

    def __init__(
        self,
        executor: Any,
        *,
        game_id: int = 0,
        ollama_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "qwen2.5:3b",
        memory_path: str = "data/memory/robloxhive.db",
        decision_interval_s: float = 0.55,
        vision_llm: bool = False,
        executor_factory: Any | None = None,
        game_context_provider: Any | None = None,
    ) -> None:
        self.executor = executor
        self.executor_factory = executor_factory
        self.game_context_provider = game_context_provider
        self.game_id = int(game_id or 0)
        self.memory = CognitiveMemory(memory_path)
        self.semantic_map = SemanticMap(self.memory, self.game_id)
        self.world = WorldModel(executor, self.game_id)
        self.actor = OllamaActor(
            OllamaConfig(
                base_url=ollama_url,
                model=ollama_model,
                vision=vision_llm,
            )
        )
        self.decision_interval_s = max(0.2, float(decision_interval_s))
        self.goal: dict[str, Any] = {
            "type": "idle",
            "instruction": "Wait for a user goal.",
            "target": None,
        }
        self.enabled = False
        self.running = False
        self.paused_reason: str | None = "waiting_for_goal"
        self.last_decision: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_world: dict[str, Any] = {}
        self.recent_actions: deque[dict[str, Any]] = deque(maxlen=24)
        self.error_count = 0
        self.decision_count = 0
        self.success_count = 0
        self.failure_count = 0
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._last_tick = 0.0
        self._knowledge_mtime: dict[int, float] = {}
        self._last_game_context_check = 0.0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="robloxhive-single-bot", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        self.enabled = False
        try:
            self.executor.direct_control({"action": "release", "mode": "reliable"})
        except Exception:
            pass

    def set_goal(
        self,
        instruction: str,
        *,
        goal_type: str = "custom",
        target: str | None = None,
    ) -> dict[str, Any]:
        goal = {
            "type": goal_type.strip() or "custom",
            "instruction": instruction.strip(),
            "target": target.strip() if isinstance(target, str) and target.strip() else None,
            "created_at": time.time(),
        }
        if goal["type"] == "follow_player" and not goal["target"]:
            raise ValueError("follow_player requires an exact Roblox username")
        with self._lock:
            self.goal = goal
            self.paused_reason = None
        self.memory.remember(
            "goal",
            f"{goal['type']}: {goal['instruction']}",
            game_id=self.game_id,
            key=goal.get("target"),
            data=goal,
            confidence=1.0,
            importance=0.9,
        )
        return goal

    def configure_ollama(
        self,
        *,
        url: str | None = None,
        model: str | None = None,
        vision: bool | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if url:
                self.actor.config.base_url = url.rstrip("/")
            if model:
                self.actor.config.model = model.strip()
            if vision is not None:
                self.actor.config.vision = bool(vision)
        return self.actor.health()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.paused_reason = "paused_by_user"
            try:
                self.executor.direct_control({"action": "release", "mode": "reliable"})
            except Exception:
                pass
        elif self.goal.get("type") == "idle":
            self.paused_reason = "waiting_for_goal"
        else:
            self.paused_reason = None

    def set_game_id(self, game_id: int) -> None:
        game_id = int(game_id or 0)
        if game_id == self.game_id:
            return
        self.semantic_map.save()
        if self.executor_factory is not None:
            try:
                self.executor = self.executor_factory(game_id)
            except Exception as exc:
                self.memory.remember(
                    "failure",
                    f"Could not rebuild executor for game {game_id}: {type(exc).__name__}: {exc}",
                    game_id=self.game_id,
                    key="executor_rebuild",
                    confidence=1.0,
                    success=False,
                    importance=0.8,
                )
                raise
        self.game_id = game_id
        self.semantic_map = SemanticMap(self.memory, self.game_id)
        self.world = WorldModel(self.executor, self.game_id)
        self.memory.remember(
            "episodic",
            f"Switched current game context to {self.game_id}",
            game_id=self.game_id,
            importance=0.65,
        )

    def _screenshot(self) -> str | None:
        if not self.actor.config.vision:
            return None
        frame_source = getattr(self.executor, "frame_source", None)
        if frame_source is None:
            return None
        try:
            import cv2
            frame = frame_source.capture()
            h, w = frame.shape[:2]
            max_w = 768
            if w > max_w:
                scale = max_w / float(w)
                frame = cv2.resize(frame, (max_w, max(1, int(h * scale))))
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 62])
            if not ok:
                return None
            return base64.b64encode(encoded.tobytes()).decode("ascii")
        except Exception:
            return None

    def _sync_legacy_knowledge(self) -> None:
        """Import the existing Internet Learning knowledge into cognitive memory."""
        path = Path("data/games") / str(self.game_id) / "knowledge.json"
        if not path.exists():
            return
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if self._knowledge_mtime.get(self.game_id) == mtime:
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        section_kind = {
            "objectives": "semantic",
            "progression": "procedural",
            "mechanics": "semantic",
            "items": "semantic",
            "enemies": "enemy",
            "locations": "map",
            "strategies": "strategy",
            "common_mistakes": "failure",
            "endgame": "procedural",
        }
        for section, kind in section_kind.items():
            for index, item in enumerate(data.get(section, []) or []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                try:
                    confidence = float(item.get("confidence", 0.55))
                except (TypeError, ValueError):
                    confidence = 0.55
                import hashlib
                stable_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                self.memory.upsert_fact(
                    kind,
                    f"research:{section}:{index}:{stable_id}",
                    text,
                    game_id=self.game_id,
                    data={
                        "section": section,
                        "sources": item.get("sources", []),
                        "verified_in_game": item.get("verified_in_game", False),
                        "source": "internet_learning",
                    },
                    confidence=max(0.1, min(confidence, 0.95)),
                    importance=0.55 if not item.get("verified_in_game") else 0.8,
                )
                if section == "locations":
                    self.semantic_map.observe_landmark(
                        text[:120],
                        kind="researched_location",
                        confidence=max(0.1, min(confidence, 0.85)),
                        value=0.5,
                        metadata={"source": "internet_learning"},
                    )
        self._knowledge_mtime[self.game_id] = mtime

    def _memory_query(self, world: dict[str, Any]) -> str:
        ui = " ".join(world.get("ui_text", [])[:15])
        entities = " ".join(str(x.get("label", "")) for x in world.get("entities", [])[:20])
        return " ".join(
            str(x)
            for x in (
                self.goal.get("type"),
                self.goal.get("instruction"),
                self.goal.get("target"),
                ui,
                entities,
            )
            if x
        )

    def _refresh_game_context(self) -> None:
        if self.game_context_provider is None:
            return
        now = time.monotonic()
        if now - self._last_game_context_check < 3.0:
            return
        self._last_game_context_check = now
        try:
            context = self.game_context_provider() or {}
            place_id = int(context.get("place_id") or 0)
        except Exception:
            return
        if place_id > 0 and place_id != self.game_id:
            self.set_game_id(place_id)

    def _remember_interpretation(self, interpretation: dict[str, Any]) -> None:
        roles = interpretation.get("roles")
        if isinstance(roles, dict):
            for subject, role in roles.items():
                self.memory.upsert_fact(
                    "role",
                    str(subject),
                    f"{subject} role is {role}",
                    game_id=self.game_id,
                    data={"subject": subject, "role": role, "source": "ollama_world_interpretation"},
                    confidence=0.65,
                    importance=0.7,
                )
        teams = interpretation.get("teams")
        if isinstance(teams, dict):
            for team, members in teams.items():
                if not isinstance(members, list):
                    continue
                for member in members:
                    self.memory.relate(
                        str(member),
                        "member_of_team",
                        str(team),
                        game_id=self.game_id,
                        confidence=0.6,
                    )
        for relation_name, kind in (("enemies", "enemy"), ("allies", "team")):
            values = interpretation.get(relation_name)
            if not isinstance(values, list):
                continue
            for value in values:
                text = str(value).strip()
                if not text:
                    continue
                self.memory.upsert_fact(
                    kind,
                    text.lower(),
                    f"{text} classified as {relation_name[:-1]}",
                    game_id=self.game_id,
                    data={"relation": relation_name, "source": "ollama_world_interpretation"},
                    confidence=0.6,
                    importance=0.75 if kind == "enemy" else 0.6,
                )

    def _remember_llm_items(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            kind = str(item.get("kind") or "episodic").lower()
            if kind not in self.memory.KINDS:
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                confidence = max(0.0, min(float(item.get("confidence", 0.5)), 1.0))
                importance = max(0.0, min(float(item.get("importance", 0.5)), 1.0))
            except (TypeError, ValueError):
                confidence, importance = 0.5, 0.5
            self.memory.remember(
                kind,
                text,
                game_id=self.game_id,
                key=str(item.get("key") or "") or None,
                data={"source": "ollama_interpretation"},
                confidence=confidence,
                importance=importance,
            )
            if kind == "map":
                name = str(item.get("key") or text).strip()
                if name:
                    self.semantic_map.observe_landmark(
                        name,
                        kind=str(item.get("map_kind") or "unknown"),
                        confidence=confidence,
                        danger=float(item.get("danger") or 0.0),
                        value=float(item.get("value") or importance),
                        metadata={"source": "ollama_interpretation"},
                    )

    def _record_result(self, action: str, payload: dict[str, Any], result: Any, reason: str) -> dict[str, Any]:
        if hasattr(result, "model_dump"):
            row = result.model_dump(mode="json")
        elif isinstance(result, dict):
            row = dict(result)
        else:
            row = {"status": "unknown", "details": {"value": str(result)}}
        success = str(row.get("status")) == ActionStatus.SUCCESS.value or row.get("ok") is True
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        event = {
            "ts": time.time(),
            "action": action,
            "payload": payload,
            "reason": reason,
            "result": row,
            "success": success,
        }
        self.recent_actions.append(event)
        self.last_result = row
        target = str(payload.get("target") or payload.get("label") or "").strip()
        previous_landmark = self.semantic_map.current
        if success and action in {"navigate", "collect", "interact"} and target:
            self.semantic_map.observe_landmark(
                target,
                kind="objective" if action == "navigate" else "interaction",
                confidence=0.72,
                value=0.55,
                metadata={"last_action": action},
            )
            if previous_landmark and previous_landmark != target.lower():
                self.semantic_map.transition(
                    previous_landmark,
                    target,
                    seconds=max(0.05, float(row.get("duration_ms") or 0) / 1000.0),
                    success=True,
                )
        self.memory.remember(
            "action" if success else "failure",
            f"{action}: {'success' if success else 'failed'}",
            game_id=self.game_id,
            key=action,
            data=event,
            confidence=0.85,
            success=success,
            importance=0.7 if not success else 0.55,
        )
        return event

    def _execute(self, action: str, payload: dict[str, Any], reason: str) -> dict[str, Any]:
        if action in {"observe", "wait"} and action not in self.executor.available():
            seconds = max(0.05, min(float(payload.get("seconds") or 0.25), 2.0))
            if action == "wait":
                time.sleep(seconds)
            result = {"ok": True, "status": "success", "details": {"passive": True}}
        else:
            result = self.executor.execute(action, payload)
        return self._record_result(action, payload, result, reason)

    def tick(self) -> None:
        if not self.enabled or self.goal.get("type") == "idle":
            return

        self._refresh_game_context()
        self._sync_legacy_knowledge()
        snapshot = self.world.observe()
        world = snapshot.compact()
        self.last_world = world

        query = self._memory_query(world)
        memories = self.memory.retrieve(query, game_id=self.game_id, limit=18)
        screenshot = self._screenshot()

        try:
            decision = self.actor.decide(
                goal=dict(self.goal),
                world=world,
                skills=self.executor.available(),
                memories=memories,
                map_summary=self.semantic_map.summary(),
                recent_actions=list(self.recent_actions),
                screenshot_base64=screenshot,
            )
        except Exception as exc:
            self.error_count += 1
            self.paused_reason = "ollama_unavailable"
            self.last_decision = {
                "action": "observe",
                "confidence": 0.0,
                "reason": str(exc),
                "error": True,
            }
            time.sleep(min(2.0, self.decision_interval_s * 2.0))
            return

        self.paused_reason = None
        self.decision_count += 1
        self.world.apply_interpretation(decision.interpretation)
        self._remember_interpretation(decision.interpretation)
        self._remember_llm_items(decision.remember)
        if decision.action == "follow_player" and self.goal.get("type") == "follow_player":
            username = str(self.goal.get("target") or "").strip()
            if username:
                decision.payload["target"] = username
                decision.payload["username"] = username

        self.last_decision = {
            "action": decision.action,
            "payload": decision.payload,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "done": decision.done,
            "interpretation": decision.interpretation,
        }

        if decision.confidence < 0.30 and decision.action not in {"observe", "wait"}:
            event = self._execute(
                "observe",
                {},
                f"Low confidence ({decision.confidence:.2f}); observe before acting.",
            )
        else:
            event = self._execute(decision.action, decision.payload, decision.reason)

        if decision.done:
            self.enabled = False
            self.paused_reason = "goal_completed_by_actor"
            self.memory.remember(
                "episodic",
                f"Goal completed: {self.goal.get('instruction')}",
                game_id=self.game_id,
                data={"goal": self.goal, "last_event": event},
                confidence=max(0.5, decision.confidence),
                success=True,
                importance=0.9,
            )

    def _loop(self) -> None:
        while self.running:
            started = time.monotonic()
            try:
                self.tick()
            except Exception as exc:
                self.error_count += 1
                self.paused_reason = f"runtime_error:{type(exc).__name__}"
                try:
                    self.executor.direct_control({"action": "release", "mode": "reliable"})
                except Exception:
                    pass
            elapsed = time.monotonic() - started
            time.sleep(max(0.04, self.decision_interval_s - elapsed))

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "enabled": self.enabled,
                "paused_reason": self.paused_reason,
                "game_id": self.game_id,
                "goal": dict(self.goal),
                "skills": self.executor.available(),
                "last_decision": self.last_decision,
                "last_result": self.last_result,
                "last_world": self.last_world,
                "recent_actions": list(self.recent_actions)[-12:],
                "ollama": {
                    "url": self.actor.config.base_url,
                    "model": self.actor.config.model,
                    "vision": self.actor.config.vision,
                    "latency_ms": self.actor.last_latency_ms,
                    "last_error": self.actor.last_error,
                },
                "memory": self.memory.stats(),
                "map": self.semantic_map.summary(),
                "stats": {
                    "decisions": self.decision_count,
                    "successes": self.success_count,
                    "failures": self.failure_count,
                    "errors": self.error_count,
                },
            }
