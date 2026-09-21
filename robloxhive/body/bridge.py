from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from robloxhive.body.skills import SkillExecutor
from robloxhive.shared.models import ActionResult, ActionStatus


class BodyBridge:
    """HTTP bridge for a Windows Body Node connected to a remote Brain Node."""

    def __init__(
        self,
        brain_url: str,
        executor: SkillExecutor,
        agent_id: str = "agent-01",
        timeout: float = 10.0,
        metadata: dict[str, Any] | None = None,
        executor_factory: Callable[[int], SkillExecutor] | None = None,
        rebind_factory: Callable[[int, int], tuple[SkillExecutor, dict[str, Any]]] | None = None,
    ) -> None:
        self.brain_url = brain_url.rstrip("/")
        self.executor = executor
        self.agent_id = agent_id
        self.timeout = timeout
        self.metadata = metadata or {}
        self.executor_factory = executor_factory
        self.rebind_factory = rebind_factory
        self.running = False
        self.armed = False
        self._last_register = 0.0

    def _json(self, path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.brain_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None

    def register(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_register < 5.0:
            return True
        try:
            description = self.executor.describe()
            live_metadata = {**self.metadata, "armed": self.armed}
            if os.name == "nt":
                try:
                    from robloxhive.body.discovery import discover_roblox_windows
                    live_metadata["windows"] = [
                        {"pid": w.pid, "hwnd": w.hwnd, "title": w.title, "alive": w.alive}
                        for w in discover_roblox_windows()
                    ]
                except Exception as exc:
                    live_metadata["window_scan_error"] = str(exc)
            self._json(
                "/api/body/register",
                method="POST",
                payload={
                    "agent_id": self.agent_id,
                    "skills": description["skills"],
                    "metadata": {
                        **live_metadata,
                        **description.get("metadata", {}),
                    },
                },
            )
            self._last_register = now
            return True
        except (URLError, HTTPError, TimeoutError, OSError):
            return False

    def poll_once(self, wait_s: float = 1.0) -> dict[str, Any] | None:
        if self.armed:
            self.executor.autonomy_tick()
        self.register()
        query = urlencode({"agent_id": self.agent_id, "timeout": max(0.0, min(wait_s, 5.0))})
        try:
            command = self._json(f"/api/body/commands/next?{query}")
        except (URLError, HTTPError, TimeoutError, OSError):
            return None

        if not command:
            return None
        payload = command.get("payload") or {}

        if command.get("type") == "PERCEPTION_PROBE":
            result = self.executor.probe(str(payload.get("label") or ""))
            self._json(
                "/api/body/perception-results",
                method="POST",
                payload={
                    "agent_id": self.agent_id,
                    "probe_id": payload.get("probe_id"),
                    "result": result,
                },
            )
            return command

        if command.get("type") == "NAVIGATION_PROBE":
            result = self.executor.navigation_probe()
            self._json(
                "/api/body/navigation-results",
                method="POST",
                payload={
                    "agent_id": self.agent_id,
                    "probe_id": payload.get("probe_id"),
                    "result": result,
                },
            )
            return command

        if command.get("type") == "BIND_INSTANCE":
            pid = int(payload.get("pid") or 0)
            result: dict[str, Any]
            if pid <= 0:
                result = {"ok": False, "error": "INVALID_PID"}
            elif self.rebind_factory is None:
                result = {"ok": False, "error": "REBIND_NOT_SUPPORTED"}
            else:
                try:
                    current_game = int(self.metadata.get("game_id") or 0)
                    executor, new_metadata = self.rebind_factory(pid, current_game)
                    try:
                        self.executor.direct_control({"action": "release"})
                    except Exception:
                        pass
                    self.executor = executor
                    self.metadata.update(new_metadata)
                    self.armed = True
                    self.register(force=True)
                    result = {
                        "ok": True,
                        "pid": self.metadata.get("pid"),
                        "hwnd": self.metadata.get("hwnd"),
                        "title": self.metadata.get("title"),
                    }
                except Exception as exc:
                    result = {
                        "ok": False,
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "pid": pid,
                    }
            self._json(
                "/api/body/bind-results",
                method="POST",
                payload={
                    "agent_id": self.agent_id,
                    "bind_id": payload.get("bind_id"),
                    "result": result,
                },
            )
            return command

        if command.get("type") == "DISARM":
            requested_pid = int(payload.get("pid") or 0)
            current_pid = int(self.metadata.get("pid") or 0)
            if requested_pid in {0, current_pid}:
                try:
                    self.executor.direct_control({"action": "release"})
                except Exception:
                    pass
                self.armed = False
                self.register(force=True)
            return command

        if command.get("type") == "JOIN_GAME":
            place_id = int(payload.get("place_id") or 0)
            result: dict[str, Any]
            if not self.armed:
                result = {"ok": False, "error": "BODY_NOT_ARMED"}
            elif place_id <= 0:
                result = {"ok": False, "error": "INVALID_PLACE_ID"}
            elif os.name != "nt":
                result = {"ok": False, "error": "JOIN_GAME_WINDOWS_ONLY"}
            else:
                try:
                    # Roblox protocol launch is intentionally generic: the
                    # dashboard chooses the place at runtime instead of binding
                    # a Body to one game in the CLI.
                    os.startfile(f"roblox://placeID={place_id}")
                    reconfigured = False
                    if self.executor_factory is not None:
                        self.executor = self.executor_factory(place_id)
                        self.metadata["game_id"] = place_id
                        reconfigured = True
                    self.register(force=True)
                    result = {
                        "ok": True,
                        "place_id": place_id,
                        "status": "launch_requested",
                        "executor_reconfigured": reconfigured,
                    }
                except Exception as exc:
                    result = {
                        "ok": False,
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "place_id": place_id,
                    }
            self._json(
                "/api/body/join-results",
                method="POST",
                payload={
                    "agent_id": self.agent_id,
                    "join_id": payload.get("join_id"),
                    "result": result,
                },
            )
            return command

        if command.get("type") == "DIRECT_INPUT":
            result = (
                self.executor.direct_control(payload)
                if self.armed
                else {"ok": False, "error": "BODY_NOT_ARMED"}
            )
            self._json(
                "/api/body/control-results",
                method="POST",
                payload={
                    "agent_id": self.agent_id,
                    "control_id": payload.get("control_id"),
                    "result": result,
                },
            )
            return command

        if command.get("type") == "GAME_CONTROL":
            result = self.executor.game_control(payload)
            self._json(
                "/api/body/game-control-results",
                method="POST",
                payload={
                    "agent_id": self.agent_id,
                    "control_id": payload.get("control_id"),
                    "result": result,
                },
            )
            return command

        if command.get("type") != "EXECUTE_SKILL":
            return command

        skill = str(payload.get("skill") or "")
        if self.armed:
            result = self.executor.execute(skill, payload)
        else:
            result = ActionResult(
                action=skill,
                status=ActionStatus.BLOCKED,
                error="BODY_NOT_ARMED",
                recoverable=True,
                details={"message": "Assign this Roblox window to the Body from Dashboard > Instances first."},
            )
        evidence = result.details.get("evidence") if isinstance(result.details, dict) else None
        evidence_payload = evidence if isinstance(evidence, dict) else result.details

        if payload.get("plan_id"):
            self._submit_plan_result(payload, result, evidence_payload)
        else:
            self._submit_manual_result(payload, result, evidence_payload)
        return command

    def _submit_plan_result(
        self,
        payload: dict[str, Any],
        result: ActionResult,
        evidence: dict[str, Any] | None,
    ) -> None:
        body = {
            "agent_id": self.agent_id,
            "plan_id": payload["plan_id"],
            "step_index": int(payload["step_index"]),
            "result": result.model_dump(mode="json"),
            "evidence": evidence or {},
        }
        self._json("/api/body/results", method="POST", payload=body)

    def _submit_manual_result(
        self,
        payload: dict[str, Any],
        result: ActionResult,
        evidence: dict[str, Any] | None,
    ) -> None:
        self._json(
            "/api/body/manual-results",
            method="POST",
            payload={
                "agent_id": self.agent_id,
                "test_id": payload.get("test_id"),
                "skill": payload.get("skill"),
                "result": result.model_dump(mode="json"),
                "evidence": evidence or {},
            },
        )

    def run_forever(self, idle_sleep_s: float = 0.03) -> None:
        self.running = True
        self.register(force=True)
        while self.running:
            command = self.poll_once(wait_s=0.15)
            if command is None:
                time.sleep(idle_sleep_s)

    def stop(self) -> None:
        self.running = False
