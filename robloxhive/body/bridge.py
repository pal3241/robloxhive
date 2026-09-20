from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from robloxhive.body.skills import SkillExecutor
from robloxhive.shared.models import ActionResult


class BodyBridge:
    """HTTP bridge for a Windows Body Node connected to a remote Brain Node."""

    def __init__(
        self,
        brain_url: str,
        executor: SkillExecutor,
        agent_id: str = "agent-01",
        timeout: float = 10.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.brain_url = brain_url.rstrip("/")
        self.executor = executor
        self.agent_id = agent_id
        self.timeout = timeout
        self.metadata = metadata or {}
        self.running = False
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
            self._json(
                "/api/body/register",
                method="POST",
                payload={
                    "agent_id": self.agent_id,
                    "skills": description["skills"],
                    "metadata": {
                        **self.metadata,
                        **description.get("metadata", {}),
                    },
                },
            )
            self._last_register = now
            return True
        except (URLError, HTTPError, TimeoutError, OSError):
            return False

    def poll_once(self, wait_s: float = 1.0) -> dict[str, Any] | None:
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

        if command.get("type") != "EXECUTE_SKILL":
            return command

        skill = str(payload.get("skill") or "")
        result = self.executor.execute(skill, payload)
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

    def run_forever(self, idle_sleep_s: float = 0.15) -> None:
        self.running = True
        self.register(force=True)
        while self.running:
            command = self.poll_once(wait_s=1.0)
            if command is None:
                time.sleep(idle_sleep_s)

    def stop(self) -> None:
        self.running = False
