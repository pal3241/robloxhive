from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(slots=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b"
    timeout_s: float = 45.0
    temperature: float = 0.15
    max_context_memories: int = 18
    vision: bool = False


@dataclass(slots=True)
class AgentDecision:
    action: str = "observe"
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    done: bool = False
    remember: list[dict[str, Any]] = field(default_factory=list)
    interpretation: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class OllamaActor:
    """Ollama is the scenario-level actor, not merely a text summarizer."""

    SKILL_GUIDE = {
        "observe": "No input. Re-read current scene before deciding.",
        "wait": "payload: {seconds}; wait for loading/cooldown/state change.",
        "navigate": "payload: {target}; approach a visible/object detector target.",
        "collect": "payload: {target, key?}; approach and collect, then verify visual change.",
        "interact": "payload: {target?, key?}; approach target if supplied and interact.",
        "follow_player": "payload: {target:<exact Roblox username>}; exact username only.",
        "click_ui": "payload: {text, button?}; find visible OCR UI text and click it.",
        "press_key": "payload: {key, seconds?}; press a game/UI key.",
        "explore": "payload: {pattern?, seconds?}; move to gather new map information.",
        "aim": "payload: {target or username, lead_seconds?}; predictive aim at a visible target.",
        "combat": "payload: {target or username, weapon_key?, button?}; aim then attack visible target.",
    }

    SYSTEM = """You are the high-level brain of ONE Roblox automation bot.
You do not emit keyboard scan codes. Choose one semantic action that best fits the CURRENT scene.
Never assume an action succeeded: use the next observation and action result.
Use memory as evidence, but prefer current observations when they conflict.
Be conservative about player identity, team, enemy, and role. Do not invent usernames or visible objects.
For follow_player, the target MUST be the exact Roblox username provided by the user.
For combat, only target something that is observed or strongly remembered for the current game.
If the screen is loading, role is uncertain, UI is blocking gameplay, or required target is missing, choose observe/wait/explore rather than a random attack.
Return JSON only.

JSON schema:
{
  "action": "<one available skill or observe|wait>",
  "payload": { ... },
  "confidence": 0.0-1.0,
  "reason": "short operational reason",
  "done": false,
  "interpretation": {
    "roles": {},
    "teams": {},
    "enemies": [],
    "allies": []
  },
  "remember": [
    {
      "kind": "episodic|semantic|procedural|social|team|enemy|role|ui|map|strategy|failure|goal|action",
      "key": "optional stable key",
      "text": "short fact worth keeping",
      "confidence": 0.0-1.0,
      "importance": 0.0-1.0
    }
  ]
}
"""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self.last_latency_ms = 0
        self.last_error: str | None = None

    def _request(self, path: str, payload: dict[str, Any] | None = None, method: str = "POST") -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = Request(
            self.config.base_url.rstrip("/") + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=self.config.timeout_s) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}

    def health(self) -> dict[str, Any]:
        try:
            data = self._request("/api/tags", None, "GET")
            models = [
                str(item.get("name") or item.get("model") or "")
                for item in data.get("models", [])
            ]
            return {
                "ok": True,
                "url": self.config.base_url,
                "model": self.config.model,
                "model_available": self.config.model in models
                or any(name.startswith(self.config.model + ":") for name in models),
                "models": models[:40],
                "latency_ms": self.last_latency_ms,
                "last_error": self.last_error,
            }
        except Exception as exc:
            return {
                "ok": False,
                "url": self.config.base_url,
                "model": self.config.model,
                "models": [],
                "latency_ms": self.last_latency_ms,
                "last_error": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            text = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*" + re.escape(fence) + r"$", "", text)
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        if start < 0:
            return {}
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            ch = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : index + 1])
                        return value if isinstance(value, dict) else {}
                    except json.JSONDecodeError:
                        return {}
        return {}

    def decide(
        self,
        *,
        goal: dict[str, Any],
        world: dict[str, Any],
        skills: list[str],
        memories: list[dict[str, Any]],
        map_summary: dict[str, Any],
        recent_actions: list[dict[str, Any]],
        screenshot_base64: str | None = None,
    ) -> AgentDecision:
        skill_set = sorted(set(skills))
        user_payload = {
            "goal": goal,
            "available_skills": skill_set,
            "skill_contracts": {
                skill: self.SKILL_GUIDE.get(skill, "Use only when its target and success condition are observable.")
                for skill in skill_set
            },
            "world": world,
            "relevant_memory": memories[: self.config.max_context_memories],
            "semantic_map": map_summary,
            "recent_actions": recent_actions[-8:],
        }

        content = json.dumps(user_payload, ensure_ascii=False)
        message: dict[str, Any] = {"role": "user", "content": content}
        if self.config.vision and screenshot_base64:
            message["images"] = [screenshot_base64]

        started = time.monotonic()
        try:
            response = self._request(
                "/api/chat",
                {
                    "model": self.config.model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": self.config.temperature},
                    "messages": [
                        {"role": "system", "content": self.SYSTEM},
                        message,
                    ],
                },
            )
            self.last_latency_ms = int((time.monotonic() - started) * 1000)
            text = str((response.get("message") or {}).get("content") or "")
            parsed = self._extract_json(text)
            self.last_error = None
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            self.last_latency_ms = int((time.monotonic() - started) * 1000)
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(self.last_error) from exc

        action = str(parsed.get("action") or "observe").strip().lower()
        allowed = set(skills) | {"observe", "wait"}
        if action not in allowed:
            action = "observe"

        payload = parsed.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        remember = parsed.get("remember")
        if not isinstance(remember, list):
            remember = []
        interpretation = parsed.get("interpretation")
        if not isinstance(interpretation, dict):
            interpretation = {}
        try:
            confidence = max(0.0, min(float(parsed.get("confidence", 0.0)), 1.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return AgentDecision(
            action=action,
            payload=payload,
            confidence=confidence,
            reason=str(parsed.get("reason") or "")[:500],
            done=bool(parsed.get("done", False)),
            remember=[item for item in remember if isinstance(item, dict)][:12],
            interpretation=interpretation,
            raw=parsed,
        )
