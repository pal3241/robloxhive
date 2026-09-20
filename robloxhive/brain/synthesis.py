from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


SECTIONS = (
    "objectives",
    "progression",
    "mechanics",
    "items",
    "enemies",
    "locations",
    "strategies",
    "common_mistakes",
    "endgame",
    "unknowns",
    "conflicts",
)


def _entry(text: str, confidence: float, sources: list[str]) -> dict[str, Any]:
    return {
        "text": text.strip(),
        "confidence": max(0.0, min(1.0, round(confidence, 2))),
        "sources": sources,
        "verified_in_game": False,
    }


def normalize_knowledge(data: dict[str, Any], game_name: str, synthesizer: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "game_name": game_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthesizer": synthesizer,
        "overview": str(data.get("overview") or "").strip(),
    }
    for section in SECTIONS:
        rows = data.get(section, [])
        if not isinstance(rows, list):
            rows = []
        normalized = []
        for row in rows[:80]:
            if isinstance(row, str):
                normalized.append(_entry(row, 0.45, []))
                continue
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or row.get("name") or "").strip()
            if not text:
                continue
            try:
                confidence = float(row.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            sources = row.get("sources", [])
            if not isinstance(sources, list):
                sources = []
            normalized.append(_entry(text, confidence, [str(x) for x in sources[:8]]))
        out[section] = normalized

    confidences = [
        item["confidence"]
        for section in SECTIONS
        for item in out[section]
        if section != "conflicts"
    ]
    out["confidence"] = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    return out


class HeuristicKnowledgeSynthesizer:
    name = "heuristic"

    def synthesize(self, game_name: str, bundle: dict[str, Any]) -> dict[str, Any]:
        sources = bundle.get("sources", [])
        result: dict[str, Any] = {"overview": ""}
        for section in SECTIONS:
            result[section] = []

        snippets: list[str] = []
        for idx, source in enumerate(sources, start=1):
            source_id = f"S{idx}"
            snippet = str(source.get("snippet") or "").strip()
            if not snippet:
                content = str(source.get("content") or "").strip()
                snippet = content[:700]
            if not snippet:
                continue
            snippets.append(snippet)
            query = str(source.get("query") or "").lower()
            quality = float(source.get("quality") or 0.5)
            confidence = quality * 0.72

            if any(word in query for word in ("beginner", "tutorial", "progression", "walkthrough")):
                result["progression"].append(_entry(snippet[:700], confidence, [source_id]))
            if any(word in query for word in ("tips", "tricks", "strategy")):
                result["strategies"].append(_entry(snippet[:700], confidence, [source_id]))
            if any(word in query for word in ("items", "weapons", "classes", "upgrades")):
                result["items"].append(_entry(snippet[:700], confidence, [source_id]))
            if any(word in query for word in ("ending", "complete", "how to win")):
                result["endgame"].append(_entry(snippet[:700], confidence, [source_id]))

        if snippets:
            result["overview"] = " ".join(snippets[:3])[:1400]
        if not result["progression"]:
            for idx, item in enumerate(bundle.get("candidate_knowledge", [])[:6], start=1):
                text = str(item.get("text") or "").strip()
                if text:
                    result["progression"].append(
                        _entry(text, float(item.get("confidence") or 0.4), [f"C{idx}"])
                    )
        return normalize_knowledge(result, game_name, self.name)


class OllamaKnowledgeSynthesizer:
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout: int = 180) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _corpus(bundle: dict[str, Any]) -> tuple[str, dict[str, str]]:
        blocks: list[str] = []
        source_map: dict[str, str] = {}
        for idx, source in enumerate(bundle.get("sources", [])[:14], start=1):
            sid = f"S{idx}"
            url = str(source.get("url") or "")
            source_map[sid] = url
            body = str(source.get("content") or source.get("snippet") or "").strip()
            if not body:
                continue
            blocks.append(
                f"[{sid}] TITLE: {source.get('title', '')}\n"
                f"URL: {url}\n"
                f"SEARCH: {source.get('query', '')}\n"
                f"QUALITY: {source.get('quality', 0.5)}\n"
                f"CONTENT:\n{body[:5500]}"
            )
        return "\n\n---\n\n".join(blocks), source_map

    def synthesize(self, game_name: str, bundle: dict[str, Any]) -> dict[str, Any]:
        corpus, source_map = self._corpus(bundle)
        if not corpus:
            return HeuristicKnowledgeSynthesizer().synthesize(game_name, bundle)

        schema = {
            "overview": "short explanation of the game and main loop",
            "objectives": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "progression": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "mechanics": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "items": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "enemies": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "locations": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "strategies": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "common_mistakes": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "endgame": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "unknowns": [{"text": "...", "confidence": 0.0, "sources": ["S1"]}],
            "conflicts": [{"text": "...", "confidence": 0.0, "sources": ["S1", "S2"]}],
        }
        system = (
            "You are RobloxHive's game knowledge compiler. Convert web research into "
            "compact operational knowledge for an autonomous game agent. Never invent facts. "
            "Keep conflicting claims separate and list them under conflicts. Treat web facts "
            "as unverified until gameplay confirms them. Cite only supplied source IDs. "
            "Return JSON only."
        )
        user = (
            f"GAME: {game_name}\n"
            f"RESEARCH OBJECTIVE: {bundle.get('objective', '')}\n\n"
            "Build knowledge from beginner/start through late game/completion. Include goals, "
            "progression order, mechanics, useful items, enemies, locations, strategies, common "
            "mistakes, and endgame/win conditions when supported. Confidence must reflect source "
            "quality and agreement.\n\n"
            f"JSON SHAPE:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"SOURCES:\n{corpus}"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0.1},
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama synthesis failed: {exc}") from exc

        content = str((raw.get("message") or {}).get("content") or "").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON knowledge") from exc

        knowledge = normalize_knowledge(parsed, game_name, f"ollama:{self.model}")
        knowledge["source_map"] = source_map
        return knowledge


class AutoKnowledgeSynthesizer:
    """Use remote/local Ollama when configured, otherwise deterministic fallback."""

    def __init__(self) -> None:
        self.model = os.getenv("ROBLOXHIVE_LLM_MODEL", "").strip()
        self.url = os.getenv("ROBLOXHIVE_OLLAMA_URL", "http://127.0.0.1:11434").strip()
        self.fallback = HeuristicKnowledgeSynthesizer()

    @property
    def name(self) -> str:
        return f"ollama:{self.model}" if self.model else self.fallback.name

    def synthesize(self, game_name: str, bundle: dict[str, Any]) -> dict[str, Any]:
        if not self.model:
            return self.fallback.synthesize(game_name, bundle)
        try:
            return OllamaKnowledgeSynthesizer(self.model, self.url).synthesize(game_name, bundle)
        except RuntimeError as exc:
            result = self.fallback.synthesize(game_name, bundle)
            result["synthesizer"] = f"heuristic-fallback ({type(exc).__name__})"
            result["synthesis_warning"] = str(exc)
            return result
