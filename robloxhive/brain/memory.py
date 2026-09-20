from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class GameMemory:
    """Durable memory isolated by Roblox Game/Universe ID."""

    def __init__(self, root: str | Path = "data/games") -> None:
        self.root = Path(root)

    def _dir(self, game_id: int) -> Path:
        return self.root / str(game_id)

    @staticmethod
    def _default(game_id: int) -> dict[str, Any]:
        return {
            "game_id": game_id,
            "game_name": None,
            "facts": [],
            "strategies": [],
            "failures": [],
            "research": {
                "source_count": 0,
                "extracted_count": 0,
                "last_researched_at": None,
                "latest_queries": [],
                "sources": [],
            },
            "knowledge_summary": {
                "available": False,
                "synthesizer": None,
                "confidence": 0.0,
                "generated_at": None,
                "section_counts": {},
            },
            "research_files": [],
            "updated_at": None,
        }

    def load_profile(self, game_id: int) -> dict[str, Any]:
        path = self._dir(game_id) / "profile.json"
        if not path.exists():
            return self._default(game_id)
        profile = self._default(game_id)
        profile.update(json.loads(path.read_text(encoding="utf-8")))
        return profile

    def save_profile(self, game_id: int, profile: dict[str, Any]) -> None:
        directory = self._dir(game_id)
        directory.mkdir(parents=True, exist_ok=True)
        profile["game_id"] = game_id
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp = directory / "profile.json.tmp"
        target = directory / "profile.json"
        temp.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(target)

    def remember(self, game_id: int, category: str, item: dict[str, Any]) -> None:
        profile = self.load_profile(game_id)
        profile.setdefault(category, []).append(item)
        self.save_profile(game_id, profile)

    def save_research(
        self,
        game_id: int,
        game_name: str,
        bundle: dict[str, Any],
    ) -> Path:
        directory = self._dir(game_id) / "research"
        directory.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = directory / f"research-{stamp}.json"
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(target)

        profile = self.load_profile(game_id)
        profile["game_name"] = game_name
        profile["research"] = {
            "source_count": bundle.get("source_count", 0),
            "extracted_count": bundle.get("extracted_count", 0),
            "last_researched_at": bundle.get("created_at"),
            "latest_queries": bundle.get("queries", []),
            "sources": [
                {
                    "title": source.get("title"),
                    "url": source.get("url"),
                    "quality": source.get("quality"),
                    "extracted": source.get("extracted"),
                    "query": source.get("query"),
                }
                for source in bundle.get("sources", [])
            ],
        }
        profile.setdefault("research_files", []).append(str(target))

        known = {
            (item.get("text"), item.get("source_url"))
            for item in profile.setdefault("facts", [])
            if isinstance(item, dict)
        }
        for item in bundle.get("candidate_knowledge", []):
            key = (item.get("text"), item.get("source_url"))
            if key not in known:
                profile["facts"].append(item)
                known.add(key)

        self.save_profile(game_id, profile)
        return target

    def save_knowledge(self, game_id: int, knowledge: dict[str, Any]) -> Path:
        directory = self._dir(game_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "knowledge.json"
        temp = directory / "knowledge.json.tmp"
        temp.write_text(json.dumps(knowledge, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(target)

        sections = (
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
        profile = self.load_profile(game_id)
        if knowledge.get("game_name"):
            profile["game_name"] = knowledge["game_name"]
        profile["knowledge_summary"] = {
            "available": True,
            "synthesizer": knowledge.get("synthesizer"),
            "confidence": knowledge.get("confidence", 0.0),
            "generated_at": knowledge.get("generated_at"),
            "section_counts": {
                section: len(knowledge.get(section, []))
                for section in sections
            },
            "warning": knowledge.get("synthesis_warning"),
        }
        self.save_profile(game_id, profile)
        return target

    def load_knowledge(self, game_id: int) -> dict[str, Any]:
        path = self._dir(game_id) / "knowledge.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def list_games(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        games: list[dict[str, Any]] = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or not directory.name.isdigit():
                continue
            profile_path = directory / "profile.json"
            if not profile_path.exists():
                continue
            try:
                profile = self.load_profile(int(directory.name))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            research = profile.get("research", {})
            knowledge = profile.get("knowledge_summary", {})
            games.append(
                {
                    "game_id": profile["game_id"],
                    "game_name": profile.get("game_name"),
                    "facts": len(profile.get("facts", [])),
                    "strategies": len(profile.get("strategies", [])),
                    "failures": len(profile.get("failures", [])),
                    "research_sources": research.get("source_count", 0),
                    "last_researched_at": research.get("last_researched_at"),
                    "knowledge_available": knowledge.get("available", False),
                    "knowledge_confidence": knowledge.get("confidence", 0.0),
                    "synthesizer": knowledge.get("synthesizer"),
                    "updated_at": profile.get("updated_at"),
                }
            )
        games.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return games
