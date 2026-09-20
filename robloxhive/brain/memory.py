from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GameMemory:
    """Simple per-game durable memory foundation.

    Later this can be backed by SQLite/vector retrieval without changing callers.
    """

    def __init__(self, root: str | Path = "data/games") -> None:
        self.root = Path(root)

    def _dir(self, game_id: int) -> Path:
        return self.root / str(game_id)

    def load_profile(self, game_id: int) -> dict[str, Any]:
        path = self._dir(game_id) / "profile.json"
        if not path.exists():
            return {"game_id": game_id, "facts": [], "strategies": [], "failures": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_profile(self, game_id: int, profile: dict[str, Any]) -> None:
        directory = self._dir(game_id)
        directory.mkdir(parents=True, exist_ok=True)
        temp = directory / "profile.json.tmp"
        target = directory / "profile.json"
        temp.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(target)

    def remember(self, game_id: int, category: str, item: dict[str, Any]) -> None:
        profile = self.load_profile(game_id)
        profile.setdefault(category, []).append(item)
        self.save_profile(game_id, profile)
