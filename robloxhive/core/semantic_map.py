from __future__ import annotations

import heapq
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

from robloxhive.core.memory import CognitiveMemory


@dataclass(slots=True)
class Landmark:
    name: str
    kind: str = "unknown"
    x: float | None = None
    y: float | None = None
    confidence: float = 0.5
    visits: int = 0
    danger: float = 0.0
    value: float = 0.0
    last_seen: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticMap:
    """Persistent topological map for arbitrary Roblox games.

    It does not pretend screen pixels are exact world coordinates. Instead it
    stores semantic landmarks and learned transitions between them, including
    travel cost, danger, success rate, and useful objects/UI associated with a
    place. This works across games where exact 3D coordinates are unavailable.
    """

    def __init__(self, memory: CognitiveMemory, game_id: int = 0) -> None:
        self.memory = memory
        self.game_id = int(game_id or 0)
        self.landmarks: dict[str, Landmark] = {}
        self.edges: dict[str, dict[str, dict[str, float]]] = {}
        self.current: str | None = None
        self._load()

    def _state_key(self) -> str:
        return f"semantic_map:{self.game_id}"

    def _load(self) -> None:
        raw = self.memory.get_state(self._state_key(), {})
        for name, row in (raw.get("landmarks") or {}).items():
            try:
                self.landmarks[name] = Landmark(**row)
            except TypeError:
                continue
        self.edges = raw.get("edges") or {}
        self.current = raw.get("current")

    def save(self) -> None:
        self.memory.set_state(
            self._state_key(),
            {
                "landmarks": {
                    name: {
                        "name": item.name,
                        "kind": item.kind,
                        "x": item.x,
                        "y": item.y,
                        "confidence": item.confidence,
                        "visits": item.visits,
                        "danger": item.danger,
                        "value": item.value,
                        "last_seen": item.last_seen,
                        "metadata": item.metadata,
                    }
                    for name, item in self.landmarks.items()
                },
                "edges": self.edges,
                "current": self.current,
            },
        )

    def observe_landmark(
        self,
        name: str,
        *,
        kind: str = "unknown",
        confidence: float = 0.6,
        danger: float | None = None,
        value: float | None = None,
        metadata: dict[str, Any] | None = None,
        set_current: bool = True,
    ) -> Landmark:
        key = name.strip().lower()
        item = self.landmarks.get(key)
        if item is None:
            item = Landmark(name=name.strip(), kind=kind, confidence=confidence)
            self.landmarks[key] = item
        item.last_seen = time.time()
        item.visits += 1
        item.confidence = max(item.confidence, max(0.0, min(confidence, 1.0)))
        if kind and kind != "unknown":
            item.kind = kind
        if danger is not None:
            item.danger = max(0.0, min(float(danger), 1.0))
        if value is not None:
            item.value = max(0.0, min(float(value), 1.0))
        if metadata:
            item.metadata.update(metadata)
        if set_current:
            self.current = key
        self.memory.upsert_fact(
            "map",
            key,
            f"Landmark {item.name}: kind={item.kind}, danger={item.danger:.2f}, value={item.value:.2f}",
            game_id=self.game_id,
            data={
                "name": item.name,
                "kind": item.kind,
                "danger": item.danger,
                "value": item.value,
                "visits": item.visits,
                "metadata": item.metadata,
            },
            confidence=item.confidence,
            importance=max(0.45, item.value, item.danger),
        )
        self.save()
        return item

    def transition(
        self,
        source: str,
        target: str,
        *,
        seconds: float,
        success: bool,
        danger: float = 0.0,
    ) -> None:
        a, b = source.strip().lower(), target.strip().lower()
        edge = self.edges.setdefault(a, {}).setdefault(
            b,
            {"attempts": 0.0, "successes": 0.0, "seconds": 0.0, "danger": 0.0},
        )
        edge["attempts"] += 1.0
        edge["successes"] += 1.0 if success else 0.0
        n = edge["attempts"]
        edge["seconds"] += (max(0.05, float(seconds)) - edge["seconds"]) / n
        edge["danger"] += (max(0.0, min(float(danger), 1.0)) - edge["danger"]) / n
        if success:
            self.current = b
        self.memory.remember(
            "map",
            f"Route {source} -> {target}: {'success' if success else 'failed'}",
            game_id=self.game_id,
            key=f"{a}->{b}",
            data=dict(edge),
            confidence=min(0.95, 0.45 + edge["attempts"] * 0.05),
            success=success,
            importance=0.55,
        )
        self.save()

    @staticmethod
    def _edge_cost(edge: dict[str, float]) -> float:
        attempts = max(1.0, float(edge.get("attempts", 1.0)))
        success_rate = float(edge.get("successes", 0.0)) / attempts
        seconds = max(0.1, float(edge.get("seconds", 1.0)))
        danger = max(0.0, min(float(edge.get("danger", 0.0)), 1.0))
        return seconds * (1.0 + danger * 2.2) * (1.0 + (1.0 - success_rate) * 1.4)

    def route(self, start: str, goal: str) -> dict[str, Any]:
        start, goal = start.strip().lower(), goal.strip().lower()
        if start == goal:
            return {"found": True, "path": [start], "cost": 0.0}
        queue: list[tuple[float, str]] = [(0.0, start)]
        distance = {start: 0.0}
        parent: dict[str, str] = {}

        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                break
            if cost > distance.get(node, math.inf):
                continue
            for nxt, edge in self.edges.get(node, {}).items():
                new_cost = cost + self._edge_cost(edge)
                if new_cost < distance.get(nxt, math.inf):
                    distance[nxt] = new_cost
                    parent[nxt] = node
                    heapq.heappush(queue, (new_cost, nxt))

        if goal not in distance:
            return {"found": False, "path": [], "cost": None}

        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return {"found": True, "path": path, "cost": round(distance[goal], 3)}

    def best_landmarks(self, *, safe: bool = False, limit: int = 12) -> list[dict[str, Any]]:
        rows = list(self.landmarks.values())
        rows.sort(
            key=lambda item: (
                (item.value * 1.4 - item.danger * (2.0 if safe else 0.8)),
                item.confidence,
                item.visits,
            ),
            reverse=True,
        )
        return [
            {
                "name": item.name,
                "kind": item.kind,
                "confidence": item.confidence,
                "danger": item.danger,
                "value": item.value,
                "visits": item.visits,
                "metadata": item.metadata,
            }
            for item in rows[: max(1, min(limit, 50))]
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "current": self.current,
            "landmarks": len(self.landmarks),
            "edges": sum(len(v) for v in self.edges.values()),
            "best": self.best_landmarks(limit=8),
        }
