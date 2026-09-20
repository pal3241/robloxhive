from __future__ import annotations

import re
from typing import Any

from robloxhive.shared.models import Goal, Plan, PlanStep


class KnowledgePlanner:
    """Turn a goal plus per-game knowledge into executable high-level steps.

    The planner never emits raw keyboard input. It emits semantic skills such as
    navigate, combat, interact, follow_player, quest, explore, or observe.
    """

    SECTION_ORDER = (
        "objectives",
        "progression",
        "strategies",
        "items",
        "locations",
        "enemies",
        "endgame",
    )

    def __init__(self, max_steps: int = 12, min_confidence: float = 0.25) -> None:
        self.max_steps = max_steps
        self.min_confidence = min_confidence

    @staticmethod
    def _skill_for(text: str) -> str:
        t = text.lower()
        if any(x in t for x in ("follow_player", "follow ", "ikuti ", "stay with", "keep up with")):
            return "follow_player"
        if any(x in t for x in ("combat", "kill", "defeat", "fight", "enemy", "boss", "bunuh", "lawan")):
            return "combat"
        if any(x in t for x in ("go to", "travel", "reach", "location", "station", "town", "pergi", "menuju")):
            return "navigate"
        if any(x in t for x in ("collect", "loot", "pickup", "gather", "ambil", "kumpul")):
            return "collect"
        if any(x in t for x in ("buy", "sell", "upgrade", "purchase", "beli", "jual")):
            return "interact"
        if any(x in t for x in ("quest", "mission", "daily", "task")):
            return "quest"
        if any(x in t for x in ("explore", "exploration", "find", "discover", "cari", "jelajah")):
            return "explore"
        return "observe_and_act"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
            if len(token) >= 3
        }

    def _score(self, goal_text: str, section: str, item: dict[str, Any]) -> float:
        text = str(item.get("text") or "")
        confidence = float(item.get("confidence") or 0.0)
        overlap = len(self._tokens(goal_text) & self._tokens(text))
        section_bonus = {
            "objectives": 0.24,
            "progression": 0.22,
            "strategies": 0.16,
            "endgame": 0.18,
            "items": 0.08,
            "locations": 0.08,
            "enemies": 0.08,
        }.get(section, 0.0)
        verified_bonus = 0.18 if item.get("verified_in_game") else 0.0
        return confidence + min(overlap * 0.08, 0.32) + section_bonus + verified_bonus

    def plan(self, game_id: int, goal: Goal, knowledge: dict[str, Any]) -> Plan:
        goal_text = " ".join(
            part for part in (goal.type, goal.target or "", str(goal.metadata.get("instruction", ""))) if part
        ).strip()

        direct_skill = self._skill_for(goal_text)
        direct_types = {"follow_player", "combat", "quest"}
        if direct_skill in direct_types and goal.type.lower() not in {"autonomous_progress", "complete_game"}:
            return Plan(
                game_id=game_id,
                goal=goal,
                knowledge_synthesizer=knowledge.get("synthesizer"),
                steps=[
                    PlanStep(
                        index=0,
                        skill=direct_skill,
                        instruction=goal_text or goal.type,
                        confidence=0.95,
                        success_condition="goal-specific verifier reports success",
                        knowledge_refs=[],
                    )
                ],
            )

        candidates: list[tuple[float, int, str, dict[str, Any]]] = []
        order = {name: idx for idx, name in enumerate(self.SECTION_ORDER)}
        for section in self.SECTION_ORDER:
            for item in knowledge.get(section, []) or []:
                if not isinstance(item, dict):
                    continue
                confidence = float(item.get("confidence") or 0.0)
                if confidence < self.min_confidence:
                    continue
                score = self._score(goal_text, section, item)
                candidates.append((score, order[section], section, item))

        # Prefer progression ordering for broad "finish/progress" goals, while
        # still allowing goal-specific relevance to move a step upward.
        broad = any(
            token in goal_text.lower()
            for token in ("complete", "finish", "progress", "tamat", "selesaikan", "autonomous")
        )
        if broad:
            candidates.sort(key=lambda row: (row[1], -row[0]))
        else:
            candidates.sort(key=lambda row: (-row[0], row[1]))

        seen: set[str] = set()
        steps: list[PlanStep] = []
        for score, _order, section, item in candidates:
            text = str(item.get("text") or "").strip()
            key = re.sub(r"\s+", " ", text.lower())
            if not text or key in seen:
                continue
            seen.add(key)
            refs = [str(ref) for ref in item.get("sources", []) if ref]
            steps.append(
                PlanStep(
                    index=len(steps),
                    skill=self._skill_for(text),
                    instruction=text,
                    confidence=min(1.0, float(item.get("confidence") or 0.0)),
                    success_condition="perception/action verifier confirms this step",
                    knowledge_refs=refs,
                    metadata={"section": section, "planner_score": round(score, 3)},
                )
            )
            if len(steps) >= self.max_steps:
                break

        if not steps:
            steps = [
                PlanStep(
                    index=0,
                    skill="observe_and_act",
                    instruction=goal_text or "Observe the game and determine the next useful action.",
                    confidence=0.25,
                    success_condition="new useful world-state or game knowledge is observed",
                    knowledge_refs=[],
                    metadata={"reason": "no applicable synthesized knowledge"},
                )
            ]

        return Plan(
            game_id=game_id,
            goal=goal,
            knowledge_synthesizer=knowledge.get("synthesizer"),
            steps=steps,
        )
