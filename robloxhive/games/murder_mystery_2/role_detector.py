from __future__ import annotations

from collections import deque

from robloxhive.games.murder_mystery_2.models import MM2Role, RoundPhase


class MM2RoleDetector:
    """Conservative OCR role/round detector.

    A role must repeat across frames before it becomes trusted so result screens
    or incidental text do not immediately enable combat.
    """

    ROLE_WORDS = {
        "murderer": MM2Role.MURDERER,
        "sheriff": MM2Role.SHERIFF,
        "innocent": MM2Role.INNOCENT,
        "hero": MM2Role.HERO,
    }

    def __init__(self, stable_frames: int = 2) -> None:
        self.stable_frames = max(1, stable_frames)
        self._history: deque[MM2Role] = deque(maxlen=self.stable_frames)
        self.current_role = MM2Role.UNKNOWN

    @staticmethod
    def phase(texts: list[str]) -> RoundPhase:
        text = " ".join(texts).lower()
        if any(x in text for x in ("waiting for your role", "you are", "role")):
            return RoundPhase.ROLE_REVEAL
        if any(x in text for x in ("victory", "murderer wins", "innocents win", "sheriff wins", "game over")):
            return RoundPhase.ROUND_END
        if any(x in text for x in ("waiting for players", "intermission", "game starts in", "next round")):
            return RoundPhase.LOBBY
        return RoundPhase.UNKNOWN

    def detect(self, texts: list[str]) -> tuple[MM2Role, float, RoundPhase]:
        lowered = [text.lower().strip() for text in texts if text.strip()]
        found = MM2Role.UNKNOWN

        # Prefer explicit "you are ..." text.
        joined = " ".join(lowered)
        for word, role in self.ROLE_WORDS.items():
            if f"you are {word}" in joined or f"you are the {word}" in joined:
                found = role
                break

        if found is MM2Role.UNKNOWN:
            for text in lowered:
                token = text.strip("! :.-")
                if token in self.ROLE_WORDS:
                    found = self.ROLE_WORDS[token]
                    break

        phase = self.phase(texts)
        if found is MM2Role.UNKNOWN:
            if phase in {RoundPhase.LOBBY, RoundPhase.ROUND_END}:
                self._history.clear()
                self.current_role = MM2Role.UNKNOWN
                return MM2Role.UNKNOWN, 0.0, phase
            if self.current_role is not MM2Role.UNKNOWN:
                return self.current_role, 0.94, RoundPhase.ROUND
            return MM2Role.UNKNOWN, 0.0, phase

        self._history.append(found)
        stable = len(self._history) == self.stable_frames and len(set(self._history)) == 1
        if stable:
            self.current_role = found
        return (
            self.current_role if stable else found,
            0.98 if stable else 0.62,
            RoundPhase.ROUND if stable else RoundPhase.ROLE_REVEAL,
        )
