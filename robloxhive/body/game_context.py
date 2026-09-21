from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote


_PLACE_PATTERNS = (
    re.compile(r"(?i)place(?:id|ID)[\s:=\"']+(\d{3,})"),
    re.compile(r"(?i)placeId%3D(\d{3,})"),
    re.compile(r"(?i)placeId%253D(\d{3,})"),
)
_UNIVERSE_PATTERNS = (
    re.compile(r"(?i)universe(?:id|ID)[\s:=\"']+(\d{3,})"),
    re.compile(r"(?i)universeId%3D(\d{3,})"),
)


def _decode_command_line(parts: list[str]) -> str:
    text = " ".join(str(part) for part in parts)
    # Roblox launch URLs can be URL encoded more than once.
    for _ in range(3):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    return text


def _first(patterns: tuple[re.Pattern[str], ...], text: str) -> int | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def detect_process_game_context(pid: int) -> dict[str, Any]:
    """Best-effort Roblox Place/Universe detection scoped to one process.

    This only reads the process command line. It never inspects or injects into
    Roblox memory, which keeps instance ownership explicit and safe.
    """

    if pid <= 0:
        return {}

    try:
        import psutil
        process = psutil.Process(pid)
        parts = process.cmdline()
    except Exception:
        return {}

    text = _decode_command_line(parts)
    place_id = _first(_PLACE_PATTERNS, text)
    universe_id = _first(_UNIVERSE_PATTERNS, text)

    result: dict[str, Any] = {"source": "process_command_line"}
    if place_id is not None:
        result["place_id"] = place_id
    if universe_id is not None:
        result["universe_id"] = universe_id
    return result if len(result) > 1 else {}
