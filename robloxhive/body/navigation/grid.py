from __future__ import annotations

from dataclasses import dataclass, field

UNKNOWN = -1
FREE = 0
BLOCKED = 1


@dataclass(slots=True)
class OccupancyGrid:
    width: int = 15
    depth: int = 15
    cells: list[list[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.cells = [[UNKNOWN for _ in range(self.width)] for _ in range(self.depth)]

    @property
    def start(self) -> tuple[int, int]:
        return (self.width // 2, 0)

    def in_bounds(self, node: tuple[int, int]) -> bool:
        x, y = node
        return 0 <= x < self.width and 0 <= y < self.depth

    def set(self, x: int, y: int, value: int) -> None:
        if self.in_bounds((x, y)):
            self.cells[y][x] = value

    def get(self, x: int, y: int) -> int:
        if not self.in_bounds((x, y)):
            return BLOCKED
        return self.cells[y][x]

    def neighbors(self, node: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        x, y = node
        out: list[tuple[tuple[int, int], float]] = []
        for dx, dy, cost in (
            (0, 1, 1.0),
            (-1, 0, 1.05),
            (1, 0, 1.05),
            (-1, 1, 1.42),
            (1, 1, 1.42),
            (0, -1, 1.2),
        ):
            nxt = (x + dx, y + dy)
            if not self.in_bounds(nxt) or self.get(*nxt) == BLOCKED:
                continue
            # Unknown space is allowed but slightly more expensive than observed-free.
            penalty = 0.35 if self.get(*nxt) == UNKNOWN else 0.0
            out.append((nxt, cost + penalty))
        return out

    def as_ascii(
        self,
        path: list[tuple[int, int]] | None = None,
        goal: tuple[int, int] | None = None,
    ) -> list[str]:
        marks = set(path or [])
        rows: list[str] = []
        for y in reversed(range(self.depth)):
            chars = []
            for x in range(self.width):
                node = (x, y)
                if node == self.start:
                    chars.append("B")
                elif goal is not None and node == goal:
                    chars.append("G")
                elif node in marks:
                    chars.append("*")
                else:
                    chars.append({UNKNOWN: "?", FREE: ".", BLOCKED: "#"}[self.get(x, y)])
            rows.append("".join(chars))
        return rows
