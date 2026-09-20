from __future__ import annotations

import heapq
import math
from itertools import count

from robloxhive.body.navigation.grid import OccupancyGrid


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(
    grid: OccupancyGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    if not grid.in_bounds(start) or not grid.in_bounds(goal):
        return []
    if grid.get(*goal) == 1:
        return []

    queue: list[tuple[float, int, tuple[int, int]]] = []
    seq = count()
    heapq.heappush(queue, (0.0, next(seq), start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

    while queue:
        _, _, current = heapq.heappop(queue)
        if current == goal:
            break

        for nxt, step_cost in grid.neighbors(current):
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + _heuristic(nxt, goal)
                heapq.heappush(queue, (priority, next(seq), nxt))
                came_from[nxt] = current

    if goal not in cost_so_far:
        return []

    path = [goal]
    node = goal
    while node != start:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path
