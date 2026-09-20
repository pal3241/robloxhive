from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from robloxhive.body.navigation.astar import astar
from robloxhive.body.navigation.grid import BLOCKED, FREE, OccupancyGrid
from robloxhive.body.navigation.motion import MotionVerifier
from robloxhive.body.navigation.obstacles import ScreenObstacleEstimator
from robloxhive.body.perception import Detection, FrameSource, PerceptionAdapter


class NavigationInput(Protocol):
    def move(self, direction: str, seconds: float) -> None: ...
    def release_all(self) -> None: ...


@dataclass(slots=True)
class NavigationConfig:
    near_area_ratio: float = 0.075
    center_tolerance: float = 0.13
    step_seconds: float = 0.18
    side_seconds: float = 0.14
    back_seconds: float = 0.16
    settle_seconds: float = 0.07
    max_iterations: int = 55
    lost_target_limit: int = 6
    stuck_limit: int = 2
    recovery_limit: int = 5
    max_goal_depth: int = 12


class LocalNavigator:
    """Egocentric local planner with A*, motion verification and replanning."""

    def __init__(
        self,
        frame_source: FrameSource,
        perception: PerceptionAdapter,
        input_backend: NavigationInput,
        obstacle_estimator: ScreenObstacleEstimator | None = None,
        motion_verifier: MotionVerifier | None = None,
        config: NavigationConfig | None = None,
    ) -> None:
        self.frame_source = frame_source
        self.perception = perception
        self.input = input_backend
        self.obstacles = obstacle_estimator or ScreenObstacleEstimator()
        self.motion = motion_verifier or MotionVerifier()
        self.config = config or NavigationConfig()
        self.last_grid: OccupancyGrid | None = None
        self.last_path: list[tuple[int, int]] = []
        self.last_goal: tuple[int, int] | None = None
        self.stats: dict[str, Any] = {
            "replans": 0,
            "recoveries": 0,
            "stuck_events": 0,
            "last_motion_score": 0.0,
            "last_result": None,
        }

    def _target_cell(self, grid: OccupancyGrid, detection: Detection) -> tuple[int, int]:
        width, height = self.perception.frame_size()
        if width <= 0 or height <= 0:
            return (grid.width // 2, min(self.config.max_goal_depth, grid.depth - 1))

        x_norm = max(0.0, min(0.999, detection.center_x / width))
        gx = min(grid.width - 1, int(x_norm * grid.width))

        area_ratio = detection.area / float(width * height)
        # Larger apparent size means closer target.
        closeness = min(1.0, area_ratio / max(self.config.near_area_ratio, 1e-6))
        depth = int(round((1.0 - closeness) * self.config.max_goal_depth))
        gy = max(1, min(grid.depth - 1, depth))
        return (gx, gy)

    def _choose_action(
        self,
        grid: OccupancyGrid,
        path: list[tuple[int, int]],
        detection: Detection,
    ) -> tuple[str, float]:
        if len(path) >= 2:
            sx, sy = path[0]
            nx, ny = path[1]
            dx, dy = nx - sx, ny - sy
            if dx < 0:
                return ("left", self.config.side_seconds)
            if dx > 0:
                return ("right", self.config.side_seconds)
            if dy < 0:
                return ("back", self.config.back_seconds)
            return ("forward", self.config.step_seconds)

        width, _ = self.perception.frame_size()
        if width > 0:
            offset = detection.center_x / width - 0.5
            if offset < -self.config.center_tolerance:
                return ("left", self.config.side_seconds)
            if offset > self.config.center_tolerance:
                return ("right", self.config.side_seconds)
        return ("forward", self.config.step_seconds)

    def _recover(self, attempt: int) -> None:
        self.stats["recoveries"] += 1
        sequence = (
            ("back", self.config.back_seconds),
            ("left", self.config.side_seconds * 1.5),
            ("right", self.config.side_seconds * 1.5),
            ("back", self.config.back_seconds * 1.5),
            ("right", self.config.side_seconds * 2.0),
        )
        direction, seconds = sequence[attempt % len(sequence)]
        self.input.move(direction, seconds)
        time.sleep(self.config.settle_seconds)

    def navigate_to(
        self,
        label: str,
        near_area_ratio: float | None = None,
        max_iterations: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        cfg = self.config
        near = near_area_ratio if near_area_ratio is not None else cfg.near_area_ratio
        iterations = max_iterations or cfg.max_iterations
        lost = 0
        stuck = 0
        recoveries = 0

        for index in range(iterations):
            detection = self.perception.find(label)
            if detection is None:
                lost += 1
                if lost > cfg.lost_target_limit:
                    self.input.release_all()
                    self.stats["last_result"] = "TARGET_LOST"
                    return False, self.diagnostics("TARGET_LOST", label, index + 1)
                self.input.move("right", cfg.side_seconds)
                time.sleep(cfg.settle_seconds)
                continue

            lost = 0
            width, height = self.perception.frame_size()
            area_ratio = (
                detection.area / float(width * height)
                if width > 0 and height > 0 else 0.0
            )
            offset = detection.center_x / width - 0.5 if width > 0 else 0.0
            if area_ratio >= near and abs(offset) <= cfg.center_tolerance:
                self.input.release_all()
                self.stats["last_result"] = "ARRIVED"
                return True, {
                    **self.diagnostics("ARRIVED", label, index + 1),
                    "confidence": detection.confidence,
                    "source": detection.source,
                    "area_ratio": round(area_ratio, 5),
                    "offset": round(offset, 4),
                }

            before = self.frame_source.capture()
            grid = self.obstacles.estimate(before)
            goal = self._target_cell(grid, detection)
            grid.set(*goal, FREE)

            # If the direct near-forward cell is strongly blocked, A* must route around it.
            path = astar(grid, grid.start, goal)
            self.stats["replans"] += 1
            self.last_grid, self.last_path, self.last_goal = grid, path, goal

            if not path:
                # Mark the center-front region as blocked and recover/re-observe.
                sx, sy = grid.start
                grid.set(sx, min(grid.depth - 1, sy + 1), BLOCKED)
                if recoveries >= cfg.recovery_limit:
                    self.input.release_all()
                    self.stats["last_result"] = "PATH_UNREACHABLE"
                    return False, self.diagnostics("PATH_UNREACHABLE", label, index + 1)
                self._recover(recoveries)
                recoveries += 1
                continue

            direction, seconds = self._choose_action(grid, path, detection)
            self.input.move(direction, seconds)
            time.sleep(cfg.settle_seconds)
            after = self.frame_source.capture()
            moved = self.motion.moved(before, after)
            self.stats["last_motion_score"] = round(self.motion.last_score, 5)

            if not moved:
                stuck += 1
                if stuck >= cfg.stuck_limit:
                    self.stats["stuck_events"] += 1
                    if recoveries >= cfg.recovery_limit:
                        self.input.release_all()
                        self.stats["last_result"] = "STUCK"
                        return False, self.diagnostics("STUCK", label, index + 1)
                    self._recover(recoveries)
                    recoveries += 1
                    stuck = 0
            else:
                stuck = 0

        self.input.release_all()
        self.stats["last_result"] = "NAVIGATION_TIMEOUT"
        return False, self.diagnostics("NAVIGATION_TIMEOUT", label, iterations)

    def diagnostics(self, reason: str | None = None, target: str | None = None, iterations: int | None = None) -> dict[str, Any]:
        return {
            "reason": reason,
            "target": target,
            "iterations": iterations,
            "stats": dict(self.stats),
            "goal_cell": list(self.last_goal) if self.last_goal else None,
            "path": [list(node) for node in self.last_path],
            "map": self.last_grid.as_ascii(self.last_path, self.last_goal) if self.last_grid else [],
        }
