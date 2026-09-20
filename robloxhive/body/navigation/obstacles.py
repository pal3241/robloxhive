from __future__ import annotations

from dataclasses import dataclass

from robloxhive.body.navigation.grid import BLOCKED, FREE, OccupancyGrid


@dataclass(slots=True)
class ObstacleConfig:
    grid_width: int = 15
    grid_depth: int = 15
    roi_top_ratio: float = 0.40
    edge_threshold: float = 0.19
    texture_threshold: float = 32.0
    free_edge_threshold: float = 0.08


class ScreenObstacleEstimator:
    """Build an egocentric local occupancy grid from the bot window.

    This is intentionally conservative. It uses edge/texture density in the
    lower game view as a cheap local obstacle cue; it is not semantic depth.
    """

    def __init__(self, config: ObstacleConfig | None = None) -> None:
        self.config = config or ObstacleConfig()
        self.last_metrics: list[list[dict[str, float]]] = []

    def estimate(self, frame) -> OccupancyGrid:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Obstacle estimation requires opencv-python and numpy") from exc

        cfg = self.config
        grid = OccupancyGrid(cfg.grid_width, cfg.grid_depth)
        h, w = frame.shape[:2]
        top = int(h * cfg.roi_top_ratio)
        roi = frame[top:h, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 70, 150)

        rh, rw = gray.shape[:2]
        metrics: list[list[dict[str, float]]] = []
        for gy in range(cfg.grid_depth):
            # Near cells use the bottom of the image; far cells use upper ROI.
            y0 = int(rh * (1.0 - (gy + 1) / cfg.grid_depth))
            y1 = int(rh * (1.0 - gy / cfg.grid_depth))
            y0, y1 = max(0, y0), max(y0 + 1, y1)
            row_metrics: list[dict[str, float]] = []
            for gx in range(cfg.grid_width):
                x0 = int(rw * gx / cfg.grid_width)
                x1 = int(rw * (gx + 1) / cfg.grid_width)
                patch_e = edges[y0:y1, x0:x1]
                patch_g = gray[y0:y1, x0:x1]
                if patch_e.size == 0:
                    row_metrics.append({"edge": 0.0, "std": 0.0})
                    continue

                edge_density = float((patch_e > 0).mean())
                texture = float(np.std(patch_g))
                row_metrics.append({"edge": edge_density, "std": texture})

                # Strong nearby structure is likely an obstacle. Farther rows
                # require stronger evidence because perspective compresses them.
                distance_factor = 1.0 + (gy / max(1, cfg.grid_depth - 1)) * 0.55
                blocked = (
                    edge_density > cfg.edge_threshold * distance_factor
                    and texture > cfg.texture_threshold
                )
                clearly_free = edge_density < cfg.free_edge_threshold

                if blocked:
                    grid.set(gx, gy, BLOCKED)
                elif clearly_free:
                    grid.set(gx, gy, FREE)
            metrics.append(row_metrics)

        # The bot's own cell is always traversable.
        sx, sy = grid.start
        grid.set(sx, sy, FREE)
        self.last_metrics = metrics
        return grid
