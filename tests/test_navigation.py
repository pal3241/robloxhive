import unittest

from robloxhive.body.navigation.astar import astar
from robloxhive.body.navigation.grid import BLOCKED, FREE, OccupancyGrid
from robloxhive.body.navigation.navigator import LocalNavigator, NavigationConfig
from robloxhive.body.perception import Detection


class FakePerception:
    def __init__(self, detections):
        self.detections = list(detections)
        self.last = self.detections[-1] if self.detections else None

    def find(self, label):
        if self.detections:
            self.last = self.detections.pop(0)
        return self.last

    def frame_size(self):
        return (1000, 600)


class FakeCapture:
    def __init__(self):
        self.i = 0

    def capture(self):
        self.i += 1
        return {"frame": self.i}


class FakeInput:
    def __init__(self):
        self.actions = []

    def move(self, direction, seconds):
        self.actions.append((direction, seconds))

    def release_all(self):
        self.actions.append(("release", 0))


class FakeObstacles:
    def __init__(self, block_forward=True):
        self.block_forward = block_forward

    def estimate(self, frame):
        grid = OccupancyGrid(9, 9)
        for y in range(grid.depth):
            for x in range(grid.width):
                grid.set(x, y, FREE)
        if self.block_forward:
            sx, sy = grid.start
            grid.set(sx, sy + 1, BLOCKED)
        return grid


class FakeMotion:
    def __init__(self, values):
        self.values = list(values)
        self.last_score = 0.0

    def moved(self, before, after):
        value = self.values.pop(0) if self.values else True
        self.last_score = 0.05 if value else 0.0
        return value


def det(w=50, h=50, x=475, y=220):
    return Detection(
        "target",
        0.9,
        x,
        y,
        w,
        h,
        source="test",
    )


class NavigationTests(unittest.TestCase):
    def test_astar_routes_around_blocked_cell(self):
        grid = OccupancyGrid(7, 7)
        for y in range(grid.depth):
            for x in range(grid.width):
                grid.set(x, y, FREE)
        sx, sy = grid.start
        grid.set(sx, sy + 1, BLOCKED)

        goal = (sx, 4)
        path = astar(grid, grid.start, goal)

        self.assertTrue(path)
        self.assertEqual(path[0], grid.start)
        self.assertEqual(path[-1], goal)
        self.assertNotIn((sx, sy + 1), path)

    def test_local_navigator_uses_detour_then_arrives(self):
        perception = FakePerception([
            det(w=50, h=50),
            det(w=230, h=230, x=385, y=150),
        ])
        inputs = FakeInput()
        nav = LocalNavigator(
            FakeCapture(),
            perception,
            inputs,
            obstacle_estimator=FakeObstacles(block_forward=True),
            motion_verifier=FakeMotion([True]),
            config=NavigationConfig(
                near_area_ratio=0.075,
                step_seconds=0.001,
                side_seconds=0.001,
                back_seconds=0.001,
                settle_seconds=0.0,
                max_iterations=4,
                max_goal_depth=6,
            ),
        )

        ok, details = nav.navigate_to("shop")

        self.assertTrue(ok)
        self.assertEqual(details["reason"], "ARRIVED")
        self.assertTrue(any(action[0] in {"left", "right"} for action in inputs.actions))
        self.assertGreaterEqual(nav.stats["replans"], 1)
        self.assertTrue(nav.last_path)

    def test_stuck_triggers_recovery(self):
        perception = FakePerception([
            det(w=50, h=50),
            det(w=230, h=230, x=385, y=150),
        ])
        inputs = FakeInput()
        nav = LocalNavigator(
            FakeCapture(),
            perception,
            inputs,
            obstacle_estimator=FakeObstacles(block_forward=False),
            motion_verifier=FakeMotion([False]),
            config=NavigationConfig(
                near_area_ratio=0.075,
                step_seconds=0.001,
                side_seconds=0.001,
                back_seconds=0.001,
                settle_seconds=0.0,
                max_iterations=4,
                stuck_limit=1,
                recovery_limit=2,
                max_goal_depth=6,
            ),
        )

        ok, _ = nav.navigate_to("shop")

        self.assertTrue(ok)
        self.assertEqual(nav.stats["stuck_events"], 1)
        self.assertGreaterEqual(nav.stats["recoveries"], 1)
        self.assertTrue(any(action[0] == "back" for action in inputs.actions))

    def test_ascii_map_contains_bot_goal_and_path(self):
        grid = OccupancyGrid(5, 5)
        for y in range(grid.depth):
            for x in range(grid.width):
                grid.set(x, y, FREE)
        goal = (2, 3)
        path = astar(grid, grid.start, goal)
        rendered = "\n".join(grid.as_ascii(path, goal))
        self.assertIn("B", rendered)
        self.assertIn("G", rendered)
        self.assertIn("*", rendered)


if __name__ == "__main__":
    unittest.main()
