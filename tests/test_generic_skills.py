import unittest

from robloxhive.body.generic_skills import GenericVisualSkills, SkillConfig
from robloxhive.body.perception import Detection
from robloxhive.body.skills import SkillExecutor
from robloxhive.shared.models import ActionStatus


class FakePerception:
    def __init__(self, detections):
        self.detections = list(detections)
        self.last = None

    def find(self, label):
        if self.detections:
            self.last = self.detections.pop(0)
        return self.last

    def frame_size(self):
        return (1000, 600)


class FakeInput:
    def __init__(self):
        self.actions = []

    def move(self, direction, seconds):
        self.actions.append(("move", direction, seconds))

    def interact(self, key="interact"):
        self.actions.append(("interact", key))

    def click_client(self, x, y, button="left"):
        self.actions.append(("click", x, y, button))

    def release_all(self):
        self.actions.append(("release",))


def det(x=450, y=200, w=100, h=100, confidence=0.9):
    return Detection("target", confidence, x, y, w, h)


class GenericSkillTests(unittest.TestCase):
    def config(self):
        return SkillConfig(
            steering_seconds=0.001,
            forward_seconds=0.001,
            settle_seconds=0.0,
            max_iterations=6,
            lost_target_limit=1,
            near_area_ratio=0.01,
            follow_far_area_ratio=0.008,
            follow_near_area_ratio=0.03,
        )

    def test_navigate_steers_then_approaches(self):
        perception = FakePerception([
            det(x=800, w=50, h=50),
            det(x=475, w=50, h=50),
            det(x=420, y=120, w=180, h=180),
        ])
        inputs = FakeInput()
        skills = GenericVisualSkills(perception, inputs, self.config())
        result = skills.navigate({"target": "shop"})
        self.assertEqual(result.status, ActionStatus.SUCCESS)
        self.assertTrue(any(a[0] == "move" and a[1] == "right" for a in inputs.actions))
        self.assertTrue(any(a[0] == "move" and a[1] == "forward" for a in inputs.actions))

    def test_collect_requires_visual_change(self):
        perception = FakePerception([
            det(x=420, y=120, w=180, h=180),
            det(x=420, y=120, w=180, h=180),
            None,
        ])
        inputs = FakeInput()
        skills = GenericVisualSkills(perception, inputs, self.config())
        result = skills.collect({"target": "fuel"})
        self.assertEqual(result.status, ActionStatus.SUCCESS)
        self.assertTrue(result.details["evidence"]["verified"])

    def test_interact_requires_visual_verification(self):
        perception = FakePerception([
            det(x=420, y=120, w=180, h=180),
            det(x=420, y=120, w=180, h=180),
            det(x=420, y=120, w=180, h=180),
        ])
        inputs = FakeInput()
        skills = GenericVisualSkills(perception, inputs, self.config())
        result = skills.interact({"target": "door"})
        self.assertEqual(result.status, ActionStatus.FAILED)
        self.assertEqual(result.error, "INTERACTION_NOT_VERIFIED")

    def test_follow_player_maintains_distance(self):
        perception = FakePerception([
            det(x=450, y=180, w=120, h=100),
            det(x=450, y=180, w=120, h=100),
            det(x=450, y=180, w=120, h=100),
        ])
        inputs = FakeInput()
        skills = GenericVisualSkills(perception, inputs, self.config())
        result = skills.follow_player({"target": "Fahri", "iterations": 5})
        self.assertEqual(result.status, ActionStatus.SUCCESS)
        self.assertTrue(result.details["evidence"]["follow_distance_stable"])

    def test_registers_four_requested_skills(self):
        skills = GenericVisualSkills(FakePerception([]), FakeInput(), self.config())
        executor = SkillExecutor()
        skills.register_into(executor)
        self.assertEqual(
            executor.available(),
            ["collect", "follow_player", "interact", "navigate"],
        )


if __name__ == "__main__":
    unittest.main()
