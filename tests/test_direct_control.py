import unittest

from robloxhive.body.skills import SkillExecutor


class FakeInput:
    def __init__(self):
        self.calls = []

    def move(self, direction, seconds):
        self.calls.append(("move", direction, seconds))

    def key(self, name, seconds=0.08):
        self.calls.append(("key", name, seconds))

    def interact(self, key="interact"):
        self.calls.append(("interact", key))

    def release_all(self):
        self.calls.append(("release",))


class DirectControlTests(unittest.TestCase):
    def test_direct_move_uses_attached_input(self):
        executor = SkillExecutor()
        fake = FakeInput()
        executor.attach_input(fake)

        result = executor.direct_control({"action": "forward", "seconds": 0.2})

        self.assertTrue(result["ok"])
        self.assertEqual(fake.calls[0][0:2], ("move", "forward"))

    def test_release_is_available_without_game_adapter(self):
        executor = SkillExecutor()
        fake = FakeInput()
        executor.attach_input(fake)

        result = executor.direct_control({"action": "release"})

        self.assertTrue(result["ok"])
        self.assertEqual(fake.calls, [("release",)])


if __name__ == "__main__":
    unittest.main()
