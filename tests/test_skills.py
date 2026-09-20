import unittest

from robloxhive.body.skills import SkillExecutor
from robloxhive.shared.models import ActionResult, ActionStatus


class SkillExecutorTests(unittest.TestCase):
    def test_unknown_skill_fails_safe(self):
        executor = SkillExecutor()
        result = executor.execute("navigate", {"target": "shop"})
        self.assertEqual(result.status, ActionStatus.BLOCKED)
        self.assertEqual(result.error, "UNSUPPORTED_SKILL")
        self.assertFalse(result.recoverable)

    def test_registered_skill_executes(self):
        executor = SkillExecutor()
        executor.register(
            "collect",
            lambda payload: ActionResult(
                action="collect",
                status=ActionStatus.SUCCESS,
                details={"verified": payload["item"] == "fuel"},
            ),
        )
        result = executor.execute("collect", {"item": "fuel"})
        self.assertTrue(result.success)
        self.assertTrue(result.details["verified"])


if __name__ == "__main__":
    unittest.main()
