import tempfile
import unittest
from pathlib import Path

from robloxhive.brain.memory import GameMemory
from robloxhive.brain.runtime import AgentRuntime
from robloxhive.shared.models import ActionResult, ActionStatus, Goal, PlanStatus, StepStatus


class RuntimeTests(unittest.TestCase):
    def _memory(self, root):
        memory = GameMemory(Path(root))
        memory.save_knowledge(
            77,
            {
                "game_name": "Example",
                "synthesizer": "test",
                "confidence": 0.8,
                "objectives": [],
                "progression": [
                    {
                        "text": "Collect fuel before leaving the starting area.",
                        "confidence": 0.8,
                        "sources": ["S1"],
                        "verified_in_game": False,
                    },
                    {
                        "text": "Travel to the next station.",
                        "confidence": 0.75,
                        "sources": ["S2"],
                        "verified_in_game": False,
                    },
                ],
                "strategies": [],
                "items": [],
                "locations": [],
                "enemies": [],
                "endgame": [],
                "mechanics": [],
                "common_mistakes": [],
                "unknowns": [],
                "conflicts": [],
            },
        )
        return memory

    def test_goal_uses_game_knowledge_and_dispatches_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(self._memory(tmp))
            plan = runtime.start_goal(
                77,
                Goal(type="complete_game", metadata={"instruction": "finish the game"}),
            )
            self.assertGreaterEqual(len(plan.steps), 2)
            command = runtime.next_command(timeout=0.1)
            self.assertIsNotNone(command)
            self.assertEqual(command.type, "EXECUTE_SKILL")
            self.assertEqual(command.payload["plan_id"], plan.id)

    def test_goal_routes_every_step_to_selected_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(self._memory(tmp))
            plan = runtime.start_goal(
                77,
                Goal(
                    type="complete_game",
                    metadata={"instruction": "finish the game", "agent_id": "agent-02"},
                ),
            )
            command = runtime.next_command(timeout=0.1, agent_id="agent-02")
            self.assertIsNotNone(command)
            self.assertEqual(command.payload["agent_id"], "agent-02")
            self.assertEqual(command.payload["plan_id"], plan.id)

    def test_success_advances_and_verifies_knowledge(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = self._memory(tmp)
            runtime = AgentRuntime(memory)
            plan = runtime.start_goal(77, Goal(type="complete_game"))
            first = plan.current_step
            runtime.next_command(timeout=0.1)

            updated = runtime.record_result(
                plan.id,
                first.index,
                ActionResult(
                    action=first.skill,
                    status=ActionStatus.SUCCESS,
                    details={"observed": "objective advanced"},
                ),
                evidence={"verified": True, "observed": "objective advanced"},
            )
            self.assertEqual(updated.steps[0].status, StepStatus.COMPLETE)
            self.assertIn(updated.status, {PlanStatus.RUNNING, PlanStatus.COMPLETE})
            knowledge = memory.load_knowledge(77)
            match = next(x for x in knowledge["progression"] if x["text"] == first.instruction)
            self.assertTrue(match["verified_in_game"])
            self.assertGreaterEqual(match["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
