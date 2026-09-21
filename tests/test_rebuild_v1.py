import tempfile
import unittest
from pathlib import Path

from robloxhive.brain.ollama_actor import AgentDecision, OllamaActor
from robloxhive.core.memory import CognitiveMemory
from robloxhive.core.semantic_map import SemanticMap
from robloxhive.shared.models import ActionResult, ActionStatus
from robloxhive.single.runtime import SingleBotRuntime


class FakeExecutor:
    perception = None
    frame_source = None

    def __init__(self):
        self.calls = []

    def available(self):
        return ["observe", "explore"]

    def execute(self, skill, payload):
        self.calls.append((skill, payload))
        return ActionResult(action=skill, status=ActionStatus.SUCCESS, details={"verified": True})

    def direct_control(self, payload):
        return {"ok": True}

    def navigation_probe(self):
        return {"available": True}

    def game_status(self):
        return {"adapter": None}


class FakeActor:
    class Config:
        vision = False
        base_url = "fake"
        model = "fake"

    config = Config()
    last_latency_ms = 1
    last_error = None

    def decide(self, **_kwargs):
        return AgentDecision(
            action="explore",
            payload={"seconds": 0.1},
            confidence=0.9,
            reason="Need more map information",
            remember=[
                {
                    "kind": "map",
                    "key": "spawn",
                    "text": "Spawn area",
                    "confidence": 0.8,
                    "importance": 0.7,
                }
            ],
            interpretation={
                "roles": {"self": "scout"},
                "teams": {"blue": ["self"]},
                "enemies": ["red_team"],
                "allies": ["blue_team"],
            },
        )


class RebuildV1Tests(unittest.TestCase):
    def test_cognitive_memory_retrieval_prefers_relevant_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = CognitiveMemory(Path(tmp) / "memory.db")
            memory.remember("semantic", "The shop sells ammo", game_id=7, confidence=0.8)
            memory.remember("semantic", "Spawn has a blue door", game_id=7, confidence=0.8)
            rows = memory.retrieve("where is ammo shop", game_id=7, limit=2)
            self.assertEqual(rows[0]["text"], "The shop sells ammo")

    def test_semantic_map_routes_using_learned_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = CognitiveMemory(Path(tmp) / "memory.db")
            smap = SemanticMap(memory, 9)
            smap.observe_landmark("spawn")
            smap.observe_landmark("shop", value=0.8)
            smap.observe_landmark("exit", value=1.0)
            smap.transition("spawn", "shop", seconds=3, success=True)
            smap.transition("shop", "exit", seconds=2, success=True)
            route = smap.route("spawn", "exit")
            self.assertTrue(route["found"])
            self.assertEqual(route["path"], ["spawn", "shop", "exit"])

    def test_ollama_json_extractor_handles_fenced_json(self):
        fence = chr(96) * 3
        row = OllamaActor._extract_json(fence + "json\n{\"action\":\"observe\"}\n" + fence)
        self.assertEqual(row["action"], "observe")

    def test_single_runtime_executes_actor_action_and_persists_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = FakeExecutor()
            runtime = SingleBotRuntime(
                executor,
                game_id=12,
                memory_path=str(Path(tmp) / "memory.db"),
            )
            runtime.actor = FakeActor()
            runtime.set_goal("Explore and learn", goal_type="explore")
            runtime.set_enabled(True)
            runtime.tick()

            self.assertEqual(executor.calls[0][0], "explore")
            self.assertEqual(runtime.success_count, 1)
            role = runtime.memory.retrieve("scout", game_id=12, kinds=["role"], limit=3)
            self.assertTrue(role)
            enemy = runtime.memory.retrieve("red_team", game_id=12, kinds=["enemy"], limit=3)
            self.assertTrue(enemy)
            self.assertIn("spawn", runtime.semantic_map.landmarks)


if __name__ == "__main__":
    unittest.main()
