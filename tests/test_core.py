import tempfile
import unittest
from pathlib import Path

from robloxhive.body.instances import InstanceManager, RobloxInstance, ProtectedInstanceError
from robloxhive.brain.memory import GameMemory
from robloxhive.shared.models import InstanceRole


class CoreTests(unittest.TestCase):
    def test_player_window_is_protected(self):
        manager = InstanceManager()
        manager.register(RobloxInstance(pid=1, hwnd=10, title="Roblox"))
        manager.mark_player(1)
        with self.assertRaises(ProtectedInstanceError):
            manager.assign_bot(1, "agent-01")

    def test_game_memory_is_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = GameMemory(Path(tmp))
            memory.remember(100, "facts", {"name": "shop"})
            memory.remember(200, "facts", {"name": "enemy"})
            self.assertEqual(memory.load_profile(100)["facts"][0]["name"], "shop")
            self.assertEqual(memory.load_profile(200)["facts"][0]["name"], "enemy")


if __name__ == "__main__":
    unittest.main()
