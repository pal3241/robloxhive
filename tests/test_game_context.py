import unittest
from unittest.mock import patch

from robloxhive.body.game_context import detect_process_game_context


class FakeProcess:
    def __init__(self, _pid):
        pass

    def cmdline(self):
        return [
            "RobloxPlayerBeta.exe",
            "roblox-player:1+placelauncherurl:https%3A%2F%2Fwww.roblox.com%2FGame%2FPlaceLauncher.ashx%3Frequest%3DRequestGame%26placeId%3D142823291",
        ]


class GameContextTests(unittest.TestCase):
    def test_detects_place_id_from_encoded_process_command_line(self):
        class FakePsutil:
            Process = FakeProcess

        with patch.dict("sys.modules", {"psutil": FakePsutil()}):
            result = detect_process_game_context(1234)

        self.assertEqual(result["place_id"], 142823291)
        self.assertEqual(result["source"], "process_command_line")


if __name__ == "__main__":
    unittest.main()
