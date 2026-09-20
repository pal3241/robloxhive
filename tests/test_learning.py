import tempfile
import time
import unittest
from pathlib import Path

from robloxhive.brain.learning import LearningManager
from robloxhive.brain.memory import GameMemory


class FakeResearcher:
    def research(self, game_name, objective=None, progress=None):
        if progress:
            progress("searching", 1, 2)
            progress("extracting", 2, 2)
        return {
            "game_name": game_name,
            "objective": objective,
            "queries": ["guide", "tips"],
            "source_count": 1,
            "extracted_count": 1,
            "sources": [{
                "title": "Guide",
                "url": "https://example.com",
                "snippet": "Start here",
                "content": "Full guide",
                "quality": 0.8,
                "extracted": True,
                "query": "guide",
            }],
            "candidate_knowledge": [{
                "text": "Start here",
                "source_url": "https://example.com",
                "confidence": 0.6,
                "verified_in_game": False,
            }],
        }


class LearningTests(unittest.TestCase):
    def test_learning_job_writes_per_game_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = GameMemory(Path(tmp))
            manager = LearningManager(memory, researcher=FakeResearcher(), workers=1)
            job = manager.start(123, "Example Game", "learn start to finish")

            deadline = time.time() + 2
            while time.time() < deadline:
                current = manager.get(job.id)
                if current and current.status in {"complete", "failed"}:
                    break
                time.sleep(0.02)

            current = manager.get(job.id)
            self.assertIsNotNone(current)
            self.assertEqual(current.status, "complete")
            profile = memory.load_profile(123)
            self.assertEqual(profile["game_name"], "Example Game")
            self.assertEqual(profile["research"]["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
