import unittest

from robloxhive.brain.synthesis import HeuristicKnowledgeSynthesizer, normalize_knowledge


class SynthesisTests(unittest.TestCase):
    def test_normalize_clamps_confidence(self):
        data = {
            "overview": "Example",
            "strategies": [
                {"text": "Do X", "confidence": 4, "sources": ["S1"]},
                {"text": "Do Y", "confidence": -1, "sources": ["S2"]},
            ],
        }
        result = normalize_knowledge(data, "Example", "test")
        self.assertEqual(result["strategies"][0]["confidence"], 1.0)
        self.assertEqual(result["strategies"][1]["confidence"], 0.0)

    def test_heuristic_preserves_source_traceability(self):
        bundle = {
            "sources": [
                {
                    "query": "Example Roblox tips tricks strategy",
                    "snippet": "Save resources before upgrading.",
                    "content": "",
                    "quality": 0.8,
                }
            ],
            "candidate_knowledge": [],
        }
        result = HeuristicKnowledgeSynthesizer().synthesize("Example", bundle)
        self.assertEqual(result["strategies"][0]["sources"], ["S1"])
        self.assertFalse(result["strategies"][0]["verified_in_game"])


if __name__ == "__main__":
    unittest.main()
