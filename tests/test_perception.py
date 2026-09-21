import unittest

from robloxhive.body.perception import Detection
from robloxhive.body.perception_fusion import CompositePerception
from robloxhive.body.tracking import DetectionTracker, PlayerTracker


class FakeAdapter:
    def __init__(self, hit=None, name="fake"):
        self.hit = hit
        self.calls = 0
        self.name = name

    def find(self, label):
        self.calls += 1
        return self.hit

    def frame_size(self):
        return (1280, 720)


class PerceptionTests(unittest.TestCase):
    def test_composite_early_exits_on_strong_onnx_hit(self):
        onnx = FakeAdapter(
            Detection("shop", 0.91, 100, 100, 80, 80, source="onnx")
        )
        fallback = FakeAdapter(
            Detection("shop", 0.99, 200, 200, 80, 80, source="template")
        )
        fusion = CompositePerception([onnx, fallback], early_exit_confidence=0.72)
        hit = fusion.find("shop")
        self.assertEqual(hit.source, "onnx")
        self.assertEqual(onnx.calls, 1)
        self.assertEqual(fallback.calls, 0)

    def test_detection_tracker_keeps_id(self):
        frames = [
            [Detection("player", 0.9, 100, 100, 80, 120, source="onnx")],
            [Detection("player", 0.9, 105, 102, 80, 120, source="onnx")],
        ]

        def detector():
            return frames.pop(0)

        tracker = DetectionTracker(detector, iou_threshold=0.2)
        first = tracker.update()[0]
        second = tracker.update()[0]
        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(second.metadata["track_age"], 2)

    def test_player_tracker_matches_requested_username_with_nameplate_ocr(self):
        def detector():
            return [
                Detection("player", 0.9, 20, 100, 80, 160, source="onnx"),
                Detection("player", 0.9, 220, 100, 80, 160, source="onnx"),
            ]

        class FakeOCR:
            def scan(self):
                return [
                    Detection("Alice_123", 0.95, 230, 70, 70, 24, source="ocr")
                ]

        player = PlayerTracker(DetectionTracker(detector), nameplate_detector=FakeOCR())
        hit = player.find("player:Alice_123")

        self.assertIsNotNone(hit)
        self.assertEqual(hit.track_id, 2)
        self.assertTrue(hit.metadata["username_verified"])
        self.assertEqual(hit.metadata["tracking_mode"], "username_nameplate_ocr")

    def test_player_tracker_never_falls_back_to_random_avatar_for_username(self):
        def detector():
            return [Detection("player", 0.9, 20, 100, 80, 160, source="onnx")]

        player = PlayerTracker(DetectionTracker(detector), nameplate_detector=None)
        self.assertIsNone(player.find("player:MissingUser"))

    def test_player_tracker_only_handles_player_route(self):
        def detector():
            return [Detection("player", 0.8, 10, 20, 50, 100, source="onnx")]

        player = PlayerTracker(DetectionTracker(detector))
        self.assertIsNone(player.find("shop"))
        hit = player.find("player:Fahri")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.source, "player_tracker")
        self.assertEqual(hit.metadata["requested_player"], "Fahri")


if __name__ == "__main__":
    unittest.main()
