import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robloxhive.body.perception import Detection
from robloxhive.body.tracking import DetectionTracker
from robloxhive.games.murder_mystery_2.combat import MM2Combat, MM2CombatConfig
from robloxhive.games.murder_mystery_2.dataset import MM2DatasetRecorder
from robloxhive.games.murder_mystery_2.event_reasoning import WitnessedKillReasoner
from robloxhive.games.murder_mystery_2.models import MM2SceneSnapshot, PlayerObservation
from robloxhive.games.murder_mystery_2.threat import MM2ThreatModel


class FakeInput:
    def __init__(self):
        self.actions = []

    def key(self, name, seconds=0.08):
        self.actions.append(("key", name, seconds))

    def aim_client(self, x, y):
        self.actions.append(("aim", x, y))

    def click_client(self, x, y, button="left"):
        self.actions.append(("click", x, y, button))


def player(track_id, x, y, w=100, h=180, *, knife=False):
    return PlayerObservation(
        track_id=track_id,
        detection=Detection(
            "player",
            0.95,
            x,
            y,
            w,
            h,
            source="test",
            track_id=track_id,
        ),
        has_knife=knife,
        murderer_confidence=0.97 if knife else 0.0,
    )


class MM2V09Tests(unittest.TestCase):
    def test_tracking_estimates_velocity(self):
        frames = [
            [Detection("player", 0.9, 100, 100, 80, 120, source="onnx")],
            [Detection("player", 0.9, 120, 100, 80, 120, source="onnx")],
        ]

        def detector():
            return frames.pop(0)

        tracker = DetectionTracker(detector, iou_threshold=0.2)
        with patch("robloxhive.body.tracking.time.monotonic", side_effect=[1.0, 1.1]):
            tracker.update()
            second = tracker.update()[0]

        vx, vy = second.metadata["velocity_px_s"]
        self.assertGreater(vx, 0)
        self.assertAlmostEqual(vy, 0.0, places=2)

    def test_sheriff_leads_moving_target(self):
        inputs = FakeInput()
        combat = MM2Combat(
            inputs,
            MM2CombatConfig(
                fire_cooldown_s=0.0,
                sheriff_lead_s=0.10,
                max_lead_px=100.0,
            ),
        )
        suspect = player(2, 400, 150)
        suspect.murderer_confidence = 0.99
        suspect.detection.metadata["velocity_px_s"] = [200.0, 0.0]
        scene = MM2SceneSnapshot((1000, 600), players=[suspect])

        fired, _ = combat.sheriff_fire(suspect, scene)
        self.assertTrue(fired)
        aim = next(a for a in inputs.actions if a[0] == "aim")
        base_x = suspect.detection.center_x
        self.assertGreater(aim[1], base_x)

    def test_witnessed_body_reinforces_visible_knife_holder(self):
        threats = MM2ThreatModel()
        reasoner = WitnessedKillReasoner()
        suspect = player(7, 400, 160, knife=True)
        baseline = MM2SceneSnapshot((1000, 600), players=[suspect], bodies=[])
        threats.update(baseline)
        self.assertEqual(reasoner.update(baseline, threats), [])

        body = Detection("dead_player", 0.95, 430, 280, 100, 60, source="onnx")
        scene = MM2SceneSnapshot((1000, 600), players=[suspect], bodies=[body])
        threats.update(scene)
        events = reasoner.update(scene, threats)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].suspect_track_id, 7)
        self.assertGreaterEqual(threats.murderer_score[7], 0.985)

    def test_first_visible_body_is_only_baseline(self):
        threats = MM2ThreatModel()
        reasoner = WitnessedKillReasoner()
        suspect = player(7, 400, 160, knife=True)
        body = Detection("dead_player", 0.95, 430, 280, 100, 60, source="onnx")
        scene = MM2SceneSnapshot((1000, 600), players=[suspect], bodies=[body])
        threats.update(scene)
        events = reasoner.update(scene, threats)
        self.assertEqual(events, [])

    def test_manual_review_clamps_boxes_to_image_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = MM2DatasetRecorder(None, root=tmp)
            sample_id = "sample-review"
            image_path = recorder.raw_images / f"{sample_id}.png"
            image_path.write_bytes(b"fake-image")
            ann_path = recorder.raw_annotations / f"{sample_id}.json"
            ann_path.write_text(
                json.dumps(
                    {
                        "id": sample_id,
                        "image": str(image_path),
                        "width": 100,
                        "height": 50,
                        "boxes": [],
                    }
                ),
                encoding="utf-8",
            )
            reviewed = recorder.review(
                sample_id,
                [
                    {
                        "label": "player",
                        "x": 90,
                        "y": 45,
                        "width": 50,
                        "height": 50,
                        "approved": True,
                    }
                ],
            )
            box = reviewed["boxes"][0]
            self.assertEqual(box["x"], 90)
            self.assertEqual(box["y"], 45)
            self.assertEqual(box["width"], 10)
            self.assertEqual(box["height"], 5)

    def test_dataset_export_only_uses_approved_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = MM2DatasetRecorder(None, root=tmp)
            sample_id = "sample1"
            image_path = recorder.raw_images / f"{sample_id}.png"
            image_path.write_bytes(b"fake-image-bytes")
            annotation = {
                "id": sample_id,
                "image": str(image_path),
                "width": 1000,
                "height": 500,
                "boxes": [
                    {
                        "label": "player",
                        "x": 100,
                        "y": 50,
                        "width": 200,
                        "height": 300,
                        "approved": True,
                    },
                    {
                        "label": "knife",
                        "x": 600,
                        "y": 100,
                        "width": 80,
                        "height": 120,
                        "approved": False,
                    },
                ],
            }
            ann_path = recorder.raw_annotations / f"{sample_id}.json"
            ann_path.write_text(json.dumps(annotation), encoding="utf-8")

            result = recorder.export_yolo(validation_ratio=0.0)
            label_path = Path(result["root"]) / "labels" / "train" / f"{sample_id}.txt"
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line]

            self.assertEqual(result["train"], 1)
            self.assertEqual(result["boxes"], 1)
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].startswith("0 "))


if __name__ == "__main__":
    unittest.main()
