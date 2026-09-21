import unittest

from robloxhive.body.perception import Detection
from robloxhive.games.murder_mystery_2.autonomy import MM2Autonomy
from robloxhive.games.murder_mystery_2.combat import MM2Combat, MM2CombatConfig
from robloxhive.games.murder_mystery_2.models import (
    MM2Role,
    MM2SceneSnapshot,
    PlayerObservation,
    RoundPhase,
)
from robloxhive.games.murder_mystery_2.role_detector import MM2RoleDetector
from robloxhive.games.murder_mystery_2.scene import MM2SceneReader
from robloxhive.games.murder_mystery_2.survival import MM2Survival, SurvivalConfig
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

    def move(self, direction, seconds):
        self.actions.append(("move", direction, seconds))

    def release_all(self):
        self.actions.append(("release",))


class FakePerception:
    adapters = []

    def __init__(self, frame_size=(1000, 600)):
        self._frame_size = frame_size

    def frame_size(self):
        return self._frame_size


def player(track_id, x, y, w=100, h=180, *, knife=False, gun=False, self_player=False):
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
        has_gun=gun,
        is_self=self_player,
        murderer_confidence=0.97 if knife else 0.0,
        sheriff_confidence=0.90 if gun else 0.0,
    )


class MM2Tests(unittest.TestCase):
    def test_role_requires_stability(self):
        detector = MM2RoleDetector(stable_frames=2)
        role, confidence, phase = detector.detect(["You are Sheriff"])
        self.assertEqual(role, MM2Role.UNKNOWN)
        self.assertEqual(confidence, 0.0)
        self.assertEqual(detector.candidate_role, MM2Role.SHERIFF)
        self.assertLess(detector.candidate_confidence, 0.9)
        self.assertEqual(phase, RoundPhase.ROLE_REVEAL)

        role, confidence, phase = detector.detect(["You are Sheriff"])
        self.assertEqual(role, MM2Role.SHERIFF)
        self.assertGreater(confidence, 0.9)
        self.assertEqual(phase, RoundPhase.ROUND)

        role, confidence, phase = detector.detect(["Coins: 12"])
        self.assertEqual(role, MM2Role.SHERIFF)
        self.assertGreater(confidence, 0.9)
        self.assertEqual(phase, RoundPhase.ROUND)

        role, confidence, phase = detector.detect(["Victory!"])
        self.assertEqual(role, MM2Role.UNKNOWN)
        self.assertEqual(phase, RoundPhase.ROUND_END)

    def test_threat_model_ignores_self_as_murderer_candidate(self):
        own = player(1, 450, 350, knife=True, self_player=True)
        other = player(2, 100, 180)
        scene = MM2SceneSnapshot((1000, 600), players=[own, other])
        model = MM2ThreatModel()
        model.update(scene)
        self.assertIsNone(model.murderer(scene))

    def test_sheriff_refuses_low_confidence_target(self):
        inputs = FakeInput()
        combat = MM2Combat(inputs)
        suspect = player(2, 500, 150)
        suspect.murderer_confidence = 0.70
        scene = MM2SceneSnapshot((1000, 600), players=[suspect])

        fired, reason = combat.sheriff_fire(suspect, scene)

        self.assertFalse(fired)
        self.assertEqual(reason, "MURDERER_CONFIDENCE_TOO_LOW")
        self.assertFalse(any(a[0] == "click" for a in inputs.actions))

    def test_sheriff_refuses_crowded_shot(self):
        inputs = FakeInput()
        combat = MM2Combat(inputs)
        murderer = player(2, 430, 120)
        murderer.murderer_confidence = 0.99
        innocent = player(3, 445, 140)
        scene = MM2SceneSnapshot((1000, 600), players=[murderer, innocent])

        fired, reason = combat.sheriff_fire(murderer, scene)

        self.assertFalse(fired)
        self.assertEqual(reason, "FRIENDLY_FIRE_RISK")

    def test_sheriff_fires_clear_confirmed_target(self):
        inputs = FakeInput()
        combat = MM2Combat(inputs, MM2CombatConfig(fire_cooldown_s=0.0))
        murderer = player(2, 430, 120)
        murderer.murderer_confidence = 0.99
        scene = MM2SceneSnapshot((1000, 600), players=[murderer])

        fired, reason = combat.sheriff_fire(murderer, scene)

        self.assertTrue(fired)
        self.assertEqual(reason, "SHOT_FIRED")
        self.assertTrue(any(a[0] == "key" and a[1] == "1" for a in inputs.actions))
        self.assertTrue(any(a[0] == "aim" for a in inputs.actions))
        self.assertTrue(any(a[0] == "click" and a[3] == "left" for a in inputs.actions))

    def test_murderer_melee_when_close_and_throw_when_far(self):
        inputs = FakeInput()
        combat = MM2Combat(
            inputs,
            MM2CombatConfig(
                melee_area_ratio=0.05,
                throw_min_area_ratio=0.005,
                melee_cooldown_s=0.0,
                throw_cooldown_s=0.0,
            ),
        )
        close = player(2, 350, 100, w=260, h=260)
        far = player(3, 480, 200, w=70, h=70)

        acted, _, mode = combat.murderer_attack(
            close, MM2SceneSnapshot((1000, 600), players=[close])
        )
        self.assertTrue(acted)
        self.assertEqual(mode, "melee")

        acted, _, mode = combat.murderer_attack(
            far, MM2SceneSnapshot((1000, 600), players=[far])
        )
        self.assertTrue(acted)
        self.assertEqual(mode, "throw")
        self.assertTrue(any(a[0] == "click" and a[3] == "right" for a in inputs.actions))

    def test_survival_strafes_away_from_close_threat(self):
        inputs = FakeInput()
        survival = MM2Survival(
            inputs,
            SurvivalConfig(danger_area_ratio=0.01, panic_area_ratio=0.02),
        )
        threat = player(2, 700, 120, w=180, h=180)
        scene = MM2SceneSnapshot((1000, 600), players=[threat])

        moved, reason = survival.evade(threat, scene)

        self.assertTrue(moved)
        self.assertEqual(reason, "PANIC_EVADE")
        self.assertTrue(any(a[0] == "move" and a[1] == "left" for a in inputs.actions))


    def test_autonomy_does_not_attack_on_single_unstable_role_frame(self):
        inputs = FakeInput()
        autonomy = MM2Autonomy(FakePerception(), inputs)
        autonomy.survival.evade = lambda *_args, **_kwargs: (False, "SAFE")

        murderer = player(2, 430, 120, knife=True)
        scene = MM2SceneSnapshot(
            (1000, 600),
            players=[murderer],
            ui_text=["You are Sheriff"],
        )
        autonomy.scene_reader.observe = lambda: scene

        status = autonomy.tick()

        self.assertEqual(status["role"], MM2Role.UNKNOWN.value)
        self.assertEqual(status["phase"], RoundPhase.ROLE_REVEAL.value)
        self.assertEqual(status["last_action"], "waiting_for_stable_role")
        self.assertFalse(any(action[0] == "click" for action in inputs.actions))

    def test_autonomy_allows_offense_after_role_stabilizes(self):
        inputs = FakeInput()
        autonomy = MM2Autonomy(FakePerception(), inputs)
        autonomy.survival.evade = lambda *_args, **_kwargs: (False, "SAFE")
        autonomy.combat.config.fire_cooldown_s = 0.0
        autonomy.tick_interval_s = 0.0

        murderer = player(2, 430, 120, knife=True)
        scene = MM2SceneSnapshot(
            (1000, 600),
            players=[murderer],
            ui_text=["You are Sheriff"],
        )
        autonomy.scene_reader.observe = lambda: scene

        first = autonomy.tick()
        second = autonomy.tick()

        self.assertEqual(first["role"], MM2Role.UNKNOWN.value)
        self.assertEqual(second["role"], MM2Role.SHERIFF.value)
        self.assertEqual(second["phase"], RoundPhase.ROUND.value)
        self.assertTrue(any(action[0] == "click" for action in inputs.actions))

    def test_visual_role_fallback_does_not_latch_during_round_end(self):
        inputs = FakeInput()
        autonomy = MM2Autonomy(FakePerception(), inputs)
        autonomy.tick_interval_s = 0.0
        autonomy.survival.evade = lambda *_args, **_kwargs: (False, "SAFE")

        own = player(1, 450, 350, knife=True, self_player=True)
        scene = MM2SceneSnapshot(
            (1000, 600),
            players=[own],
            ui_text=["Victory!"],
        )
        autonomy.scene_reader.observe = lambda: scene

        autonomy.tick()
        status = autonomy.tick()

        self.assertEqual(status["role"], MM2Role.UNKNOWN.value)
        self.assertEqual(status["phase"], RoundPhase.ROUND_END.value)
        self.assertEqual(status["stats"]["knife_swings"], 0)
        self.assertEqual(status["stats"]["knife_throws"], 0)

    def test_scene_preserves_zero_track_id_and_separates_untracked_players(self):
        reader = MM2SceneReader(FakePerception())
        detections = [
            Detection("player", 0.9, 20, 20, 50, 80, source="test", track_id=0),
            Detection("player", 0.9, 120, 20, 50, 80, source="test", track_id=None),
            Detection("player", 0.9, 220, 20, 50, 80, source="test", track_id=None),
        ]
        reader._detections = lambda: detections
        reader._texts = lambda: []

        scene = reader.observe()
        track_ids = [p.track_id for p in scene.players]

        self.assertEqual(track_ids[0], 0)
        self.assertEqual(track_ids[1:], [-1, -2])
        self.assertEqual(len(set(track_ids)), 3)


if __name__ == "__main__":
    unittest.main()
