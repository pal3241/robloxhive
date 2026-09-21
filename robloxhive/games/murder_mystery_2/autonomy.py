from __future__ import annotations

import time
from typing import Any

from robloxhive.games.murder_mystery_2.combat import MM2Combat
from robloxhive.games.murder_mystery_2.event_reasoning import WitnessedKillReasoner
from robloxhive.games.murder_mystery_2.models import MM2Role, MM2State, RoundPhase
from robloxhive.games.murder_mystery_2.role_detector import MM2RoleDetector
from robloxhive.games.murder_mystery_2.scene import MM2SceneReader
from robloxhive.games.murder_mystery_2.survival import MM2Survival
from robloxhive.games.murder_mystery_2.threat import MM2ThreatModel


class MM2Autonomy:
    """Low-latency per-round autonomy for Murder Mystery 2.

    Survival is always evaluated before role-specific offense. Combat is gated
    by a stable role and, for Sheriff/Hero, high-confidence murderer evidence.
    """

    def __init__(
        self,
        perception: Any,
        input_backend: Any,
        navigator: Any | None = None,
    ) -> None:
        self.perception = perception
        self.input = input_backend
        self.navigator = navigator
        self.scene_reader = MM2SceneReader(perception)
        self.role_detector = MM2RoleDetector(stable_frames=2)
        self.threats = MM2ThreatModel()
        self.kill_reasoner = WitnessedKillReasoner()
        self.survival = MM2Survival(input_backend)
        self.combat = MM2Combat(input_backend)
        self.state = MM2State(enabled=True)
        self._last_tick = 0.0
        self.tick_interval_s = 0.10

    def control(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "").lower()
        if action == "enable":
            self.state.enabled = True
        elif action == "disable":
            self.state.enabled = False
            self.input.release_all()
        elif action == "role_override":
            value = str(payload.get("role") or "unknown").lower()
            self.state.role_override = None if value in {"", "auto", "none"} else MM2Role(value)
        elif action == "clear_role":
            self.state.role_override = None
        else:
            return {"ok": False, "error": "UNKNOWN_MM2_CONTROL"}
        return {"ok": True, "state": self.status()}

    def _choose_murderer_target(self, scene):
        if not scene.players:
            return None
        # Sheriff/gun holder first; then the visually nearest/largest target.
        others = [p for p in scene.players if not p.is_self]
        gun_holders = [p for p in others if p.has_gun]
        if gun_holders:
            return max(gun_holders, key=lambda p: p.detection.area)
        return max(others, key=lambda p: p.detection.area) if others else None

    def tick(self) -> dict[str, Any]:
        now = time.monotonic()
        if not self.state.enabled or now - self._last_tick < self.tick_interval_s:
            return self.status()
        self._last_tick = now
        self.state.ticks += 1

        scene = self.scene_reader.observe()
        self.threats.update(scene)
        kill_events = self.kill_reasoner.update(scene, self.threats)

        detected_role, role_confidence, phase = self.role_detector.detect(scene.ui_text)

        # Visual self-weapon evidence is a fallback when OCR role text is unavailable.
        own = next((p for p in scene.players if p.is_self), None)
        if detected_role is MM2Role.UNKNOWN and own is not None:
            if own.has_knife:
                detected_role, role_confidence, phase = MM2Role.MURDERER, 0.90, RoundPhase.ROUND
                self.role_detector.current_role = MM2Role.MURDERER
            elif own.has_gun:
                detected_role, role_confidence, phase = MM2Role.SHERIFF, 0.84, RoundPhase.ROUND
                self.role_detector.current_role = MM2Role.SHERIFF

        role = self.state.role_override or detected_role
        self.state.role = role
        self.state.role_confidence = 1.0 if self.state.role_override else role_confidence
        self.state.phase = phase

        murderer = self.threats.murderer(scene)
        sheriff = self.threats.sheriff(scene)
        self.state.murderer_track_id = murderer.track_id if murderer else None
        self.state.murderer_confidence = murderer.murderer_confidence if murderer else 0.0
        self.state.sheriff_track_id = sheriff.track_id if sheriff else None

        # Survival always runs before offense.
        survival_threat = sheriff if role is MM2Role.MURDERER else murderer
        if survival_threat is not None:
            moved, reason = self.survival.evade(survival_threat, scene)
            if moved:
                self.state.mode = "survival"
                self.state.last_action = reason
                self.state.last_reason = (
                    "visible_sheriff_too_close"
                    if role is MM2Role.MURDERER
                    else "visible_murderer_too_close"
                )
                self.state.survival_moves += 1
                return self._finish(scene)

        if role in {MM2Role.SHERIFF, MM2Role.HERO}:
            self.state.mode = "sheriff_combat"
            if murderer is None:
                self.state.last_action = "hold_fire"
                self.state.last_reason = "murderer_not_confirmed"
                return self._finish(scene)

            fired, reason = self.combat.sheriff_fire(murderer, scene)
            self.state.target_track_id = murderer.track_id
            self.state.last_action = reason
            self.state.last_reason = None if fired else reason
            if fired:
                self.state.shots += 1
            return self._finish(scene)

        if role is MM2Role.MURDERER:
            self.state.mode = "murderer_hunt"
            target = self._choose_murderer_target(scene)
            if target is None:
                self.state.last_action = "search_players"
                self.state.last_reason = "no_visible_target"
                return self._finish(scene)

            self.state.target_track_id = target.track_id
            acted, reason, attack_mode = self.combat.murderer_attack(target, scene)
            self.state.last_action = reason
            self.state.last_reason = None if acted else reason
            if acted and attack_mode == "melee":
                self.state.knife_swings += 1
            elif acted and attack_mode == "throw":
                self.state.knife_throws += 1
            elif not acted and attack_mode == "approach":
                # Use fast local steering toward the visible player; next tick re-evaluates.
                w, _ = scene.frame_size
                offset = target.detection.center_x / w - 0.5 if w > 0 else 0.0
                if abs(offset) > 0.12:
                    self.input.move("right" if offset > 0 else "left", 0.12)
                else:
                    self.input.move("forward", 0.16)
            return self._finish(scene)

        if role is MM2Role.INNOCENT:
            self.state.mode = "survive"
            self.state.last_action = "observe_and_survive"
            self.state.last_reason = None
            return self._finish(scene)

        self.state.mode = "observe"
        self.state.last_action = "waiting_for_stable_role"
        self.state.last_reason = None
        return self._finish(scene)

    def _finish(self, scene) -> dict[str, Any]:
        self.state.diagnostics = {
            "players_visible": len(scene.players),
            "knives_visible": len(scene.knives),
            "guns_visible": len(scene.guns),
            "dropped_guns_visible": len(scene.dropped_guns),
            "bodies_visible": len(scene.bodies),
            "ui_text_sample": scene.ui_text[:12],
            "kill_events_recent": self.kill_reasoner.recent(),
            "threat_evidence_recent": list(self.threats.evidence_log)[-10:],
            "new_kill_events": len(kill_events),
        }
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "adapter": "murder_mystery_2",
            "enabled": self.state.enabled,
            "role": self.state.role.value,
            "role_confidence": round(self.state.role_confidence, 3),
            "role_override": self.state.role_override.value if self.state.role_override else None,
            "phase": self.state.phase.value,
            "murderer_track_id": self.state.murderer_track_id,
            "murderer_confidence": round(self.state.murderer_confidence, 3),
            "sheriff_track_id": self.state.sheriff_track_id,
            "target_track_id": self.state.target_track_id,
            "mode": self.state.mode,
            "last_action": self.state.last_action,
            "last_reason": self.state.last_reason,
            "stats": {
                "ticks": self.state.ticks,
                "survival_moves": self.state.survival_moves,
                "shots": self.state.shots,
                "knife_swings": self.state.knife_swings,
                "knife_throws": self.state.knife_throws,
            },
            "diagnostics": dict(self.state.diagnostics),
        }
