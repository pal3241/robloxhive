from __future__ import annotations

from threading import Lock
from typing import Any

from robloxhive.brain.goals import GoalManager
from robloxhive.brain.memory import GameMemory
from robloxhive.brain.planner import KnowledgePlanner
from robloxhive.shared.command_bus import Command, CommandBus
from robloxhive.shared.models import ActionResult, ActionStatus, Goal, Plan, PlanStatus, StepStatus


class AgentRuntime:
    """Connect GoalManager -> KnowledgePlanner -> CommandBus -> result verification."""

    def __init__(
        self,
        memory: GameMemory,
        goals: GoalManager | None = None,
        planner: KnowledgePlanner | None = None,
        commands: CommandBus | None = None,
    ) -> None:
        self.memory = memory
        self.goals = goals or GoalManager()
        self.planner = planner or KnowledgePlanner()
        self.commands = commands or CommandBus()
        self._plans: dict[str, Plan] = {}
        self._active_plan_id: str | None = None
        self._lock = Lock()

    def start_goal(self, game_id: int, goal: Goal) -> Plan:
        knowledge = self.memory.load_knowledge(game_id)
        self.goals.interrupt(goal)
        plan = self.planner.plan(game_id, goal, knowledge)
        with self._lock:
            self._plans[plan.id] = plan
            self._active_plan_id = plan.id
        self._dispatch_current(plan)
        return plan

    def _dispatch_current(self, plan: Plan) -> None:
        step = plan.current_step
        if step is None:
            plan.status = PlanStatus.COMPLETE
            return
        step.status = StepStatus.RUNNING
        plan.status = PlanStatus.RUNNING
        self.commands.publish(
            Command(
                source="planner",
                type="EXECUTE_SKILL",
                payload={
                    "plan_id": plan.id,
                    "game_id": plan.game_id,
                    "step_index": step.index,
                    "skill": step.skill,
                    "instruction": step.instruction,
                    "success_condition": step.success_condition,
                    "metadata": step.metadata,
                },
            )
        )

    def get_plan(self, plan_id: str | None = None) -> Plan | None:
        with self._lock:
            pid = plan_id or self._active_plan_id
            return self._plans.get(pid) if pid else None

    def record_result(
        self,
        plan_id: str,
        step_index: int,
        result: ActionResult,
        evidence: dict[str, Any] | None = None,
    ) -> Plan:
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise KeyError(f"Unknown plan {plan_id}")
            if step_index < 0 or step_index >= len(plan.steps):
                raise IndexError("step_index out of range")
            step = plan.steps[step_index]
            if step_index != plan.current_step_index:
                raise ValueError("Result does not belong to the active step")

            step.attempts += 1
            step.last_error = result.error
            step.evidence = evidence or result.details

            if result.status == ActionStatus.SUCCESS:
                step.status = StepStatus.COMPLETE
                self._record_experience(plan, step, True)
                self._verify_referenced_knowledge(plan.game_id, step, True, evidence)
                plan.current_step_index += 1
                if plan.current_step_index >= len(plan.steps):
                    plan.status = PlanStatus.COMPLETE
                    self.goals.active = None
                    return plan
            elif result.recoverable and step.attempts < step.max_attempts:
                step.status = StepStatus.PENDING
                self._record_experience(plan, step, False)
            else:
                step.status = StepStatus.FAILED
                plan.status = PlanStatus.BLOCKED
                self._record_experience(plan, step, False)
                self._verify_referenced_knowledge(plan.game_id, step, False, evidence)
                return plan

        self._dispatch_current(plan)
        return plan

    def _record_experience(self, plan: Plan, step: Any, success: bool) -> None:
        self.memory.remember(
            plan.game_id,
            "experiences",
            {
                "goal": plan.goal.type,
                "goal_target": plan.goal.target,
                "plan_id": plan.id,
                "step": step.instruction,
                "skill": step.skill,
                "success": success,
                "attempt": step.attempts,
                "evidence": step.evidence,
            },
        )

    def _verify_referenced_knowledge(
        self,
        game_id: int,
        step: Any,
        success: bool,
        evidence: dict[str, Any] | None,
    ) -> None:
        # Source IDs alone are not enough to identify one knowledge row safely.
        # We therefore update entries by exact instruction text, preserving source traceability.
        knowledge = self.memory.load_knowledge(game_id)
        changed = False
        for section, rows in knowledge.items():
            if not isinstance(rows, list):
                continue
            for item in rows:
                if not isinstance(item, dict) or item.get("text") != step.instruction:
                    continue
                item["last_gameplay_evidence"] = evidence or {}
                item["verification_attempts"] = int(item.get("verification_attempts", 0)) + 1
                if success:
                    item["verified_in_game"] = True
                    item["confidence"] = min(1.0, max(float(item.get("confidence", 0.0)), 0.9))
                else:
                    item["verification_failures"] = int(item.get("verification_failures", 0)) + 1
                    item["confidence"] = max(0.05, float(item.get("confidence", 0.0)) * 0.75)
                changed = True
        if changed:
            self.memory.save_knowledge(game_id, knowledge)

    def next_command(self, timeout: float | None = None) -> Command | None:
        return self.commands.receive(timeout=timeout)
