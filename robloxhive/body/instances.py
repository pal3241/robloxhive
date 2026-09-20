from __future__ import annotations

from dataclasses import dataclass
from robloxhive.shared.models import InstanceRole


class ProtectedInstanceError(RuntimeError):
    pass


class InstanceOwnershipError(RuntimeError):
    pass


@dataclass(slots=True)
class RobloxInstance:
    pid: int
    hwnd: int
    title: str
    role: InstanceRole = InstanceRole.UNASSIGNED
    agent_id: str | None = None
    protected: bool = False
    alive: bool = True


class InstanceManager:
    """Authoritative PID/HWND ownership registry.

    Deliberately never guesses a replacement window when an assigned process dies.
    """

    def __init__(self) -> None:
        self._instances: dict[int, RobloxInstance] = {}

    def register(self, instance: RobloxInstance) -> None:
        self._instances[instance.pid] = instance

    def mark_player(self, pid: int) -> None:
        inst = self._instances[pid]
        inst.role = InstanceRole.PLAYER
        inst.agent_id = None
        inst.protected = True

    def assign_bot(self, pid: int, agent_id: str) -> None:
        inst = self._instances[pid]
        if inst.protected:
            raise ProtectedInstanceError(f"PID {pid} is protected")
        inst.role = InstanceRole.BOT
        inst.agent_id = agent_id

    def resolve_for_agent(self, agent_id: str) -> RobloxInstance:
        matches = [i for i in self._instances.values() if i.agent_id == agent_id and i.alive]
        if len(matches) != 1:
            raise InstanceOwnershipError(
                f"Expected exactly one live instance for {agent_id}, found {len(matches)}"
            )
        inst = matches[0]
        if inst.protected or inst.role != InstanceRole.BOT:
            raise ProtectedInstanceError(f"Refusing control of PID {inst.pid}")
        return inst

    def mark_dead(self, pid: int) -> None:
        if pid in self._instances:
            self._instances[pid].alive = False

    def list(self) -> list[RobloxInstance]:
        return list(self._instances.values())
