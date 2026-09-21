import unittest

from robloxhive.shared.command_bus import Command, CommandBus


class CommandBusRoutingTests(unittest.TestCase):
    def test_receive_for_keeps_other_agent_commands(self):
        bus = CommandBus()
        bus.publish(Command(source="dashboard", type="DIRECT_INPUT", payload={"agent_id": "agent-02"}))
        bus.publish(Command(source="dashboard", type="DIRECT_INPUT", payload={"agent_id": "agent-01"}))

        first = bus.receive_for("agent-01", timeout=0.01)
        second = bus.receive_for("agent-02", timeout=0.01)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.payload["agent_id"], "agent-01")
        self.assertEqual(second.payload["agent_id"], "agent-02")

    def test_untargeted_commands_default_to_agent_01(self):
        bus = CommandBus()
        bus.publish(Command(source="planner", type="EXECUTE_SKILL", payload={"skill": "navigate"}))

        self.assertIsNone(bus.receive_for("agent-02", timeout=0.01))
        command = bus.receive_for("agent-01", timeout=0.01)
        self.assertIsNotNone(command)
        self.assertEqual(command.type, "EXECUTE_SKILL")


if __name__ == "__main__":
    unittest.main()
