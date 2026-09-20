from robloxhive.brain.goals import GoalManager
from robloxhive.brain.memory import GameMemory
from robloxhive.shared.command_bus import CommandBus


def main() -> None:
    GoalManager()
    GameMemory()
    CommandBus()
    print("RobloxHive 0.1.0 foundation ready.")
    print("Brain, Body, memory, instance ownership, and command bus initialized.")


if __name__ == "__main__":
    main()
