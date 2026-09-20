# RobloxHive

Modular autonomous Roblox agent architecture.

RobloxHive separates the **Brain Node** (LLM, goals, planner, per-game memory) from the Windows-only **Body Node** (Roblox, pyrobloxbot, window ownership, local failsafes). The dashboard/CLI communicate through a command bus; Roblox chat is not used as a control channel.

## Safety architecture
- Every bot is bound to an explicit Roblox PID/HWND.
- Player-owned windows can be marked PROTECTED.
- No automatic reassignment after a bot process disappears.
- Local body watchdog releases controls if the brain disconnects.
- Actions return structured results and are verified/retried by higher layers.

## Layout
- `robloxhive/brain`: goals, planner, memory
- `robloxhive/body`: Windows instance ownership, controllers, failsafes
- `robloxhive/shared`: protocol/models shared across devices
- `robloxhive/interface`: CLI/dashboard entry points
- `data/games/<game_id>/`: per-game persistent memory

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\activate
pip install -e .
python -m robloxhive
```

This first foundation intentionally keeps low-level Roblox automation behind interfaces so the Brain can later run on Android/another PC/server while Windows remains the Roblox interface.
