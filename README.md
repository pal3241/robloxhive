# RobloxHive

Distributed autonomous Roblox agent: **Brain Node** for reasoning/memory and a Windows-only **Body Node** for Roblox control, perception, and realtime navigation.

## v0.7.0 — Local Mapping + A* Navigation

The `navigate` skill now uses a local navigation engine instead of only steering toward the target image.

```text
Target detection
      ↓
Bot-window capture
      ↓
Local obstacle estimation
      ↓
Egocentric occupancy grid
      ↓
A*
      ↓
Local waypoint / next cell
      ↓
Move
      ↓
Motion verification
      ↓
stuck?
  ├── no → rebuild map + replan
  └── yes → recovery → replan
```

The planner remains semantic: the Brain says **navigate to shop**; the Windows Body decides the actual local path.

## Local occupancy grid

The navigator maintains an egocentric grid around the bot.

```text
? ? ? ? ? ? ? G ? ? ?
? ? ? ? * * * * ? ? ?
? ? # # * # # ? ? ? ?
? ? # * * # ? ? ? ? ?
? . * * # ? ? ? ? ? ?
? . * # ? ? ? ? ? ? ?
? . * . . ? ? ? ? ? ?
? . B . . ? ? ? ? ? ?
```

Legend:

- `B` — bot
- `G` — current local goal
- `*` — A* path
- `#` — likely obstacle
- `.` — observed free
- `?` — unknown

Unknown cells are still traversable but have a higher A* cost, so the bot prefers known-free space without becoming unable to explore.

## Obstacle estimation

`ScreenObstacleEstimator` uses the lower portion of the Roblox frame and divides it into local grid cells.

For each cell it measures:

- edge density
- local texture/variance
- perspective-adjusted thresholds

Strong nearby structure is marked blocked. Low-edge regions can be marked free.

This is deliberately lightweight for CPU-only operation. It is a local 2D visual estimate, **not yet monocular SLAM or true Roblox geometry**.

## A* pathfinding

`robloxhive/body/navigation/astar.py` performs A* over:

- forward
- left/right strafe
- diagonal forward cells
- limited backwards movement

Costs favor:

1. observed-free cells
2. short paths
3. forward movement
4. unknown cells only when useful

Blocked cells are never entered.

The occupancy map is rebuilt repeatedly while moving, so dynamic changes cause replanning instead of committing to one stale path.

## Motion / stuck detection

Every navigation movement captures:

```text
frame BEFORE
   ↓
movement input
   ↓
frame AFTER
```

The central game view is compared. If the visual change is below the motion threshold, the movement is treated as suspicious/stuck.

Repeated failed displacement triggers the recovery ladder:

```text
1. stop
2. back
3. strafe left
4. strafe right
5. longer back/side escape
6. rebuild map
7. A* replan
8. return STUCK / PATH_UNREACHABLE if recovery budget is exhausted
```

That failure is returned to the Brain rather than looping into a wall forever.

## Skill integration

These skills now use the local navigator when approaching a target:

- `navigate`
- `collect`
- `interact`

`follow_player` keeps its faster visual-tracking loop because the target is dynamic and continual.

```text
collect fuel
   ↓
navigator.navigate_to("fuel")
   ↓
A* / avoid obstacles
   ↓
arrive
   ↓
interact
   ↓
verify collection
```

The old visual approach code remains as a fallback when a navigator is not attached, which keeps unit testing and alternate Body implementations simple.

## Navigation Dashboard

Dashboard tabs now include:

```text
Overview
Instances
Belajar
Agent
Skills
Perception
Navigation   ← v0.7
Memory
```

The Navigation tab exposes:

- local occupancy map
- current A* path
- local goal cell
- replan count
- stuck-event count
- recovery count
- last motion score
- last result
- Body online/offline state

You can request a **Navigation Probe** without moving the character. It retrieves the current navigator state from the Windows Body.

## Perception stack

Navigation builds on the v0.6 fused perception stack:

```text
PlayerTracker
      ↓
ONNX object detector
      ↓
OCR / UI text
      ↓
Template fallback
```

Objects/locations are still found by perception; A* determines how to locally move toward them.

## Windows Body

Install:

```powershell
pip install -e ".[windows]"
```

Find Roblox windows:

```powershell
python -m robloxhive body --list-windows
```

Start the explicitly selected bot instance:

```powershell
python -m robloxhive body ^
  --brain-url http://192.168.1.50:8765 ^
  --agent-id agent-01 ^
  --game-id 123456789 ^
  --pid 24680
```

Optional detector:

```powershell
python -m robloxhive body ^
  --brain-url http://192.168.1.50:8765 ^
  --agent-id agent-01 ^
  --game-id 123456789 ^
  --pid 24680 ^
  --model data\models\123456789\detector.onnx ^
  --labels data\models\123456789\labels.txt
```

## Brain stays lightweight

The Brain still does not need OpenCV, NumPy, ONNX Runtime, or Windows APIs.

```text
Phone / remote Brain
├── Qwen / Ollama
├── internet learning
├── game memory
├── Goal Manager
└── high-level Planner

Windows Body
├── Roblox
├── fused perception
├── occupancy mapping
├── A*
├── stuck/recovery
└── HWND-scoped control
```

## Per-game learning and memory

The **Belajar** tab researches tutorials, progression, tips, items, and completion strategies from the internet, then synthesizes:

```text
data/games/<game_id>/
├── profile.json
├── knowledge.json
└── research/
    └── research-<timestamp>.json
```

The Goal Manager and Planner use this knowledge to decide **where and why** to go. The Windows navigator decides **how** to get there locally.

## Safety / failure behavior

- Human Roblox windows can remain `PROTECTED`.
- Body control is tied to an explicit PID/HWND.
- Multiple Roblox windows are never auto-guessed.
- Capture remains scoped to the bot window.
- Unknown skills fail closed.
- Navigation has bounded iterations and bounded recovery.
- `STUCK`, `TARGET_LOST`, `PATH_UNREACHABLE`, and `NAVIGATION_TIMEOUT` are explicit failures.
- A navigation action is not declared successful until the visual target is close and centered.

## Current boundary

v0.7.0 is a **local egocentric A* navigator**, not full persistent world-scale SLAM.

It can:

- detect local likely obstacles
- generate a local grid
- route around blocked cells
- continuously replan
- verify visual movement
- recover from stuck states
- expose the map and path in the dashboard

For very large worlds, the next layer should add a persistent **topological/semantic waypoint graph** (Spawn → Station → Shop → Mine, etc.) above this local navigator. That graph can use A* globally while v0.7 handles obstacle avoidance between nearby waypoints.
