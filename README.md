# RobloxHive

## v0.5.0 — Generic Visual Skills + Skills Dashboard

Four requested Windows Body skills are now implemented:

- `navigate` — locate a visual target, steer toward it, and approach until it is near.
- `collect` — navigate to an item, interact, then require visual change/disappearance as verification.
- `interact` — navigate to a target, press the configured interaction key, and require visual change when a target is provided.
- `follow_player` — track a player template and maintain a configurable visual distance band.

The realtime loop is lightweight and stays on the Windows Body:

```text
TemplateVision
   ↓
Detection (x/y/size/confidence)
   ↓
GenericVisualSkills
   ↓
Win32MessageInput → assigned Roblox HWND
   ↓
capture again
   ↓
ActionResult + evidence
```

This is a **visual baseline**, not the final A*/SLAM pathfinder. It does not yet understand arbitrary 3D geometry or obstacles. The code deliberately verifies actions from perception and fails/retries instead of claiming success blindly.

### Dashboard Skills tab

The dashboard now shows connected Windows Body nodes, their advertised skills, PID/HWND/Game ID metadata, and recent skill results. You can manually test:

```text
navigate
collect
interact
follow_player
```

Each test is sent through the same Body command channel used by the autonomous planner.

### Run a Windows Body

Install the Body dependencies:

```powershell
pip install -e ".[windows]"
```

List Roblox windows:

```powershell
python -m robloxhive body --list-windows
```

When more than one Roblox window exists, RobloxHive intentionally refuses to guess which one is the bot. Start it with the bot PID explicitly:

```powershell
python -m robloxhive body ^
  --brain-url http://192.168.1.50:8765 ^
  --agent-id agent-01 ^
  --game-id 123456789 ^
  --pid 24680
```

The Brain can be running on a phone/another PC. The Windows Body reports its skills back to the dashboard every few seconds.

### Visual templates

The current lightweight detector uses OpenCV template matching:

```text
data/templates/
└── <game_id>/
    ├── shop/
    │   ├── shop-1.png
    │   └── shop-2.png
    ├── fuel/
    │   └── fuel.png
    ├── door/
    │   └── door.png
    └── Fahri/
        └── player.png
```

The dashboard target field uses these labels. Multiple screenshots per label improve robustness across angle/UI changes.

### Important input limitation

The default `Win32MessageInput` sends HWND-scoped Windows messages so it does not intentionally take over the global keyboard/mouse. Some Roblox builds may ignore background `WM_KEY*`/mouse messages because the game can use lower-level input APIs. RobloxHive therefore always relies on visual verification. If the target does not move/change, the skill fails instead of pretending the input worked.

---


Modular autonomous Roblox agent architecture with a split **Brain Node** and Windows **Body Node**.

## v0.4.0 — Knowledge → Goal → Planner → Body → Evidence

RobloxHive now connects learned game knowledge to an executable agent loop.

```text
Internet
   ↓
Research
   ↓
Knowledge Synthesis
   ↓
Per-game knowledge.json
   ↓
Goal Manager
   ↓
KnowledgePlanner
   ↓
PlanStep[]
   ↓
CommandBus
   ↓
Windows Body / SkillExecutor
   ↓
ActionResult + gameplay evidence
   ↓
retry / advance / block
   ↓
Experience Memory + Knowledge Verification
```

The Brain never sends raw keyboard timing as the high-level plan. It emits semantic skills such as:

- `navigate`
- `collect`
- `combat`
- `follow_player`
- `quest`
- `interact`
- `explore`
- `observe_and_act`

Game adapters on the Windows Body register implementations for those skills. Unknown skills fail closed with `UNSUPPORTED_SKILL` instead of falling through to arbitrary input.

## Goal system

The Dashboard now has an **Agent** tab. A user can choose a learned game and submit goals such as:

- Autonomous Progress
- Complete Game
- Quest / Daily
- Follow Player
- Combat
- Explore

Example:

```text
Game: Dead Rails
Goal: Complete Game
Instruction:
Use the learned game knowledge to finish the game, verify each step
from gameplay, and recover/retry when a step fails.
```

The planner loads that Game ID's `knowledge.json`, scores relevant knowledge, and creates a bounded plan. Broad progression goals prefer the learned progression ordering; direct goals like follow/combat/quest become direct semantic skills.

## Failsafe and verification behavior

Every PlanStep tracks:

```text
status
attempts
max_attempts
confidence
success_condition
last_error
evidence
knowledge source refs
```

A recoverable failure is retried up to the step's configured attempt limit. A non-recoverable failure blocks the plan instead of continuing blindly.

**Action success is not treated as proof that an internet claim is true.** Knowledge is promoted to `verified_in_game=true` only when Body/perception evidence explicitly sends:

```json
{"verified": true}
```

Contradictory gameplay evidence can send:

```json
{"contradicts": true}
```

which reduces confidence while preserving the original source traceability.

## Brain ↔ Windows Body protocol

The Brain/dashboard exposes:

```text
POST /api/agent/goals
GET  /api/agent/active-plan
GET  /api/agent/plans/{plan_id}

GET  /api/body/commands/next
POST /api/body/results
```

The Windows Body can use `BodyBridge` to poll the Brain over LAN:

```python
from robloxhive.body.bridge import BodyBridge
from robloxhive.body.skills import SkillExecutor

skills = SkillExecutor()

# Register real handlers from a generic/game-specific adapter.
# skills.register("navigate", ...)
# skills.register("combat", ...)
# skills.register("collect", ...)

bridge = BodyBridge(
    brain_url="http://192.168.1.50:8765",
    executor=skills,
    agent_id="agent-01",
)
bridge.run_forever()
```

This means the heavy Brain can run on an Android phone, another PC, or a server while Windows remains the Roblox interface.

## Internet learning and synthesis

The **Belajar** tab researches:

1. beginner tutorial
2. full walkthrough / start-to-finish
3. progression / wiki
4. tips, tricks and strategy
5. important items, weapons, classes and upgrades
6. ending / completion / how to win
7. the user's custom objective

The research corpus is synthesized using Ollama/Qwen when configured, with a deterministic fallback when no model is available.

Knowledge sections include:

- objectives
- progression
- mechanics
- items/upgrades
- enemies
- locations
- strategies/tips
- common mistakes
- endgame/win conditions
- unknowns
- conflicts

## Per-game memory

Memory is isolated by Roblox Game/Universe ID:

```text
data/games/<game_id>/
├── profile.json
├── knowledge.json
└── research/
    └── research-<timestamp>.json
```

Runtime experiences are also appended to the per-game profile, allowing the same game to accumulate successes, failures, and verification evidence across sessions.

## Use Qwen / Ollama on another device

### Linux / Termux

```bash
export ROBLOXHIVE_LLM_MODEL="qwen2.5:3b"
export ROBLOXHIVE_OLLAMA_URL="http://127.0.0.1:11434"
python -m robloxhive dashboard --host 0.0.0.0 --port 8765
```

### Windows PowerShell

```powershell
$env:ROBLOXHIVE_LLM_MODEL="qwen2.5:3b"
$env:ROBLOXHIVE_OLLAMA_URL="http://192.168.1.50:11434"
python -m robloxhive dashboard --host 0.0.0.0 --port 8765
```

If `ROBLOXHIVE_LLM_MODEL` is not set, internet learning still works with the lightweight heuristic synthesizer.

## Windows instance safety

- Controlled Roblox instances are bound to PID + HWND.
- Human/player windows can be marked `PROTECTED`.
- The Body does not guess a replacement bot window after a process disappears.
- Automatic reassignment is intentionally avoided.
- Unknown semantic skills fail safe.
- Local watchdogs can release controls if the Brain connection disappears.

## Install

### Brain/dashboard

```bash
pip install -e ".[brain]"
python -m robloxhive dashboard --host 0.0.0.0 --port 8765
```

### Windows Body

```powershell
pip install -e ".[windows]"
```

### Everything on Windows

```powershell
pip install -e ".[all]"
```

## Current boundary

v0.4.0 connects knowledge to the execution protocol, but generic high-level skills such as navigation, combat, quest handling, collection, and interaction still require their actual perception/controller implementations. The protocol deliberately fails safe until those handlers are registered rather than pretending unsupported behavior works.
