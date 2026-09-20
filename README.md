# RobloxHive

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
