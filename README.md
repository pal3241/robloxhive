# RobloxHive

Distributed autonomous Roblox agent with a remote-capable Brain Node and a Windows Body Node.

## v0.8.0 — First Game Adapter: Murder Mystery 2

The first specialized RobloxHive game adapter is **Murder Mystery 2** (official place ID `142823291`).

The adapter is deliberately kept outside the generic core:

```text
robloxhive/
├── brain/
├── body/
├── games/
│   └── murder_mystery_2/
│       ├── models.py
│       ├── role_detector.py
│       ├── scene.py
│       ├── threat.py
│       ├── survival.py
│       ├── combat.py
│       └── autonomy.py
└── ...
```

That keeps role/combat rules isolated while reusing the universal perception, navigation, planner, memory, and Windows-instance systems.

## MM2 runtime loop

MM2 behavior runs locally on the Windows Body because survival, dodging, aiming, and attack timing need lower latency than an LLM loop.

```text
Bot window
   ↓
Fused Perception
   ↓
MM2 Scene Reader
   ↓
Role Detector + Threat Model
   ↓
SURVIVAL FIRST
   ↓
Role Policy
   ├── Innocent → survive / observe
   ├── Sheriff  → survive / confirmed shot
   ├── Hero     → survive / confirmed shot
   └── Murderer → survive / target / melee or throw
   ↓
HWND-scoped input
```

The Brain/Qwen can still decide strategy, but the realtime reflex layer stays local.

## Role detection

Roles supported:

- `innocent`
- `sheriff`
- `hero`
- `murderer`
- `unknown`

Role detection first uses OCR from the role reveal. A role must be observed consistently before offensive behavior becomes active.

Once stable, the role is **latched for the round** and is cleared on lobby/round-end detection. This prevents the bot from forgetting its role after the reveal text disappears.

There is also a self-weapon fallback:

- own visible knife → likely Murderer
- own visible gun → Sheriff/Hero-style gun policy

The dashboard has a manual role override for debugging, but default operation is `AUTO`.

## Always-on survival

Survival is evaluated before offense.

### Innocent / Sheriff / Hero / Unknown

If a high-confidence visible knife holder approaches too closely:

```text
detect murderer
   ↓
measure apparent distance
   ↓
strafe away
   ↓
very close?
   └── back + jump
```

### Murderer

The Murderer also protects itself. A visible gun holder/Sheriff can trigger evasive movement before the next attack decision.

Survival actions are counted in the MM2 dashboard.

## Sheriff / Hero combat safety

Sheriff and Hero are intentionally conservative.

The gun is only fired when:

1. a target has strong Murderer evidence,
2. Murderer confidence is at least the configured threshold (default 93%),
3. another visible avatar does not overlap the planned aim point,
4. the local shot cooldown is ready.

```text
suspect
  ↓
confirmed knife evidence?
  ↓
confidence >= 93%?
  ↓
crowd clear?
  ↓
equip slot 1
  ↓
aim upper torso
  ↓
left click
```

A low-confidence suspect produces `MURDERER_CONFIDENCE_TOO_LOW`, not a speculative shot.

A crowded aim produces `FRIENDLY_FIRE_RISK`.

## Murderer combat

The Murderer prioritizes a visible gun holder first, then another visible player.

Attack mode is selected by apparent distance:

```text
target close
   → equip knife
   → aim
   → left click melee

target medium/far
   → equip knife
   → aim
   → right click throw

target too far
   → approach / steer
   → re-evaluate next tick
```

Melee and throwing each have local cooldowns to prevent input spam.

The likely self avatar is filtered so the Murderer does not choose its own third-person avatar as a target.

## Threat model

MM2 threat tracking associates detected weapons with nearby tracked avatars.

Recommended detector labels are included at:

```text
config/mm2-labels.txt
```

Contents:

```text
player
knife
gun
dropped_gun
dead_player
```

Held and dropped guns are treated separately. A player standing near a dropped Sheriff gun is **not** automatically marked as Sheriff.

## Dashboard

Dashboard tabs now include:

```text
Overview
Instances
Belajar
Agent
Skills
Perception
Navigation
MM2        ← v0.8
Memory
```

The MM2 tab shows:

- Body online/offline state
- autonomy enabled/disabled
- detected role
- role confidence
- role override
- current mode
- last action / reason
- confirmed Murderer track and confidence
- Sheriff/gun-holder track
- current attack target
- survival move count
- Sheriff/Hero shots
- knife melee count
- knife throw count
- visible players / knives / guns / bodies

Controls:

- Enable autonomy
- Disable autonomy
- Role override: Auto / Innocent / Sheriff / Hero / Murderer

## Perception requirement

For full MM2 behavior, the Windows Body needs a detector that can recognize at least:

```text
player
knife
gun
```

Useful additional labels:

```text
dropped_gun
dead_player
```

A compatible ONNX model can be placed at:

```text
data/models/142823291/
├── detector.onnx
└── labels.txt
```

RobloxHive still keeps OCR and template perception as fallbacks, but multi-player combat works best with an object detector because the scene reader needs several avatars simultaneously.

## pyrobloxbot and window ownership

`pyrobloxbot` remains a Windows dependency and RobloxHive is compatible with its window-targeting model.

However, when the human and the bot are both playing on the same laptop, RobloxHive's realtime MM2 layer defaults to the explicit PID/HWND-scoped Body input abstraction. This avoids intentionally sending global keyboard/mouse actions to whichever window happens to be focused.

The bot window is still selected explicitly with:

```powershell
python -m robloxhive body --list-windows
```

then:

```powershell
python -m robloxhive body ^
  --brain-url http://192.168.1.50:8765 ^
  --agent-id agent-01 ^
  --game-id 142823291 ^
  --pid 24680 ^
  --model data\models\142823291\detector.onnx ^
  --labels config\mm2-labels.txt
```

When `--game-id 142823291` is used, the MM2 adapter is attached automatically and autonomy starts enabled.

## Universal systems reused by MM2

MM2 uses the existing RobloxHive stack:

- explicit PID/HWND ownership
- protected human-player window
- fused ONNX/OCR/template perception
- player tracking
- local occupancy map
- A* navigation
- stuck/recovery handling
- per-game memory
- Brain/Body separation
- dashboard diagnostics

MM2-specific rules do not leak into generic navigation or planning.

## Current boundary

v0.8 provides the **role/survival/combat state machine and execution path**, but real match quality depends heavily on the MM2 detector model.

In particular, robust identification of multiple moving avatars, held knives, held guns, and dropped weapons requires representative training images from actual MM2 rounds.

Future MM2 work can add:

- nameplate-to-track association
- witnessed-kill evidence
- corpse-event reasoning
- dropped-gun Hero pickup
- projectile/throw lead prediction
- persistent map knowledge per MM2 map
- learned dodge timing
- combat outcome verification
