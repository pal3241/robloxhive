# RobloxHive

Distributed autonomous Roblox agent: **Brain Node** for reasoning/memory and a Windows-only **Body Node** for Roblox control/perception.

## v0.6.0 — Fused Perception

RobloxHive no longer depends only on screenshot templates. The Windows Body now supports an ordered perception stack:

```text
follow_player target
        ↓
PlayerTracker
        ↓
ONNX object detector
        ↓
OCR / UI text
        ↓
Template fallback
```

For ordinary object targets the player tracker is skipped, so `navigate shop` cannot accidentally lock onto an avatar.

A strong ONNX/OCR hit exits the fusion loop early to reduce CPU cost while Roblox is running.

### Perception sources

**ONNX object detector**
- YOLOv8-style ONNX output.
- CPUExecutionProvider by default.
- Custom game-specific class labels from `labels.json` or `labels.txt`.
- Used for objects, items, avatars, machines, doors, enemies, etc.

**Player tracker**
- Lightweight IoU tracking on top of detector results.
- Stable `track_id` across nearby frames.
- Activated by `player:<target>`, which is what `follow_player` now requests.
- The current visual tracker follows a detected avatar; exact username identity still needs nameplate/OCR association when multiple players look similar.

**UI/OCR**
- Optional Tesseract-backed UI text recognition through `pytesseract`.
- Useful for buttons, labels, counters and visible nameplates.
- Automatically disabled if the Tesseract executable is unavailable.

**Template fallback**
- Existing per-game OpenCV templates remain available.
- Only used when stronger perception backends fail to identify the target.

## Model layout

By default the Body looks for:

```text
data/models/<game_id>/
├── detector.onnx
├── labels.json
└── labels.txt
```

Only one labels file is necessary.

Example `labels.txt`:

```text
player
fuel
shop
door
enemy
train
station
```

You can also point to another model manually with `--model` and `--labels`.

## Perception Dashboard

The Dashboard now has:

```text
Overview
Instances
Belajar
Agent
Skills
Perception   ← v0.6
Memory
```

The **Perception** tab shows:

- connected Windows Body
- active perception adapters
- ONNX status
- OCR status
- model path
- last perception source
- frame size
- live target probe results

Example probe:

```text
Target: fuel

FOUND
source: onnx
confidence: 91%
box: 421,188 103×77
```

For player tracking:

```text
player:Fahri
```

The returned detection includes a track ID when the player detector/tracker is active.

## Windows Body

Install Body dependencies:

```powershell
pip install -e ".[windows]"
```

Tesseract OCR is optional. If it is not installed system-wide, RobloxHive simply reports OCR as unavailable.

Find Roblox windows:

```powershell
python -m robloxhive body --list-windows
```

Run the selected bot window:

```powershell
python -m robloxhive body ^
  --brain-url http://192.168.1.50:8765 ^
  --agent-id agent-01 ^
  --game-id 123456789 ^
  --pid 24680
```

Explicit model:

```powershell
python -m robloxhive body ^
  --brain-url http://192.168.1.50:8765 ^
  --agent-id agent-01 ^
  --game-id 123456789 ^
  --pid 24680 ^
  --model data\models\123456789\detector.onnx ^
  --labels data\models\123456789\labels.txt
```

Disable OCR:

```powershell
python -m robloxhive body --pid 24680 --game-id 123456789 --no-ocr
```

## Implemented generic skills

The following skills use the fused perception layer:

- `navigate`
- `collect`
- `interact`
- `follow_player`

```text
Knowledge
   ↓
Goal Manager
   ↓
Planner
   ↓
semantic skill
   ↓
Fused Perception
   ↓
HWND-scoped input
   ↓
visual verification
   ↓
ActionResult
   ↓
retry / next step / memory
```

### navigate

Finds the target through fused perception, steers toward its screen center, approaches until its apparent size reaches the configured near threshold, then returns evidence.

### collect

Approaches the target, sends interaction, and requires a visible change/disappearance before it is considered verified.

### interact

Approaches the target and sends the interaction key. A targeted interaction without a measurable visual change is reported as `INTERACTION_NOT_VERIFIED`.

### follow_player

Requests `player:<name>`, uses the player tracking route and maintains an apparent-size distance band.

## Brain on another device

The heavy Brain can stay on Android/another PC/server:

```bash
pip install -e ".[brain]"

export ROBLOXHIVE_LLM_MODEL="qwen2.5:3b"
export ROBLOXHIVE_OLLAMA_URL="http://127.0.0.1:11434"

python -m robloxhive dashboard --host 0.0.0.0 --port 8765
```

ONNX/OpenCV/Tesseract Python packages are **not** part of the Brain dependency set.

## Learning and per-game memory

The **Belajar** tab researches a game from the internet, including:

- beginner tutorials
- walkthrough/progression
- tips and strategy
- items/upgrades
- win conditions/endgame

Research is synthesized into:

```text
data/games/<game_id>/
├── profile.json
├── knowledge.json
└── research/
    └── research-<timestamp>.json
```

Knowledge retains confidence, source references, and gameplay verification state.

## Window safety

- Bot control is bound to an explicit Roblox PID/HWND.
- Human/player instances can be marked `PROTECTED`.
- RobloxHive refuses to guess a bot process when multiple windows exist.
- No automatic reassignment after a bot process disappears.
- Perception capture stays scoped to the assigned bot HWND.
- Unknown skills fail closed.

## Current technical boundary

The new detector/tracker greatly reduces dependence on templates, but RobloxHive still needs a compatible object-detection model for game-specific visual classes. The repository intentionally does not ship a large pretrained game model.

Navigation is still **visual steering**, not full 3D SLAM/A* pathfinding yet. The next navigation layer can build a local obstacle map and waypoints on top of these detections without changing the Goal/Planner/Skill protocol.
