# RobloxHive

Distributed autonomous Roblox agent with a remote-capable Brain Node and a Windows Body Node.

## v1.0.0 — Generic Dashboard Sessions, Multi-Instance Control, and Reliable Input

v1.0 makes the Windows Body game-agnostic at startup. A Body can start with `game_id=0`, report every Roblox window it can see, be rebound to a selected PID from the dashboard, and join a Place ID chosen at runtime.

Highlights:

- Windows Body reports Roblox PID/HWND inventory to a remote Brain, so the Instances tab works even when the dashboard runs on Android/Linux.
- Dashboard assignment actually rebinds the selected Body to the selected Roblox PID.
- Human/player instances can be marked protected; changing a bound instance to player/unassigned disarms that Body.
- Body commands are routed by `agent_id`, preventing one multi-account Body from consuming another agent's command.
- Generic **Join Game** accepts any Roblox Place ID from the dashboard.
- Joining a new Place ID rebuilds the perception/navigation/game-adapter stack for that game without restarting RobloxHive.
- New guarded foreground SendInput controller is the default because Roblox often ignores background `WM_KEY*` messages.
- Dashboard has a W/A/S/D, Jump, Interact, and Release Keys control pad for verifying real control before starting autonomous behavior.
- Existing MM2 specialization remains available automatically when its Place ID is selected.

### Recommended v1 Body startup

Start Roblox/ExoPanda first, then list windows:

```powershell
python -m robloxhive body --list-windows
```

Start a generic Body on one Roblox instance:

```powershell
python -m robloxhive body `
  --brain-url http://192.168.1.6:8765 `
  --agent-id agent-01 `
  --pid 16800
```

Then open **Dashboard → Instances**:

1. choose the Windows Body,
2. mark your personal Roblox window **THIS IS ME**,
3. assign the bot window to the selected Body,
4. test W/A/S/D,
5. enter any Place ID and press **Join Game**.

The default input mode is `auto`, which currently uses guarded foreground SendInput. Use `--input-mode message` only for experimental background control.

For Termux/Android, install the Brain without native search dependencies:

```bash
pip install -e ".[brain]"
```

Internet research is optional and can be added separately on platforms where `ddgs` installs cleanly:

```bash
pip install -e ".[research]"
```


## v0.9.0 — MM2 Dataset, Witnessed-Kill Reasoning, and Predictive Aim

Murder Mystery 2 is the first specialized RobloxHive game adapter. v0.9 adds the training loop required to build a dedicated visual detector and improves combat reasoning for moving targets.

## MM2 realtime loop

```text
Roblox bot HWND
      ↓
Fused perception
      ↓
Tracking + screen velocity
      ↓
MM2 Scene Reader
      ↓
Role Detector
      ↓
Threat Model
      ├── visible knife/gun evidence
      └── witnessed-kill evidence
      ↓
SURVIVAL FIRST
      ↓
Role Policy
 ┌────────────┼─────────────┐
 ▼            ▼             ▼
Innocent   Sheriff/Hero   Murderer
survive    lead + fire    melee/lead throw
```

The realtime MM2 reflex loop stays on the Windows Body. The Brain/Qwen can remain on another device.

## Predictive aim

Tracked detections now maintain an exponentially smoothed screen-space velocity:

```text
velocity_px_s = [vx, vy]
speed_px_s
track_age
```

Sheriff/Hero gun aim and Murderer knife throws use a bounded lead point:

```text
predicted_x = current_x + vx × lead_time
predicted_y = current_y + vy × lead_time
```

Lead is clamped so a noisy tracker cannot aim far outside the target region.

Default behavior:

- Sheriff lead horizon: 0.08 s
- Knife throw lead horizon: 0.20 s
- Maximum screen lead: 140 px

Close-range knife melee still aims directly at the current upper-torso position.

## Witnessed-kill reasoning

Visible knife ownership remains the strongest Murderer signal, but v0.9 can also use new corpse events as supporting evidence.

```text
previous frame: no corpse
current frame : new dead_player detection
             ↓
find nearby tracked players
             ↓
knife holder nearby?
     ├── yes → very strong Murderer evidence
     └── no  → only reinforce an already suspicious player
```

Proximity alone never turns an innocent player into a confirmed Murderer.

The first corpse snapshot after Body startup is treated as a baseline so joining mid-round does not create fake witnessed-kill events.

Recent kill events and threat evidence are visible in MM2 diagnostics.

## MM2 dataset pipeline

v0.9 adds a full dataset workflow.

```text
Bot window
   ↓
Capture Frame
   ↓
data/datasets/mm2/raw
   ↓
detector proposals
   ↓
human review
   ↓
APPROVED boxes only
   ↓
YOLO train/val export
   ↓
Ultralytics fine-tune
   ↓
ONNX export
   ↓
data/models/142823291/detector.onnx
```

### Dataset classes

Default MM2 classes:

```text
player
knife
gun
dropped_gun
dead_player
```

The label template is also stored at:

```text
config/mm2-labels.txt
```

## Dataset storage

Raw captures:

```text
data/datasets/mm2/
└── raw/
    ├── images/
    │   └── <sample-id>.png
    └── annotations/
        └── <sample-id>.json
```

Each annotation contains:

- image dimensions
- class
- bounding box
- detector confidence
- detection source
- approved/rejected state
- optional note

Detector proposals are **not** automatically trusted as training ground truth. Only approved boxes are exported.

## MM2 Dashboard dataset workflow

The MM2 dashboard now contains:

- Capture Frame
- Refresh Samples
- screenshot preview
- detector proposals drawn over the screenshot
- a manual drag-box annotator
- class selector: player / knife / gun / dropped_gun / dead_player
- per-box approved/pending state
- Save Review
- Approve All Boxes
- Reject All Boxes
- Export Approved → YOLO
- dataset sample/box counters

This means the first dataset can be created **without any detector model installed**. Capture a frame, drag boxes around visible objects, assign the class, and save the review. When a model already exists, its detections appear as proposals that can be corrected rather than trusted automatically.

Recommended workflow:

1. Join MM2 with the selected bot window.
2. Open the **MM2** dashboard tab.
3. Capture representative situations:
   - several avatar appearances
   - knife equipped
   - gun equipped
   - dropped gun
   - dead player/body
   - different maps, rooms, distances, lighting, skins and camera angles
4. Open each sample.
5. Drag missing boxes manually and select the correct class.
6. Toggle or reject wrong proposals.
7. Save the review.
8. Export the reviewed dataset.

## YOLO export

Approved samples are exported to:

```text
data/datasets/mm2/yolo/
├── dataset.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Unknown/unapproved boxes are excluded.

The export uses a deterministic train/validation split.

## Training the MM2 detector

Install training dependencies:

```powershell
pip install -e ".[training]"
```

Then train:

```powershell
python -m robloxhive mm2-train
```

Defaults:

```text
dataset    data/datasets/mm2/yolo/dataset.yaml
base model yolov8n.pt
epochs     50
imgsz      640
batch      8
output     runs/mm2
```

Custom example:

```powershell
python -m robloxhive mm2-train ^
  --dataset data\datasets\mm2\yolo\dataset.yaml ^
  --base-model yolov8n.pt ^
  --epochs 80 ^
  --imgsz 640 ^
  --batch 8 ^
  --device cpu
```

The best checkpoint is exported to raw-output ONNX without embedded NMS so RobloxHive's lightweight ONNX Runtime parser can consume it.

Final runtime files:

```text
data/models/142823291/
├── detector.onnx
├── labels.txt
└── training.json
```

## Running the MM2 Body

Install Windows runtime dependencies:

```powershell
pip install -e ".[windows]"
```

Find Roblox windows:

```powershell
python -m robloxhive body --list-windows
```

Start the bot instance explicitly:

```powershell
python -m robloxhive body ^
  --brain-url http://192.168.1.50:8765 ^
  --agent-id agent-01 ^
  --game-id 142823291 ^
  --pid 24680 ^
  --model data\models\142823291\detector.onnx ^
  --labels data\models\142823291\labels.txt
```

When game ID `142823291` is selected, the MM2 adapter attaches automatically.

## Role behavior

### Innocent

- always-on survival
- evade confirmed knife holder
- continue observing and building threat evidence
- never attack from an unknown role

### Sheriff / Hero

- survival is evaluated first
- only fire at high-confidence Murderer
- default fire threshold: 93%
- do not fire when another avatar overlaps the predicted aim point
- use tracked velocity to lead moving targets
- gun cooldown prevents input spam

### Murderer

- survival can evade a visible gun holder
- prioritize visible gun holder
- close target → melee knife
- medium/far target → predictive knife throw
- too-far target → approach and re-evaluate
- self avatar is excluded from targets

## Role safety

Role OCR must stabilize before offensive behavior starts.

Once stable, the role is latched until lobby/round-end detection. If OCR is unavailable, the bot's own visible weapon can provide a fallback role hint.

Manual dashboard role override is available for debugging only.

## Architecture

```text
Phone / remote Brain
├── Internet learning
├── Qwen / Ollama
├── per-game memory
├── Goal Manager
└── Planner
          │
          │ LAN API
          ▼
Windows Body
├── explicit PID/HWND
├── Roblox capture
├── ONNX detector
├── OCR
├── player tracking
├── velocity estimation
├── local occupancy map + A*
├── MM2 role detector
├── witnessed-kill reasoner
├── survival
├── predictive combat
└── dataset recorder
```

## Current boundary

The code now contains the full MM2 collection/training/export/runtime pipeline, but the repository does **not** contain a trained MM2 detector yet because no reviewed MM2 screenshots have been supplied.

The next practical step is to collect and review real MM2 frames. Once enough varied examples exist, `mm2-train` can create the first dedicated ONNX model and the runtime can begin real-match detector tuning.
