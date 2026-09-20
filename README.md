# RobloxHive

Modular autonomous Roblox agent architecture with a split **Brain Node** and Windows **Body Node**.

## v0.3.0 — Internet Learning + Knowledge Synthesis

RobloxHive can now turn web research into structured per-game knowledge instead of storing only raw snippets.

The learning pipeline is:

```text
Dashboard: "Learn Dead Rails"
        ↓
InternetResearcher
        ↓
beginner guides / walkthroughs / progression / tips / items / endings
        ↓
raw research corpus + source URLs
        ↓
KnowledgeSynthesizer
        ├── Ollama/Qwen when configured
        └── deterministic fallback when no LLM is available
        ↓
knowledge.json
        ↓
Goal Manager / Planner (next consumer layer)
```

Knowledge is organized into:

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
- conflicts between sources

Every synthesized entry keeps confidence, source IDs, and `verified_in_game=false` until gameplay confirms it.

## Per-game memory

Memory is isolated by Roblox Game/Universe ID:

```text
data/games/<game_id>/
├── profile.json
├── knowledge.json
└── research/
    └── research-<timestamp>.json
```

`research/*.json` keeps the source corpus. `knowledge.json` is the compact operational knowledge used by the agent. Keeping them separate allows RobloxHive to re-synthesize old research with a better LLM later.

## Use Qwen / Ollama on another device

The Windows machine does not need to run the heavy model.

Run Ollama/Qwen on the Brain device (for example an Android phone, another PC, or server), then configure RobloxHive:

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

If `ROBLOXHIVE_LLM_MODEL` is not set, learning still works with the lightweight heuristic synthesizer.

For smaller devices, only a bounded subset of each researched page is sent to the LLM. Raw full research remains stored on disk.

## Dashboard

Tabs:

- **Overview** — Brain/Body status.
- **Instances** — discover Roblox PID/HWND and explicitly assign PLAYER or BOT-01.
- **Belajar** — launch internet learning jobs and watch search/extraction/synthesis progress.
- **Memory** — inspect per-game knowledge, confidence, research source count, and raw debug profile.

Roblox chat is not used as a command channel.

## Windows instance safety

- Every controlled Roblox instance is bound to PID + HWND.
- Human/player windows can be marked `PROTECTED`.
- The Body does not guess a replacement bot window after a process disappears.
- Automatic reassignment is intentionally avoided.
- Local failsafes release input when control is lost.

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

## Architecture

```text
Brain Node — HP / PC / server
├── Internet Research
├── Knowledge Synthesis
├── Per-game Memory
├── Goal Manager
├── Planner
└── LLM / Qwen
          │
          │ API / WebSocket
          ▼
Windows Body
├── Roblox Client
├── Instance Discovery
├── PID/HWND Ownership
├── PyRobloxBot backend
├── Capture / Perception
└── Local Failsafes
```

The remaining major integration step is to feed this synthesized knowledge into autonomous planning and then verify learned claims against live gameplay/perception.
