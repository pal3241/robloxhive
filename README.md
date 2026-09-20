# RobloxHive

Modular autonomous Roblox agent architecture.

RobloxHive separates the **Brain Node** (LLM, goals, planner, internet research, per-game memory) from the Windows-only **Body Node** (Roblox, pyrobloxbot, PID/HWND ownership, local failsafes). Roblox chat is not used as a control channel.

## v0.2.0 foundation

- Windows Roblox window discovery using PID + HWND.
- Explicit PLAYER vs BOT assignment in the dashboard.
- Player instances are PROTECTED.
- No automatic reassignment after a bot process disappears.
- Learning jobs that research a Roblox game from the internet.
- Per-game memory stored under the Roblox Game/Universe ID.
- Dashboard tabs for Instances, Belajar, and Memory.
- Research keeps URLs, search query, extraction status, source quality, and verification state.
- Internet claims begin as unverified knowledge; gameplay can verify or reject them later.

## Internet learning

The **Belajar** tab creates a research job with multiple search intents:

1. beginner tutorial
2. full walkthrough / start-to-finish
3. progression / wiki
4. tips, tricks and strategy
5. important items, weapons, classes and upgrades
6. ending / completion / how to win
7. the user's custom learning objective

RobloxHive uses DDGS as the initial metasearch/extraction backend. Search results and extractable page content are saved to:

```text
data/games/<game_id>/
├── profile.json
└── research/
    └── research-<timestamp>.json
```

The raw corpus is kept separate from compact game memory so a local/remote LLM can later synthesize deeper knowledge without changing the research pipeline.

## Install

### Brain/dashboard (Windows, Linux, Android/Termux)

```bash
pip install -e ".[brain]"
python -m robloxhive dashboard --host 0.0.0.0 --port 8765
```

Open `http://127.0.0.1:8765` locally. When the Brain runs on another device, use that device's LAN address.

### Windows Body dependencies

```powershell
pip install -e ".[windows]"
```

Or install everything on Windows:

```powershell
pip install -e ".[all]"
```

## Architecture

```text
Brain Node (HP / PC / server)
├── Goal Manager
├── Planner
├── Internet Research
├── Game Memory
└── LLM / knowledge synthesis (next layer)
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

The Windows body is intentionally kept separate from the heavy Brain so Qwen/LLM, research and long-term memory can move to a phone, another PC, or a server without rewriting Roblox control code.

## Safety architecture

- Every bot must be bound to an explicit Roblox PID/HWND.
- Human-player windows can be marked PROTECTED.
- The body does not guess a new bot window when the assigned process disappears.
- Local watchdogs release controls if the Brain disconnects.
- Actions return structured results and are intended to be verified before higher-level goals advance.

## Notes

The v0.2.0 window manager identifies and protects instances, but reliable background input to a non-focused Roblox client still needs platform testing. Roblox/Windows may reject some background input techniques, so RobloxHive does not claim simultaneous independent input is solved yet.
