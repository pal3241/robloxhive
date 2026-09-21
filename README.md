# RobloxHive 1.0.0

RobloxHive is a **single-bot, local-first Roblox automation research project**.

The normal setup now uses one Windows laptop only:

- Windows runs Roblox, capture, OCR/detection, navigation, controller, dashboard, semantic map and long-term memory.
- Ollama can run on the same laptop or on another reachable device such as an Android phone.
- There is no required Brain/Body split for the main v1.0 workflow.
- The old distributed commands remain for compatibility, but python -m robloxhive run is the recommended entry point.

## Architecture

Python handles orchestration, perception, memory, semantic skills and Ollama integration.

The dashboard is HTML/CSS/JavaScript with a classic dark-blue sidebar.

An optional zero-dependency Rust library accelerates small hot math operations such as predictive aim. If the DLL is not built, RobloxHive automatically uses the Python fallback.

~~~text
Roblox HWND
   |
   +-- window capture
   +-- OCR / ONNX / templates
   +-- player + username tracking
   +-- local navigation
   +-- semantic world model
   |
   v
SingleBotRuntime
   +-- CognitiveMemory (SQLite)
   +-- SemanticMap
   +-- old Internet Learning knowledge
   +-- action/result verification
   |
   v
Ollama Actor
   |
   +-- observe / wait
   +-- navigate / explore
   +-- collect / interact / click_ui
   +-- follow_player
   +-- aim / combat
   +-- press_key
~~~

Ollama is not only used to summarize text. It receives the current world state, user goal, relevant memory, map knowledge, recent action outcomes and available skill contracts, then selects the next semantic action.

## Install on Windows

Python 3.11 is recommended.

~~~powershell
git clone https://github.com/pal3241/robloxhive.git
cd robloxhive

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e ".[single]"
~~~

Tesseract is optional but recommended for UI text and username/nameplate recognition.

## Ollama on the same laptop

Start Ollama normally and make sure the model exists:

~~~powershell
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
~~~

RobloxHive defaults to:

~~~text
http://127.0.0.1:11434
qwen2.5:3b
~~~

A small text model is enough for the scenario-level loop because realtime aiming/navigation stays deterministic on Windows.

For a vision-capable Ollama model, start RobloxHive with --vision-llm or enable Vision in Settings. Screenshots are resized and JPEG-compressed before being sent.

## Ollama on a phone or another PC

The Windows bot still runs completely on the laptop. Only Ollama is remote.

Make Ollama reachable on the LAN, then set its URL in Dashboard -> Settings, for example:

~~~text
http://192.168.1.6:11434
~~~

Or start with:

~~~powershell
python -m robloxhive run --pid 16800 --ollama-url http://192.168.1.6:11434
~~~

## Run

Open the Roblox bot account first.

Find the Roblox PID:

~~~powershell
python -m robloxhive run --list-windows
~~~

Example:

~~~text
PID=16800 HWND=48301638 TITLE='Robloxianp2v4f2p0'
~~~

Start the bot:

~~~powershell
python -m robloxhive run --pid 16800
~~~

Dashboard:

~~~text
http://127.0.0.1:8765
~~~

If there is exactly one Roblox window, --pid can be omitted.

## Dashboard

The v1 dashboard is intentionally simple and single-bot oriented:

~~~text
Overview
Agent
Game & Controls
Vision & UI
Memory
Map
Learning
Settings
~~~

There are no Body selectors or multi-agent heartbeat lists.

### Agent

Give the AI a high-level custom scenario, for example:

~~~text
Cari objective utama, pahami UI yang ada, coba selesaikan objective,
dan belajar dari kegagalan. Jangan mengulang aksi gagal tanpa perubahan.
~~~

Ollama receives the current scene and decides a semantic action. Every action result is stored and fed back into later decisions.

### Follow Player

Use the exact Roblox username.

RobloxHive prefers:

~~~text
player detector + tracker
        +
exact nameplate OCR
~~~

If a game-specific player detector is unavailable, it can fall back to exact OCR nameplate following. It does not silently follow a random avatar if the requested username cannot be verified.

### UI interaction

The generic UI skill can:

~~~text
find visible OCR text
        ->
click its center
        ->
observe the next state
~~~

This lets Ollama handle custom menus and prompts instead of relying only on pre-written per-game code.

### FPS / shooting games

Generic combat includes:

- tracked target velocity
- predictive lead
- upper-torso aiming
- bounded lead correction
- optional weapon hotkey
- click/fire action verification on the next AI cycle

MM2 keeps its specialized role/threat logic. Generic games can use the general aim and combat skills.

## Memory

Main memory database:

~~~text
data/memory/robloxhive.db
~~~

Memory types:

~~~text
episodic
semantic
procedural
social
team
enemy
role
ui
map
strategy
failure
goal
action
research
~~~

RobloxHive also stores relations such as:

~~~text
player -> member_of_team -> blue
~~~

Failures are deliberately remembered so the LLM sees what did not work before choosing another action.

## Semantic map

The high-level map is not only a temporary screen grid.

It remembers landmarks, routes, travel time, route success rate, danger, location value, visits and useful interactions.

This is a topological/semantic map, so it can still learn games where exact 3D coordinates are unavailable.

The existing local occupancy-grid + A* navigation remains available for immediate obstacle handling.

## Internet Learning

The old research pipeline remains available in the Learning tab.

Research is saved under:

~~~text
data/games/<game_id>/
~~~

When knowledge.json changes, the single-bot runtime imports useful knowledge into CognitiveMemory automatically:

- objectives -> semantic memory
- progression/endgame -> procedural memory
- enemies -> enemy memory
- locations -> map memory
- strategies -> strategy memory
- common mistakes -> failure memory

Internet knowledge is context, not ground truth. Current gameplay observations and verified action outcomes take priority.

Install web research support separately:

~~~powershell
pip install -e ".[research]"
~~~

## Optional Rust accelerator

Rust is not required.

Build it with:

~~~powershell
cd native\robloxhive-native
cargo build --release
cd ..\..
~~~

RobloxHive automatically detects:

~~~text
native/robloxhive-native/target/release/robloxhive_native.dll
~~~

If it is absent, the Python fallback is used.

## Useful launch options

Local Ollama:

~~~powershell
python -m robloxhive run --pid 16800 --ollama-model qwen2.5:3b
~~~

Remote Ollama:

~~~powershell
python -m robloxhive run --pid 16800 --ollama-url http://192.168.1.6:11434 --ollama-model qwen2.5:3b
~~~

Vision-capable Ollama model:

~~~powershell
python -m robloxhive run --pid 16800 --ollama-model YOUR_VISION_MODEL --vision-llm
~~~

Explicit detector:

~~~powershell
python -m robloxhive run --pid 16800 --detector-model data\models\GAME_ID\detector.onnx --labels data\models\GAME_ID\labels.txt
~~~

Background input is available but some Roblox builds may ignore window messages. The default reliable input path validates the assigned HWND before sending input.

## Legacy distributed mode

The older Brain/Body commands remain for compatibility:

~~~text
python -m robloxhive dashboard
python -m robloxhive body
~~~

New development should target single mode unless there is a specific reason to distribute the runtime.

## Safety model

The bot is bound to one explicit Roblox HWND/PID.

RobloxHive does not intentionally fall back to desktop-wide capture or guess a different Roblox window if the assigned one disappears.

The AI can choose only registered semantic skills; arbitrary shell commands or arbitrary code produced by Ollama are never executed.
