Geometry Dash — Minimal Python Demo
==================================

Overview

--------
This is a compact, educational Geometry Dash-style demo written with
Pygame. It's designed to be easy to run locally, easy to read, and easy to
extend. The project includes a simple music player, jump SFX (with a
synthesis fallback), visual effects, and a small set of gameplay parameters
that you can tune in `config.py`.

Repository layout
-----------------

- `geometry_dash_game.py` — main game loop and UI
- `audio.py` — small MusicManager and SFX helper
- `visuals.py` — helpers for rainbow overlays and related rendering
- `config.py` — centralized configuration (display, physics, file paths)
- `requirements.txt` — pinned Python dependencies
- `README.md` — this document

Quick start (Windows PowerShell)
--------------------------------

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the game from the repository root:

```powershell
python geometry_dash_project\geometry_dash_game.py
```

Controls
--------

- Space: hold to jump
- R: restart after game over
- M: toggle music pause/unpause
- Right arrow: next track
- Up / Down: increase/decrease music volume
- Shift+Up / Shift+Down: increase/decrease SFX volume
- Click and drag vertical sliders on the right side to adjust Music and SFX volume

Audio setup
-----------

- Place your music files under `geometry_dash_project/sounds/level songs`.
- Place SFX (optional) under `geometry_dash_project/sounds/sound effects`.
- If no jump SFX file is present, the game will attempt to synthesize a short tone using numpy. If numpy is not installed, jump sound will be silent.

Settings persistence
--------------------

The game saves `music_volume` and `sfx_volume` to `settings.json` in the
project folder when you change them. These values are loaded on startup.

Developer notes
---------------

- Config: `config.py` centralizes tunable gameplay and visual parameters.
- Audio: `audio.py` contains a lightweight `MusicManager` that scans the `DEFAULT_MUSIC_DIR` for tracks, supports play/pause/next/prev, and exposes a `set_volume()` API.
- Visuals: `visuals.py` contains helper functions to pre-render wide rainbow surfaces used for a scrolling overlay effect.
