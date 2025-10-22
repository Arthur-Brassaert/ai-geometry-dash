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
& .\.venv\Scripts\Activate.ps1
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

Training & Logs
---------------

This repository includes training scripts that save trained models and TensorBoard logs to a project-local folder so everything is colocated for easy inspection.

- Trained models directory (models are saved here):

  geometry_dash_project/trained_models/

- TensorBoard event logs (each run gets a timestamped subfolder):

  geometry_dash_project/trained_models/logs/

- To view TensorBoard, run the provided monitor script from the repository root:

  ```powershell
  & .venv\Scripts\python.exe geometry_dash_project\monitor_tensorboard.py
  ```

  Or run tensorboard directly and point it at the canonical logs folder:

  ```powershell
  tensorboard --logdir geometry_dash_project\trained_models\logs
  ```

Installing dependencies

 - GPU (recommended if you have a compatible NVIDIA GPU):

    ```powershell
    pip install -r geometry_dash_project\requirements.txt
    ```

    The `requirements.txt` file includes an extra-index URL so pip can find the prebuilt CUDA +cu124 wheels for `torch`/`torchvision`.

 - CPU-only (use this on machines without CUDA):

    ```powershell
    pip install -r geometry_dash_project\requirements-cpu.txt
    ```

Training guide and tuning
------------------------

Use `ai_training_with_full_game.py` to train an agent using PPO. The script groups important
settings in a `TrainingConfig` and exposes common overrides via CLI flags so you can tweak
behavior without editing code.

Quick examples (PowerShell):

Run a short smoke test (steps):

```powershell
& .venv\Scripts\python.exe geometry_dash_project\ai_training_with_full_game.py --timesteps 20000 --n-envs 4 --model-name smoke_test
```

Run for 1 hour (wall-clock) and stop early if time expires:

```powershell
& .venv\Scripts\python.exe geometry_dash_project\ai_training_with_full_game.py --train-seconds 3600 --n-envs 8 --model-name run_1h
```

Adjust PPO hyperparameters from the CLI. Useful flags include:

- `--policy-net 256 256` (network architecture)
- `--learning-rate 0.0003`
- `--n-steps 2048`
- `--batch-size 128`
- `--n-epochs 10`
- `--gamma 0.99`
- `--gae-lambda 0.95`
- `--clip-range 0.2`
- `--ent-coef 0.01`
- `--vf-coef 0.5`
- `--max-grad-norm 0.5`

Recommended quick presets:

- Fast test: `--timesteps 50_000 --n-envs 4 --policy-net 128 128`

- Medium training (desktop GPU): `--timesteps 1_000_000 --n-envs 8 --policy-net 256 256`

- Long training (multiple hours): `--timesteps 5_000_000 --n-envs 16 --policy-net 256 256`

TensorBoard & logs

- TensorBoard logs are placed under `geometry_dash_project/trained_models/logs/`.

- Use `geometry_dash_project/monitor_tensorboard.py` (recommended) or point `tensorboard` at that folder.

Saving normalization stats

- The script saves VecNormalize statistics next to the model as `<modelname>_vecnormalize.pkl`.

- Load this file when running the playback script so the agent sees the same normalized observations.

```powershell
# Example: run TensorBoard directly (if installed in the active venv)
tensorboard --logdir geometry_dash_project\trained_models\logs
```
