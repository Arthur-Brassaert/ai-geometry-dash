# Geometry Dash — Minimal Python Demo

![CI](https://github.com/Arthur-Brassaert/ai-geometry-dash/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

An educational, compact Geometry Dash-style demo and training project.

This repository contains:

- A simple Pygame-based Geometry Dash game and headless environment for training.
- Training scripts using Stable-Baselines3 (PPO) with VecNormalize, evaluation tooling, and playback utilities.

Table of Contents
-----------------

- [Geometry Dash — Minimal Python Demo](#geometry-dash--minimal-python-demo)
  - [Table of Contents](#table-of-contents)
  - [Quick start](#quick-start)
  - [Repository layout](#repository-layout)
  - [Install](#install)
  - [Run the game](#run-the-game)
    - [Quickstart helper script](#quickstart-helper-script)
  - [Training](#training)
  - [Evaluate a saved model](#evaluate-a-saved-model)
  - [Playback (run agent in the game)](#playback-run-agent-in-the-game)
  - [Artifacts \& where to find them](#artifacts--where-to-find-them)
  - [TensorBoard](#tensorboard)
  - [Troubleshooting](#troubleshooting)
  - [Development notes](#development-notes)
  - [Controls](#controls)
  - [Audio setup](#audio-setup)
  - [Settings persistence](#settings-persistence)
  - [Developer notes](#developer-notes)
  - [Training \& Logs](#training--logs)
  - [Training guide and tuning](#training-guide-and-tuning)


## Quick start

PowerShell (from repository root):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```


## Repository layout

- `geometry_dash_game.py` — main Pygame loop / UI (play the game locally)
- `geometry_dash_env.py` — Gym-like environment wrappers & headless sim
- `ai_training_with_full_game.py` — SB3/PPO training harness (vectorized envs)
- `ai_playing_full_game.py` — run an agent inside the real game (playback)
- `evaluate_best_model.py` — quick evaluation harness for saved checkpoints
- `trained_models/` — model zips, VecNormalize pickles, and TensorBoard logs
- `monitor_tensorboard.py` — helper to launch TensorBoard from the venv


## Install

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

2. Install dependencies (GPU or CPU):

- GPU (if you have a compatible NVIDIA GPU and want a CUDA-enabled PyTorch):

```powershell
python -m pip install -r requirements.txt
```

- CPU-only:

```powershell
python -m pip install -r requirements-cpu.txt
```


## Run the game

Run the game locally with rendering:

```powershell
python geometry_dash_project\geometry_dash_game.py
```

Controls: Space (jump), R (restart), M (music pause), arrow keys and Shift for audio volume.


### Quickstart helper script

If you prefer a one-shot script that sets up the venv and installs dependencies, use the included PowerShell helper:

```powershell
.\scripts\quickstart.ps1
```
The script is located at `geometry_dash_project/scripts/quickstart.ps1` and performs venv creation, activation guidance, and dependency installation.


## Training

Use `ai_training_with_full_game.py` to train a PPO agent. The script exposes common options via CLI flags and groups settings in a `TrainingConfig`.

Example (PowerShell):

```powershell
& .\.venv\Scripts\python.exe .\ai_training_with_full_game.py --timesteps 2000000 --n-envs 8 --tb-run-name my_run
```

Stop after a wall-clock duration instead of steps:

```powershell
& .\.venv\Scripts\python.exe .\ai_training_with_full_game.py --train-seconds 3600 --n-envs 8 --tb-run-name my_run
```

Useful CLI flags (high-level):

- `--timesteps` — total timesteps to learn for
- `--train-seconds` — stop after N seconds (wall-clock)
- `--n-envs` — number of parallel envs
- `--eval-freq` — evaluation frequency (timesteps)
- `--eval-episodes` — episodes per evaluation
- `--tb-run-name` — folder/run name used for TensorBoard and saved files
- `--keep-archives` — number of timestamped best archives to retain


## Evaluate a saved model

Evaluate a model zip and (optionally) its VecNormalize stats:

```powershell
& .\.venv\Scripts\python.exe .\evaluate_best_model.py --episodes 20 --model-path trained_models\best_model.zip
```

If `--model-path` is omitted the evaluator prefers the archive listed in `trained_models\best_overall.json` (if present), then `trained_models\best_model.zip`, then the newest `.zip`.


## Playback (run agent in the game)

Run the trained agent in the real game (default: auto-select canonical best):

```powershell
& .\.venv\Scripts\python.exe .\ai_playing_full_game.py --model-path trained_models\best_model.zip --headless False
```

Notes:

- The playback script attempts to load a matching `<model_basename>_vecnormalize.pkl` to apply the same normalization used during training. If it is missing behavior may differ.


## Artifacts & where to find them

All training artifacts are colocated under `trained_models/` (inside this project directory).

- `<run>.zip` — final model checkpoint (zipped SB3 checkpoint)
- `<run>_vecnormalize.pkl` — VecNormalize stats for consistent inference
- `best_model.zip` — canonical copy of the global-best checkpoint (overwritten on new best)
- `best_YYYYMMDD_HHMMSS.zip` — timestamped archives of checkpoints when a new global best occurred
- `best_overall.json` — metadata about the global best (best_mean_reward, timestamp, and archive/canonical paths when available)
- `last_improvements.log` — plain-text append log of improvement events
- `logs/` — TensorBoard event logs for each run


## TensorBoard

Start TensorBoard (from venv) and point it to the logs folder:

```powershell
& .\.venv\Scripts\python.exe -m tensorboard.main --logdir .\trained_models\logs --port 6006
```

Open http://localhost:6006 in your browser.


## Troubleshooting

- TensorBoard shows no runs: ensure `--logdir` points to `trained_models/logs` inside the project and that event files exist.
- Playback looks wrong: confirm a matching `<model>_vecnormalize.pkl` is present and is loaded by the playback script.
- Strange evaluation variance: procedural levels cause high variance; increase `--eval-episodes` (e.g., 50–200) for stable comparisons.


## Development notes

- Training uses `VecNormalize` and saves the normalization state next to the model. Load this during inference for consistent observations.
- `NotifyingEvalCallback` logs improvements and writes `best_overall.json` and `last_improvements.log` when a new global-best is found.
- The training callback also creates a canonical `best_model.zip` and timestamped archives; `--keep-archives` controls retention.


---

If you want me to also: (a) run a lint/format pass for GitHub-flavored Markdown, (b) add badges or CI snippets, or (c) add a short Quickstart script, tell me which and I'll add it.


## Interpreting PPO / training terminal output

When you run `ai_training_with_full_game.py` (PPO via Stable-Baselines3) you will see frequent terminal lines reporting training progress and occasional evaluation summaries. Here's a short guide to the most common outputs and what they mean:

- Timesteps / Progress bars: SB3 prints a progress bar or periodic updates showing total timesteps executed (e.g. `Progress: 1.2e+06/2.0e+06`). This shows how far you are through the requested `--timesteps`.

- FPS / performance: you may see `fps` or `n_updates` metrics. Higher `fps` (frames per second) means faster environment stepping; `n_updates` indicates how many gradient updates the optimizer has performed.

- Eval summaries (printed by the EvalCallback):
  - `Eval num_timesteps=2000, mean_reward=123.45 +/- 67.89` — an evaluation run was executed after the configured `--eval-freq` steps. `mean_reward` is averaged over `--eval-episodes` episodes and the `+/-` value is the standard deviation.
  - The EvalCallback prints `New best mean reward!` when the current `mean_reward` improves on the stored global best. That triggers saving a checkpoint and updating `best_overall.json`.

- Saved artifact messages: when the callback saves a model you'll see messages like `Saving new best to trained_models/best_20251022_212300.zip` or `Saving canonical best_model.zip`. These indicate where checkpoints and VecNormalize pickles are written.

- Timed stops / early stop: if you supplied `--train-seconds`, you'll see a message from the `TimedStopCallback` like `TimedStopCallback: stopping training after ~3600s` and the training loop will gracefully finish and save the final model.

- Warnings & errors: pay attention to Pytorch warnings about device (CPU vs GPU), missing CUDA, or deterministic seed messages. These can affect performance or reproduceability.

- Typical sequence on an improving run:
  1. Training runs and periodically logs timesteps/fps.
 2. Evaluation executes at `--eval-freq` and prints `mean_reward` / `std`.
 3. When `mean_reward` exceeds the previous best, you see `New best mean reward!` and saving of a timestamped archive + an updated `best_model.zip` and `best_overall.json`.

If you want, I can add an example terminal transcript (sanitized) showing a short training run with evals and the callback saving files — say the number of eval episodes you'd like me to include (e.g. 10 or 20). 

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
