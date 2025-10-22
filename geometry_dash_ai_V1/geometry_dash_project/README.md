# Geometry Dash — Minimal Python Demo

Last updated: 2025-10-22

![CI](https://github.com/Arthur-Brassaert/ai-geometry-dash/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

An educational, compact Geometry Dash-style demo and training project.

This repository contains:

- A simple Pygame-based Geometry Dash game and headless environment for training.
- Training scripts using Stable-Baselines3 (PPO) with VecNormalize, evaluation tooling, and playback utilities.

## Table of Contents

- [Geometry Dash — Minimal Python Demo](#geometry-dash--minimal-python-demo)
  - [Table of Contents](#table-of-contents)
  - [Install](#install)
  - [Run the game](#run-the-game)
    - [Quickstart helper script](#quickstart-helper-script)
  - [Playback (run agent in the game)](#playback-run-agent-in-the-game)
  - [Repository structure](#repository-structure)
  - [TensorBoard](#tensorboard)
  - [Troubleshooting](#troubleshooting)
  - [Development notes](#development-notes)
  - [Interpreting PPO / training terminal output](#interpreting-ppo--training-terminal-output)

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

Use `scripts/quickstart.ps1` to create a venv and install the CPU requirements. The script will:

- create a `.venv` folder
- (optionally) activate the venv in PowerShell
- install packages from `requirements-cpu.txt`

If `--model-path` is omitted the evaluator prefers the archive listed in `trained_models\best_overall.json` (if present), then `trained_models\best_model.zip`, then the newest `.zip`.

## Playback (run agent in the game)

All training artifacts are colocated under `trained_models/` (inside this project directory).

- `<run>.zip` — final model checkpoint (zipped SB3 checkpoint)
- `<run>_vecnormalize.pkl` — VecNormalize stats for consistent inference
- `best_model.zip` — canonical copy of the global-best checkpoint (overwritten on new best)
- `best_YYYYMMDD_HHMMSS.zip` — timestamped archives of checkpoints when a new global best occurred
- `best_overall.json` — metadata about the global best (best_mean_reward, timestamp, and archive/canonical paths when available)
- `last_improvements.log` — plain-text append log of improvement events
- `logs/` — TensorBoard event logs for each run

## Repository structure

A compact view of the main files and folders in this project:

```text
geometry_dash_project/              # Main Python package and scripts
├─ geometry_dash_game.py            # Pygame demo (visual game loop)
├─ ai_training_with_full_game.py    # Training script (PPO, EvalCallback, TimedStop)
├─ ai_playing_full_game.py          # Playback / inference runner
├─ monitor_training.py              # Small utility to inspect logs and artifacts
├─ audio.py                         # Audio helpers and music playback
├─ visuals.py                       # Rendering helpers for the Pygame demo
├─ config.py                        # Game and training configuration constants
├─ settings.json                    # Optional user settings used at runtime
├─ highscore.txt                    # Local highscore persisted by the demo
├─ requirements.txt                 # Python deps (GPU-capable defaults)
├─ requirements-cpu.txt             # Alternative CPU-only deps
├─ scripts/                         # Helper scripts (quickstart, CI helpers)
└─ trained_models/                  # Trained artifacts, logs and metadata
  ├─ best_model.zip                # Canonical global-best model (overwrite on improve)
  ├─ <run>.zip                     # Run-specific SB3 checkpoints
  ├─ <run>_vecnormalize.pkl        # VecNormalize state saved per run
  ├─ best_overall.json             # Metadata about the canonical best
  └─ logs/                         # TensorBoard event files
```

## TensorBoard

Start TensorBoard (from venv) and point it to the logs folder:

```powershell
& .\.venv\Scripts\python.exe -m tensorboard.main --logdir .\trained_models\logs --port 6006
```

Open [http://localhost:6006](http://localhost:6006) in your browser.

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

Typical sequence on an improving run:

1. Training runs and periodically logs timesteps/fps.
2. Evaluation executes at `--eval-freq` and prints `mean_reward` / `std`.
3. When `mean_reward` exceeds the previous best, you see `New best mean reward!` and saving of a timestamped archive + an updated `best_model.zip` and `best_overall.json`.
