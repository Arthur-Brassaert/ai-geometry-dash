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
  - [PPO Training Terminal Output Samenvatting](#ppo-training-terminal-output-samenvatting)
  - [Tips voor interpretatie](#tips-voor-interpretatie)

## Install

  1 . Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

  2 . Install dependencies (GPU or CPU):

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

## PPO Training Terminal Output Samenvatting

| **Categorie** | **Metric**                 | **Omschrijving**                                                                 | **Voorbeeld / Waarde** |
|---------------|----------------------------|-------------------------------------------------------------------------------|------------------------|
| **Eval**      | `num_timesteps`            | Totaal aantal timesteps tot de evaluatie                                       | 1,520,000              |
|               | `episode_reward`           | Gemiddelde reward per episode + standaarddeviatie                               | 567.01 +/- 427.81      |
|               | `Episode length`           | Gemiddelde lengte van de episodes + std                                         | 182.60 +/- 20.83       |
|               | `mean_ep_length`           | Gemiddelde lengte van episodes (zelfde als boven, vaak afgerond)               | 183                     |
|               | `mean_reward`              | Gemiddelde reward (afgerond)                                                   | 567                     |
| **Rollout**   | `ep_len_mean`              | Gemiddelde episode lengte in de rollout                                        | 185                     |
|               | `ep_rew_mean`              | Gemiddelde episode reward in de rollout                                        | 674                     |
|               | `fps`                      | Frames per second van de omgeving                                              | 1085                    |
|               | `iterations`               | Aantal gradient update iteraties                                               | 93                      |
|               | `time_elapsed`             | Totale tijd in seconden sinds start van training                               | 1403                    |
|               | `total_timesteps`          | Totale timesteps uitgevoerd tot nu                                            | 1,523,712               |
| **Train**     | `approx_kl`                | Geschatte KL-divergentie tussen oude en nieuwe policy                          | 0.0013                  |
|               | `clip_fraction`            | Fractie van updates die de clip limiet raken                                   | 0.00028                 |
|               | `clip_range`               | Clip range parameter (meestal 0.2 bij PPO)                                     | 0.2                     |
|               | `entropy_loss`             | Negatieve entropie van de policy; hogere negatieve waarde → minder exploratie | -0.606                  |
|               | `explained_variance`       | Hoe goed de value function de returns verklaart                                | 0.746                   |
|               | `learning_rate`            | Huidige learning rate van de optimizer                                         | 0.0003                  |
|               | `loss`                     | Totale loss (policy + value)                                                  | 3.73e+03                |
|               | `n_updates`                | Totaal aantal optimizer updates tot nu                                         | 920                     |
|               | `policy_gradient_loss`     | Loss van de policy gradient                                                    | -0.000288               |
|               | `value_loss`               | Loss van de value function                                                     | 6.8e+03                 |

## Tips voor interpretatie

- **Eval reward**: gebruik dit om echte prestaties te monitoren (hogere mean_reward = beter).  
- **Rollout ep_rew_mean**: korte-termijn training feedback. Soms hoger dan eval reward door overfitting op de environment.  
- **approx_kl**: houd < 0.01 voor stabiele updates.  
- **entropy_loss**: negatief → lagere exploratie; te laag kan leiden tot vroegtijdig vastlopen.  
- **explained_variance**: dichtbij 1 → value function voorspelt goed; dichtbij 0 → slecht.  
