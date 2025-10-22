# AI Geometry Dash

![CI](https://github.com/Arthur-Brassaert/ai-geometry-dash/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

AI Geometry Dash is a compact educational project containing:

- A small Pygame-based Geometry Dash demo (playable locally)
- A headless Gym-like environment for training agents
- Training harnesses using Stable-Baselines3 (PPO) and experimental DQN trainers

See the project README for full usage, training and evaluation details:

👉 [geometry_dash_ai_v1/geometry_dash_project/README](geometry_dash_ai_v1/geometry_dash_project/README.md)

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

1. Install dependencies (inside the venv):

```powershell
pip install -r geometry_dash_ai_v1/geometry_dash_project/requirements.txt

```

## Repository structure

```text
ai-geometry-dash/
├─ geometry_dash_ai_v1/
│  └─ geometry_dash_project/
│     ├─ ai_training_with_full_game.py
│     ├─ ai_playing_full_game.py
│     ├─ evaluate_best_model.py
│     ├─ geometry_dash_game.py
│     ├─ geometry_dash_env.py
│     ├─ monitor_tensorboard.py
│     ├─ trained_models/
│     │  ├─ best_model.zip
│     │  ├─ best_overall.json
│     │  └─ logs/
│     ├─ scripts/
│     │  └─ quickstart.ps1
│     └─ README.md
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ .venv/
└─ README.md
```

## Running trainers

Use the project training script for SB3/PPO experiments. Example (inside the venv):

```powershell
& .\.venv\Scripts\python.exe geometry_dash_ai_v1/geometry_dash_project/ai_training_with_full_game.py --timesteps 2000000 --n-envs 8 --tb-run-name my_run
```

To stop training after wall-clock time use `--train-seconds` (seconds).

## Where artifacts are written

Training artifacts, models and TensorBoard logs are colocated under the project. Primary artifacts live under `geometry_dash_ai_v1/geometry_dash_project/trained_models/` and include:

- `*.zip` — saved SB3 model checkpoints
- `*_vecnormalize.pkl` — VecNormalize state for consistent inference
- `best_model.zip` — canonical copy of the global best (overwritten when a new best is found)
- `best_YYYYMMDD_HHMMSS.zip` — timestamped archives of historical bests
- `best_overall.json` — metadata about the global best (mean reward, archive path)
- `logs/` — TensorBoard event logs

## TensorBoard

Start TensorBoard pointing to the project's `trained_models/logs/` folder (from inside the venv):

```powershell
& .\.venv\Scripts\python.exe -m tensorboard.main --logdir geometry_dash_ai_v1/geometry_dash_project/trained_models/logs --port 6006
```

## Contributing

Contributions are welcome. For small changes (docs, minor fixes) open a PR against `main`. For larger changes please open an issue first so we can coordinate.

## License

This repository does not include an explicit license file. If you plan to publish or share, consider adding an open-source license (MIT / Apache-2.0 recommended).

## Questions or help

If you want me to (a) reformat the README to strictly conform to a specific linter, (b) add CI checks (flake8 / mypy), or (c) populate `best_overall.json` with a current canonical checkpoint, tell me which action to take and I'll do it.
