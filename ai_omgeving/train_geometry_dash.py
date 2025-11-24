import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from geometry_dash_env import GeometryDashEnv
import os
from datetime import datetime
from pathlib import Path
import torch
from tqdm import tqdm

# ------------------------------------
# Reward Parameters
# ------------------------------------
REWARD_SURVIVAL = 1.0
REWARD_JUMP_SUCCESS = 10.0
REWARD_OBSTACLE_AVOID = 5.0
PENALTY_CRASH = -50.0
PENALTY_LATE_JUMP = -20.0
PENALTY_EARLY_JUMP = -10.0
REWARD_PROGRESS = 0.001

# ------------------------------------
# Observation Parameters
# ------------------------------------
OBS_HORIZON = 200
OBS_RESOLUTION = 4

# ------------------------------------
# PPO Parameters
# ------------------------------------
TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS = 16
LEARNING_RATE = 3e-4
N_STEPS = 4096
EVAL_FREQ = 5_000
CHECKPOINT_FREQ = 50_000

# ------------------------------------
# Directories
# ------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_MODELS_DIR = SCRIPT_DIR / 'models'
BEST_MODEL_DIR = SCRIPT_DIR / 'best_model'
SAVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_LOG_DIR = SCRIPT_DIR / 'gd_tensorboard'
BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)

safe_ts = datetime.now().strftime('%H-%M__%d-%m-%y')
base_folder_name = f"(gd_ppo) {safe_ts}"
folder_name = base_folder_name
counter = 1
log_dir = BASE_LOG_DIR / folder_name
while log_dir.exists():
    folder_name = f"{base_folder_name}_{counter}"
    log_dir = BASE_LOG_DIR / folder_name
    counter += 1

# ------------------------------------
# Stable Progress Bar Callback
# ------------------------------------
class StableProgressBar(BaseCallback):
    """Progress bar updates enkel bij elke 1% vooruitgang."""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_pct = 0

    def _on_training_start(self):
        self.pbar = tqdm(
            total=100,
            desc="Training Progress",
            ncols=100,
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}% • ETA: {remaining}"
        )

    def _on_step(self):
        steps_done = self.model.num_timesteps
        pct = int((steps_done / self.total_timesteps) * 100)
        if pct > self.last_pct:
            self.pbar.update(pct - self.last_pct)
            self.last_pct = pct
        return True

    def _on_training_end(self):
        self.pbar.n = 100
        self.pbar.refresh()
        self.pbar.close()


class EvalOnceCallback(BaseCallback):
    """Toont enkel een regel bij evaluatie, niet constant."""
    def __init__(self, eval_callback, verbose=1):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.last_mean = None

    def _on_step(self):
        last_mean = getattr(self.eval_callback, 'last_mean_reward', None)
        if last_mean is not None and last_mean != self.last_mean:
            self.last_mean = last_mean
            print(f" Evaluatie: mean_reward={self.last_mean:.2f}")
        return True

# ------------------------------------
# Environment Factory
# ------------------------------------
def make_env():
    return GeometryDashEnv(
        render_mode=None,
        reward_survival=REWARD_SURVIVAL,
        reward_jump_success=REWARD_JUMP_SUCCESS,
        reward_obstacle_avoid=REWARD_OBSTACLE_AVOID,
        penalty_crash=PENALTY_CRASH,
        penalty_late_jump=PENALTY_LATE_JUMP,
        penalty_early_jump=PENALTY_EARLY_JUMP,
        reward_progress_scale=REWARD_PROGRESS,
        obs_horizon=OBS_HORIZON,
        obs_resolution=OBS_RESOLUTION,
        random_levels=True
    )

# ------------------------------------
# Create Environments
# ------------------------------------
env = VecNormalize(make_vec_env(make_env, n_envs=NUM_ENVS, monitor_dir=str(log_dir)))
eval_env = VecNormalize(make_vec_env(make_env, n_envs=1, monitor_dir=str(log_dir / "eval")))

# ------------------------------------
# CLI for Resume
# ------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume-model', type=str, default=None)
args = parser.parse_args()

resume_model_path = None
if args.resume_model:
    resume_model_path = Path(args.resume_model)
elif args.resume:
    zips = sorted(BEST_MODEL_DIR.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    if zips:
        resume_model_path = zips[0]

# ------------------------------------
# Callbacks
# ------------------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=str(SAVE_MODELS_DIR),
    name_prefix='gd_ppo'
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(BEST_MODEL_DIR),
    log_path=str(log_dir),
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
    verbose=0
)

progress_callback = StableProgressBar(TOTAL_TIMESTEPS)
eval_once_callback = EvalOnceCallback(eval_callback)

# ------------------------------------
# PPO Model Creation / Resume
# ------------------------------------
if resume_model_path is not None and resume_model_path.exists():
    print(f"Resuming model from: {resume_model_path}")
    model = PPO.load(str(resume_model_path))
    model.set_env(env)
    model.policy.set_training_device("cuda" if torch.cuda.is_available() else "cpu")
else:
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        verbose=1,
        tensorboard_log=str(BASE_LOG_DIR),
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs={"net_arch": [256, 256, 128]},
    )

# ------------------------------------
# Training
# ------------------------------------
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback, eval_once_callback, progress_callback],
)

# ------------------------------------
# Save Final
# ------------------------------------
model.save(BEST_MODEL_DIR / "gd_ppo_final_model")
env.save(BEST_MODEL_DIR / "vec_normalize.pkl")
eval_env.save(BEST_MODEL_DIR / "vec_normalize_eval.pkl")

print("Training KLAAR!")
print(f"Beste model in: {BEST_MODEL_DIR}")
print(f"Logs in: {BASE_LOG_DIR / folder_name}")
