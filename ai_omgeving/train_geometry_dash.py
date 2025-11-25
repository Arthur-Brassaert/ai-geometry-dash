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
import time
from tqdm import tqdm

# ------------------------------------
# Reward Parameters
# ------------------------------------
REWARD_SURVIVAL = 0.02
REWARD_JUMP_SUCCESS = 0.5
REWARD_OBSTACLE_AVOID = 1.0
PENALTY_CRASH = -20.0
PENALTY_LATE_JUMP = -1.0
PENALTY_EARLY_JUMP = -0.5
REWARD_PROGRESS = 0.01

# ------------------------------------
# Observation Parameters
# ------------------------------------
OBS_HORIZON = 300
OBS_RESOLUTION = 3

# ------------------------------------
# PPO Parameters
# ------------------------------------
TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS = 16
LEARNING_RATE = 2e-4
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

# Create unique tensorboard run dir
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
# Advanced Progress Bar Callback
# ------------------------------------
class AdvancedProgressBarCallback(BaseCallback):
    """Voortgangsbalk met realtime eindtijd en evaluatie netjes naast elkaar."""
    def __init__(self, total_timesteps, eval_callback=None, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.last_percent = 0
        self.eval_callback = eval_callback

    def _on_training_start(self):
        self.start_time = time.time()
        self.pbar = tqdm(
            total=100,
            desc="Training Progress",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}% • {postfix}",
            leave=True
        )

    def _on_step(self):
        steps_done = self.model.num_timesteps
        percent_done = int((steps_done / self.total_timesteps) * 100)
        if percent_done > self.last_percent:
            self.last_percent = percent_done

            elapsed = time.time() - self.start_time
            frac_done = steps_done / self.total_timesteps
            if frac_done > 0:
                end_timestamp = self.start_time + elapsed / frac_done
                end_time_str = datetime.fromtimestamp(end_timestamp).strftime("%H:%M")
            else:
                end_time_str = "--:--"

            eval_str = ""
            if self.eval_callback is not None:
                mean_reward = getattr(self.eval_callback, "last_mean_reward", None)
                if mean_reward is not None:
                    eval_str = f"Evaluatie: mean_reward={mean_reward:.2f}"

            postfix = f"Eindtijd: {end_time_str}"
            if eval_str:
                postfix += "  " + eval_str

            self.pbar.n = percent_done
            self.pbar.set_postfix_str(postfix)
            self.pbar.refresh()
        return True

    def _on_training_end(self):
        self.pbar.n = 100
        self.pbar.refresh()
        self.pbar.close()


# ------------------------------------
# Env Factory
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
# Helpers to load previous best
# ------------------------------------
def find_vecnormalize_in_best(best_dir: Path) -> Path | None:
    eval_p = best_dir / 'vec_normalize_eval.pkl'
    train_p = best_dir / 'vec_normalize.pkl'
    if eval_p.exists():
        return eval_p
    if train_p.exists():
        return train_p
    return None


def find_latest_model(best_dir: Path) -> Path | None:
    candidate = best_dir / 'best_model.zip'
    if candidate.exists():
        return candidate

    zips = sorted(best_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


# ------------------------------------
# Create environments
# ------------------------------------
env = VecNormalize(make_vec_env(make_env, n_envs=NUM_ENVS, monitor_dir=str(log_dir)))
eval_env = VecNormalize(make_vec_env(make_env, n_envs=1, monitor_dir=str(log_dir / "eval")))


# ------------------------------------
# AUTOMATIC RESUME LOGIC
# ------------------------------------
resume_model_path = find_latest_model(BEST_MODEL_DIR)

if resume_model_path is not None and resume_model_path.exists():
    print(f"Automatisch hervatten vanaf: {resume_model_path}")

    vec_p = find_vecnormalize_in_best(BEST_MODEL_DIR)
    if vec_p is not None:
        try:
            env = VecNormalize.load(str(vec_p), env)
            env.training = True
            env.norm_reward = True
            print(f"Loaded VecNormalize from: {vec_p}")
        except Exception as e:
            print(f"Kon VecNormalize niet laden: {e}")

    try:
        model = PPO.load(str(resume_model_path))
        model.set_env(env)
        model.policy.set_training_device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Kon bestaand model niet laden ({e}) — nieuw model gestart.")
        resume_model_path = None

# Geen bestaand model → nieuwe training
if resume_model_path is None:
    print("Geen bestaand model gevonden — start nieuwe PPO training.")
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

progress_callback = AdvancedProgressBarCallback(
    TOTAL_TIMESTEPS,
    eval_callback=eval_callback
)

# ------------------------------------
# TRAINING
# ------------------------------------
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback, progress_callback],
    tb_log_name=folder_name
)

# ------------------------------------
# SAVE FINAL MODEL + NORMALIZERS
# ------------------------------------
model.save(BEST_MODEL_DIR / "gd_ppo_final_model")
env.save(BEST_MODEL_DIR / "vec_normalize.pkl")
eval_env.save(BEST_MODEL_DIR / "vec_normalize_eval.pkl")

print("Training KLAAR!")
print(f"Beste model in: {BEST_MODEL_DIR}")
print(f"Logs in: {BASE_LOG_DIR / folder_name}")
