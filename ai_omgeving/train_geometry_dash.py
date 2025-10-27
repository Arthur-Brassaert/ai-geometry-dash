import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from geometry_dash_env import GeometryDashEnv
import os
from datetime import datetime
from pathlib import Path
import torch  # Voor GPU-check

# ------------------------------------
# Reward Parameters (Tune these for better training on random levels)
# ------------------------------------
# REWARD_SURVIVAL: Reward per timestep survived (encourages longer play)
REWARD_SURVIVAL = 1.0
# REWARD_JUMP_SUCCESS: Bonus for successful jumps over obstacles
REWARD_JUMP_SUCCESS = 10.0
# REWARD_OBSTACLE_AVOID: Bonus for avoiding obstacles without jumping
REWARD_OBSTACLE_AVOID = 5.0
# PENALTY_CRASH: Penalty for crashing (make larger to discourage deaths)
PENALTY_CRASH = -50.0
# PENALTY_LATE_JUMP: Penalty for jumping too late
PENALTY_LATE_JUMP = -20.0
# PENALTY_EARLY_JUMP: Penalty for jumping too early
PENALTY_EARLY_JUMP = -10.0
# NEW: REWARD_PROGRESS — small reward for moving forward (distance-based)
REWARD_PROGRESS = 0.001  # Tune: 0.0005–0.005 depending on level length

# ------------------------------------
# Observation Parameters (for forward vision on random levels)
# ------------------------------------
OBS_HORIZON = 200   # Pixels/steps ahead to scan for upcoming obstacles
OBS_RESOLUTION = 4  # Downsample factor for future obstacles

# ------------------------------------
# PPO Training Parameters (Tune these for optimization)
# ------------------------------------
TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS = 16
LEARNING_RATE = 3e-4
N_STEPS = 4096
EVAL_FREQ = 5_000
CHECKPOINT_FREQ = 50_000


# ------------------------------------
# Directories and Setup
# ------------------------------------
# Keep model artifacts inside the ai_omgeving folder (next to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_MODELS_DIR = SCRIPT_DIR / 'models'
BEST_MODEL_DIR = SCRIPT_DIR / 'best_model'
SAVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_LOG_DIR = Path(__file__).resolve().parent / 'gd_tensorboard'
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
# Environment Factory with progress reward
# ------------------------------------
def make_env():
    return GeometryDashEnv(
        render_mode=None,  # No render in training for speed
        reward_survival=REWARD_SURVIVAL,
        reward_jump_success=REWARD_JUMP_SUCCESS,
        reward_obstacle_avoid=REWARD_OBSTACLE_AVOID,
        penalty_crash=PENALTY_CRASH,
        penalty_late_jump=PENALTY_LATE_JUMP,
        penalty_early_jump=PENALTY_EARLY_JUMP,
        reward_progress_scale=REWARD_PROGRESS, 
        obs_horizon=OBS_HORIZON,
        obs_resolution=OBS_RESOLUTION,
        random_levels=True  # Randomized level generation on reset()
    )


def find_vecnormalize_in_best(best_dir: Path) -> Path | None:
    eval_p = best_dir / 'vec_normalize_eval.pkl'
    train_p = best_dir / 'vec_normalize.pkl'
    if eval_p.exists():
        return eval_p
    if train_p.exists():
        return train_p
    return None


def find_latest_model(best_dir: Path) -> Path | None:
    # prefer an explicit best_model.zip at project root
    project_root = Path(__file__).resolve().parent
    cand_root = project_root / 'best_model.zip'
    if cand_root.exists():
        return cand_root
    # otherwise find newest zip in best_dir
    zips = sorted(best_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    if zips:
        return zips[0]
    return None

# ------------------------------------
# Create environments and normalizers
# ------------------------------------
env = VecNormalize(make_vec_env(make_env, n_envs=NUM_ENVS, monitor_dir=str(log_dir)))
eval_env = VecNormalize(make_vec_env(make_env, n_envs=1, monitor_dir=str(log_dir / "eval")))

# ------------------------------------
# CLI: support resuming from an existing best model
# ------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Resume training from latest model in ./best_model')
parser.add_argument('--resume-model', type=str, default=None, help='Path to a specific model zip to resume from')
args = parser.parse_args()

resume_model_path: Path | None = None
if args.resume_model:
    resume_model_path = Path(args.resume_model)
elif args.resume:
    resume_model_path = find_latest_model(BEST_MODEL_DIR)

# If a vec-normalizer exists alongside saved models, load it into env so normalization continues from previous run
if resume_model_path is not None:
    try:
        vec_p = find_vecnormalize_in_best(BEST_MODEL_DIR)
        if vec_p is not None:
            try:
                env = VecNormalize.load(str(vec_p), env)
                env.training = True
                env.norm_reward = True
                print(f"Loaded VecNormalize from {vec_p} (resuming)")
            except Exception as e:
                print(f"Failed to load VecNormalize while resuming: {e}")
    except Exception:
        pass

# ------------------------------------
# Callbacks for evaluation and checkpoints
# ------------------------------------
checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=str(SAVE_MODELS_DIR), name_prefix='gd_ppo')
eval_callback = EvalCallback(eval_env, best_model_save_path=str(BEST_MODEL_DIR), log_path=str(log_dir), eval_freq=EVAL_FREQ, deterministic=True, render=False)

# ------------------------------------
# PPO Model setup
# ------------------------------------
# If requested, try to resume from an existing model; otherwise create a new PPO instance
if resume_model_path is not None and resume_model_path.exists():
    try:
        print(f"Resuming model from: {resume_model_path}")
        model = PPO.load(str(resume_model_path))
        try:
            model.set_env(env)
        except Exception:
            pass
        # ensure device matches current availability
        model.policy.set_training_device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Failed to load resume model ({resume_model_path}): {e}. Creating new model instead.")
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
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, eval_callback], tb_log_name=folder_name)

# ------------------------------------
# Save and Finalize
# ------------------------------------
model.save(os.path.join(BEST_MODEL_DIR, "gd_ppo_final_model"))
env.save(os.path.join(BEST_MODEL_DIR, "vec_normalize.pkl"))
eval_env.save(os.path.join(BEST_MODEL_DIR, "vec_normalize_eval.pkl"))

# ------------------------------------
# Canonicalize Best Model
# ------------------------------------
def _canonicalize_best_model(best_dir: Path, project_root: Path):
    """Find the newest best-model ZIP in best_dir and copy it to project_root/best_model.zip"""
    import shutil, json, time
    if not best_dir.exists():
        print(f"_canonicalize_best_model: best_dir does not exist: {best_dir}")
        return
    zips = sorted(best_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        print("_canonicalize_best_model: geen zip-bestanden gevonden in best_model map.")
        return
    src = zips[0]
    dest = project_root / 'best_model.zip'
    try:
        shutil.copy2(src, dest)
        meta = {'source': str(src.name), 'copied_to': str(dest), 'timestamp': time.time()}
        with open(project_root / 'best_model_meta.json', 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, indent=2)
        print(f"Gecanonicaliseerd beste model: {src.name} -> {dest}")
    except Exception as e:
        print(f"Kon best model niet canonicaliseren: {e}")

# NOTE: canonicalization (copying the newest zip to project_root/best_model.zip)
# was causing duplicate best-model files in two locations. Disabled by default
# to avoid unnecessary duplication of large zip files. If you want to re-enable
# the behavior, uncomment the line below.
# _canonicalize_best_model(BEST_MODEL_DIR, Path(__file__).resolve().parent)
print("Canonicalization disabled: best model is kept in ./best_model/ (no project-root copy).")
print(f"Training KLAAR! Beste model in ./best_model/, logs in {BASE_LOG_DIR / folder_name}")
