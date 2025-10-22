import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from geometry_dash_env import GeometryDashEnv
import os
from logging_config import get_tb_log_root
from datetime import datetime


def select_cuda_device(preferred_substrs=('rtx', 'nvidia'), explicit_index=None, explicit_name=None, prefer_largest_mem=True):
    """Return an integer CUDA device index or None.

    Selection order:
    - If explicit_index is provided and valid, use it.
    - If explicit_name provided, pick the first device whose name contains that substring.
    - Otherwise pick the first device matching any preferred_substrs.
    - Fallback: pick device with largest total memory (if prefer_largest_mem).
    - If no CUDA devices available or force CPU, return None.
    """
    if explicit_index is not None:
        if not torch.cuda.is_available():
            return None
        if 0 <= explicit_index < torch.cuda.device_count():
            return explicit_index
        else:
            raise ValueError(f"Requested CUDA index {explicit_index} is out of range")

    if not torch.cuda.is_available():
        return None

    count = torch.cuda.device_count()

    # collect device info (index, name, memory)
    devices = []
    for i in range(count):
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = '<unknown>'
        try:
            mem = torch.cuda.get_device_properties(i).total_memory
        except Exception:
            mem = 0
        devices.append((i, name, mem))

    print('Available CUDA devices:')
    for i, name, mem in devices:
        print(f'  {i}: {name} ({mem / 1024**3:.1f} GB)')

    if explicit_name:
        for i, name, mem in devices:
            if explicit_name.lower() in name.lower():
                print(f"Selecting CUDA device {i}: {name} (matched name '{explicit_name}')")
                return i

    # prefer substr matches
    for pref in preferred_substrs:
        for i, name, mem in devices:
            if pref.lower() in name.lower():
                print(f"Selecting CUDA device {i}: {name} (matched '{pref}')")
                return i

    # fallback: choose largest memory device
    if prefer_largest_mem:
        best = max(devices, key=lambda x: x[2])
        print(f"No preferred device found; selecting largest GPU {best[0]}: {best[1]}")
        return best[0]

    return None


parser = argparse.ArgumentParser()
parser.add_argument('--force-cpu', action='store_true', help='Force CPU even if CUDA is available')
parser.add_argument('--cuda-index', type=int, default=None, help='Explicit CUDA device index to use (overrides auto-detection)')
parser.add_argument('--cuda-name', type=str, default=None, help='Match a substring of the CUDA device name to select a device')
args = parser.parse_args()

parser.add_argument('--timesteps', type=int, default=5_000_000, help='Total timesteps to train for (use small number for smoke tests)')
args = parser.parse_args()

cuda_index = None
if not args.force_cpu:
    try:
        cuda_index = select_cuda_device(explicit_index=args.cuda_index, explicit_name=args.cuda_name)
    except ValueError as e:
        print('CUDA selection error:', e)
        cuda_index = None

if cuda_index is None:
    device_str = 'cpu'
else:
    device_str = f'cuda:{cuda_index}'

print(f"🎮 Training on device: {device_str}")
if device_str.startswith('cuda'):
    try:
        idx = cuda_index if cuda_index is not None else 0
        print(f"Device name: {torch.cuda.get_device_name(idx)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(idx).total_memory / 1024**3:.1f} GB")
        # set torch default device for this process
        try:
            torch.cuda.set_device(idx)
        except Exception:
            pass
    except Exception:
        pass

# Project-local trained_models folder (absolute) and canonical tensorboard root
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
trained_models_dir = os.path.join(PROJECT_DIR, 'trained_models')
os.makedirs(trained_models_dir, exist_ok=True)
tensorboard_root = get_tb_log_root()

# Create environment
def make_env():
    def _init():
        env = GeometryDashEnv(headless=True)
        return Monitor(env)
    return _init

# Use multiple environments for faster training
env = DummyVecEnv([make_env() for _ in range(4)])

# Create model optimized for RTX 3050
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,        # Steps per update
    batch_size=128,      # Larger batches for GPU
    n_epochs=10,         # Training epochs per update
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,     # GAE parameter
    clip_range=0.2,      # Clipping parameter
    ent_coef=0.01,       # Encourage exploration
    vf_coef=0.5,         # Value function coefficient
    max_grad_norm=0.5,   # Gradient clipping
    tensorboard_log=tensorboard_root,
    policy_kwargs=dict(
        net_arch=[256, 256]  # Network architecture
    ),
    device=device_str,       # Force GPU usage (e.g. 'cpu' or 'cuda:0')
    verbose=1
)

# Callback to save best model
class NotifyingEvalCallback(EvalCallback):
    """EvalCallback that prints/logs a notice when evaluation improves."""
    def __init__(self, *args, notify_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_best = self.best_mean_reward if hasattr(self, 'best_mean_reward') else -float('inf')
        self.notify_file = notify_file

    def _on_step(self) -> bool:
        # store previous best, let parent update it and save model
        prev = getattr(self, 'best_mean_reward', -float('inf'))
        result = super()._on_step()
        new = getattr(self, 'best_mean_reward', prev)
        try:
            if new is not None and new > prev:
                msg = f"🔔 Improvement detected: mean reward {new:.3f} (was {prev:.3f}) at {datetime.utcnow().isoformat()}Z"
                print(msg)
                # terminal bell
                print('\a')
                if self.notify_file:
                    try:
                        with open(self.notify_file, 'a', encoding='utf-8') as f:
                            f.write(msg + '\n')
                    except Exception:
                        pass
        except Exception:
            pass
        return result


eval_callback = NotifyingEvalCallback(
    env,
    best_model_save_path=trained_models_dir,
    log_path=tensorboard_root,
    eval_freq=5000,
    deterministic=True,
    render=False,
    notify_file=os.path.join(trained_models_dir, 'last_improvements.log')
)

print("🚀 Starting RTX 3050 Training!")
print("💡 Expected speed: 800-1500 steps/second (5-10x faster than CPU!)")

# Debug: show where model params live and CUDA memory usage
try:
    import torch
    if device_str.startswith('cuda') and torch.cuda.is_available():
        print('torch.cuda.memory_allocated:', torch.cuda.memory_allocated())
        print('torch.cuda.memory_reserved:', torch.cuda.memory_reserved())
except Exception:
    pass

# Train the model
model.learn(
    total_timesteps=5_000_000,  # Start with 5 million steps
    callback=eval_callback,
    tb_log_name="RTX_3050_Training"
)

# Save final model
model.save(os.path.join(trained_models_dir, "geometry_dash_rtx_3050_final"))
print("✅ Training completed! Model saved.")

env.close()