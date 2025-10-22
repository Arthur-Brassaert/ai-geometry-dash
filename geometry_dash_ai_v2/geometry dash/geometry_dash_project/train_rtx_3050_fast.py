import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from geometry_dash_env import GeometryDashEnv
import os

# Auto-select the best CUDA device (prefer an NVIDIA RTX card if available)
def select_preferred_cuda(preferred_substrs=('rtx', 'nvidia', '3050')):
    if not torch.cuda.is_available():
        return None
    count = torch.cuda.device_count()
    devices = []
    for i in range(count):
        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = '<unknown>'
        devices.append((i, name))

    # print list for debugging
    print('Available CUDA devices:')
    for i, name in devices:
        try:
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f'  {i}: {name} ({mem:.1f} GB)')
        except Exception:
            print(f'  {i}: {name}')

    # pick a device whose name contains any preferred substring
    for pref in preferred_substrs:
        for i, name in devices:
            if pref.lower() in name.lower():
                print(f"Selecting CUDA device {i}: {name} (matched '{pref}')")
                return torch.device(f'cuda:{i}')

    # fallback to device 0
    print('No preferred CUDA device found; using device 0')
    return torch.device('cuda:0')

device = select_preferred_cuda(('rtx 3050', 'rtx', 'nvidia'))
assert device is not None, 'No CUDA device available - check CUDA / PyTorch installation'
# set the CUDA device index for the current process
try:
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device.index)
except Exception:
    # older torch versions may not have device.index populated; ignore
    pass

print(f"🎮 Training on device: {device}")
try:
    idx = device.index if hasattr(device, 'index') and device.index is not None else 0
    print(f"Device name: {torch.cuda.get_device_name(idx)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(idx).total_memory / 1024**3:.1f} GB")
except Exception:
    pass

# Create directories
os.makedirs('./trained_models', exist_ok=True)
os.makedirs('./training_logs', exist_ok=True)

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
    tensorboard_log="./training_logs/",
    policy_kwargs=dict(
        net_arch=[256, 256]  # Network architecture
    ),
    device=device,       # Force GPU usage
    verbose=1
)

# Callback to save best model
eval_callback = EvalCallback(
    env,
    best_model_save_path='./trained_models/',
    log_path='./training_logs/',
    eval_freq=5000,
    deterministic=True,
    render=False
)

print("🚀 Starting RTX 3050 Training!")
print("💡 Expected speed: 800-1500 steps/second (5-10x faster than CPU!)")

# Train the model
model.learn(
    total_timesteps=5_000_000,  # Start with 5 million steps
    callback=eval_callback,
    tb_log_name="RTX_3050_Training"
)

# Save final model
model.save("./trained_models/geometry_dash_rtx_3050_final")
print("✅ Training completed! Model saved.")

env.close()