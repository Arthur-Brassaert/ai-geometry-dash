import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.geometry_dash_headless import HeadlessGeometryDashEnv  # onze headless env

# --- Hyperparameters ---
NUM_ENVS = 8               # aantal parallelle omgevingen
TOTAL_TIMESTEPS = 500_000  # totale timesteps
LR = 0.0005
GAMMA = 0.99

# --- Device ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on device:", DEVICE)

# --- Maak vectorized env ---
def make_env():
    def _init():
        env = HeadlessGeometryDashEnv()
        return Monitor(env)  # Monitor voor logging
    return _init

env = DummyVecEnv([make_env() for _ in range(NUM_ENVS)])

# --- Maak model ---
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="tensorboard_log",
    learning_rate=LR,
    gamma=GAMMA,
    device=DEVICE,
)

# --- Checkpoints folder ---
os.makedirs("models", exist_ok=True)

# --- Callback voor checkpoints elke 50k stappen ---
from stable_baselines3.common.callbacks import BaseCallback

class SaveCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = f"{self.save_path}/dqn_checkpoint_{self.num_timesteps}.zip"
            self.model.save(path)
            if self.verbose:
                print(f"Checkpoint saved: {path}")
        return True

callback = SaveCallback(save_freq=50_000, save_path="models")

# --- Train ---
# Provide a tb_log_name so tensorboard creates a subfolder per run
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, tb_log_name="run1")

# --- Save final model ---
model.save("models/final_model")
print("Training finished, model saved in models/final_model.zip")
