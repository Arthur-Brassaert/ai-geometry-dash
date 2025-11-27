import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path
from geometry_dash_env import GeometryDashEnv

# ---------------------------------------------------------
# Pad naar model + normalisatie-bestanden
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "best_model" / "best_model.zip"
VECNORM_PATH = SCRIPT_DIR / "best_model" / "vec_normalize.pkl"

# ---------------------------------------------------------
# Environment settings – EXACT hetzelfde als tijdens training
# ---------------------------------------------------------
ENV_PARAMS = dict(
    render_mode="human",
    reward_survival=0.02,
    reward_jump_success=0.5,
    reward_obstacle_avoid=1.0,
    penalty_crash=-20.0,
    penalty_late_jump=-1.0,
    penalty_early_jump=-0.5,
    reward_progress_scale=0.01,
    obs_horizon=300,
    obs_resolution=3,
    random_levels=True
)


# ---------------------------------------------------------
# Laad Model + VecNormalize
# ---------------------------------------------------------
def load_model_and_env():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model niet gevonden: {MODEL_PATH}")

    print(f"📦 Model laden: {MODEL_PATH}")

    # Environment maken (shape = 100)
    base_env = GeometryDashEnv(**ENV_PARAMS)

    env = DummyVecEnv([lambda: base_env])

    # Normalisatie
    if VECNORM_PATH.exists():
        print("📊 VecNormalize laden…")
        env = VecNormalize.load(str(VECNORM_PATH), env)
        env.training = False
        env.norm_reward = False
    else:
        print("⚠️ Geen vec_normalize.pkl gevonden. Je model werkt, maar kan slechter presteren.")

    # Model laden
    model = PPO.load(str(MODEL_PATH), device="cpu")

    return model, env


# ---------------------------------------------------------
# Speelloop
# ---------------------------------------------------------
def run_agent(model, env):
    print("🚀 AI starten…")
    obs = env.reset()
    done = False

    pygame.init()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Render game window
        env.envs[0].render()

        if done:
            obs = env.reset()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    model, env = load_model_and_env()
    run_agent(model, env)


if __name__ == "__main__":
    main()
