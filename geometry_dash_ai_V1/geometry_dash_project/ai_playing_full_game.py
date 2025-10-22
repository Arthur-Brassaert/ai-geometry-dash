import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from geometry_dash_env import GeometryDashEnv
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model zip')
    parser.add_argument('--loop', action='store_true', help='Loop playback')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_dir, 'trained_models')
    if args.model_path is None:
        # prefer the canonical best_model.zip if present
        preferred = os.path.join(models_dir, 'best_model.zip')
        if os.path.exists(preferred):
            args.model_path = preferred
        else:
            # fallback to an auto-selected latest model
            args.model_path = os.path.join(models_dir, 'ai_full_game.zip')

    if not os.path.exists(args.model_path):
        # Try to pick the newest .zip in trained_models
        if os.path.isdir(models_dir):
            candidates = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.zip')]
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                args.model_path = candidates[0]
                print('Auto-selected latest model:', args.model_path)
            else:
                raise FileNotFoundError(f'Model not found: {args.model_path} and no .zip models in {models_dir}')
        else:
            raise FileNotFoundError(f'Model not found: {args.model_path}')

    print('Loading model:', args.model_path)
    model = PPO.load(args.model_path)

    # Create a single environment for rendering
    env = GeometryDashEnv(headless=False)

    # Attempt to load VecNormalize stats if present (same basename as model)
    vecnorm_path = os.path.join(models_dir, os.path.basename(args.model_path).replace('.zip', '_vecnormalize.pkl'))
    use_vecnorm = False
    vec_env = None
    if os.path.exists(vecnorm_path):
        try:
            # Wrap real env in DummyVecEnv so we can apply VecNormalize and use its normalize_obs
            vec_env = DummyVecEnv([lambda: GeometryDashEnv(headless=False)])
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            use_vecnorm = True
            print('Loaded VecNormalize stats from', vecnorm_path)
        except Exception as e:
            print('Warning: failed to load VecNormalize stats:', e)

    try:
        while True:
            obs, _ = env.reset()
            done = False
            while not done:
                if use_vecnorm and vec_env is not None:
                    # VecNormalize expects batched observations
                    obs_norm = vec_env.normalize_obs(np.asarray(obs)[None])
                    action, _ = model.predict(obs_norm, deterministic=True)
                    action = int(action[0])
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(int(action))
                env.render()
                done = terminated or truncated
            if not args.loop:
                break
    finally:
        env.close()


if __name__ == '__main__':
    main()
