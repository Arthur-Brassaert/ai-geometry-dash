import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from geometry_dash_env import GeometryDashEnv


def evaluate(model_path=None, n_episodes=20, headless=True):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_dir, 'trained_models')
    if model_path is None:
        # pick newest zip
        candidates = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.zip')]
        if not candidates:
            raise FileNotFoundError('No .zip models in ' + models_dir)
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        model_path = candidates[0]
        print('Auto-selected model:', model_path)

    print('Loading model:', model_path)
    model = PPO.load(model_path)

    # Try to load VecNormalize if available
    vecnorm_path = os.path.join(models_dir, os.path.basename(model_path).replace('.zip', '_vecnormalize.pkl'))
    use_vecnorm = False
    vec_env = None
    if os.path.exists(vecnorm_path):
        try:
            vec_env = DummyVecEnv([lambda: GeometryDashEnv(headless=headless)])
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            use_vecnorm = True
            print('Loaded VecNormalize stats from', vecnorm_path)
        except Exception as e:
            print('Warning: failed to load VecNormalize stats:', e)

    rewards = []
    for ep in range(n_episodes):
        if use_vecnorm:
            obs = vec_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = vec_env.step(action)
                ep_reward += float(reward[0])
                done = bool(terminated[0] or truncated[0])
            rewards.append(ep_reward)
        else:
            env = GeometryDashEnv(headless=headless)
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += float(reward)
                done = terminated or truncated
            env.close()
            rewards.append(ep_reward)
        print(f'Episode {ep+1}/{n_episodes} reward: {rewards[-1]:.3f}')

    rewards = np.array(rewards)
    print('\nEvaluation results:')
    print(f'  episodes: {len(rewards)}')
    print(f'  mean reward: {rewards.mean():.3f}')
    print(f'  std reward: {rewards.std():.3f}')
    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    evaluate(args.model, args.episodes, headless=True)
