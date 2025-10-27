"""Run the best saved PPO model in the same environment used for training.

This script recreates the training-time environment wrappers (DummyVecEnv + VecNormalize)
and runs the model in a window so you can watch the agent play. It draws the agent's
internal `player` and `obstacles` (spikes as triangles) so visuals match training.

Usage: python run_best_model.py [--model path] [--deterministic] [--max-steps N]
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from geometry_dash_env import GeometryDashEnv
import config


def find_vecnormalize(best_model_dir: Path) -> Path | None:
    eval_p = best_model_dir / 'vec_normalize_eval.pkl'
    train_p = best_model_dir / 'vec_normalize.pkl'
    if eval_p.exists():
        return eval_p
    if train_p.exists():
        return train_p
    return None


def make_render_env():
    return GeometryDashEnv(render_mode='human')


def unwrap_inner_env(env):
    inner = env
    try:
        # Drill through common wrappers (VecNormalize, DummyVecEnv)
        while True:
            if hasattr(inner, 'envs') and hasattr(inner, 'envs'):
                # DummyVecEnv exposes .envs list
                inner = inner.envs[0]
                continue
            if hasattr(inner, 'venv'):
                inner = inner.venv
                continue
            break
    except Exception:
        pass
    return inner


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', default=os.path.join('ai_omgeving', 'best_model', 'best_model.zip'))
    p.add_argument('--deterministic', action='store_true', help='Use deterministic actions')
    p.add_argument('--max-steps', type=int, default=0, help='Exit after N environment steps (0 = run forever)')
    p.add_argument('--no-vecnorm', action='store_true', help='Do not load VecNormalize even if present')
    args = p.parse_args()

    model_path = Path(args.model)
    best_model_dir = model_path.parent

    if not model_path.exists():
        print(f"Model not found: {model_path}. Train first or point --model to the zip file.")
        return

    # Build env wrapper like in training (single env)
    env = DummyVecEnv([make_render_env])

    # Load normalization if available (prefer eval file)
    if not args.no_vecnorm:
        vec_path = find_vecnormalize(best_model_dir)
        if vec_path is not None:
            try:
                env = VecNormalize.load(str(vec_path), env)
                env.training = False
                env.norm_reward = False
                print(f"Loaded VecNormalize from {vec_path}")
            except Exception as e:
                print(f"Failed to load VecNormalize from {vec_path}: {e}")

    # Load model
    try:
        model = PPO.load(str(model_path))
        # attach env so callbacks or predict consistency is maintained
        try:
            model.set_env(env)
        except Exception:
            pass
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return

    # init pygame for drawing
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    # Reset env and force immediate spawn for visuals
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, (tuple, list)) and len(reset_result) > 0 else reset_result
    inner = unwrap_inner_env(env)
    if hasattr(inner, 'spawn_timer'):
        try:
            inner.spawn_timer = 0
            inner.last_group_right_x = -9999
        except Exception:
            pass

    running = True
    steps = 0
    start = time.time()

    while running:
        dt = clock.tick(config.FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Predict action (model expects batch obs from VecEnv)
        action, _ = model.predict(obs, deterministic=args.deterministic)

        # Step
        result = env.step(action)
        # support both Gymnasium 5-tuple and VecEnv 4-tuple
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            try:
                obs, reward, dones, infos = result
                # VecEnv returns arrays for dones/infos when vectorized
                done = bool(dones[0]) if isinstance(dones, (list, tuple, np.ndarray)) else bool(dones)
                info = infos[0] if isinstance(infos, (list, tuple)) and infos else (infos if not isinstance(infos, (list, tuple)) else {})
            except Exception:
                # fallback
                obs = result[0]
                done = False
                info = {}

        # Draw frame: simple sky, ground, player, obstacles (triangles for spikes)
        screen.fill((135, 206, 235))  # sky
        pygame.draw.rect(screen, (80, 80, 80), (0, config.HEIGHT - 80, config.WIDTH, 80))

        render_env = unwrap_inner_env(env)
        # draw player
        player = getattr(render_env, 'player', None)
        if player is not None:
            try:
                pygame.draw.rect(screen, (200, 30, 30), (int(player.x), int(player.y), int(player.w), int(player.h)))
            except Exception:
                pass

        # draw obstacles (match game: draw all obstacles as isosceles triangles, textured when available)
        for o in getattr(render_env, 'obstacles', []):
            ox = int(o.x)
            oy = int(o.y)
            ow = int(getattr(o, 'w', 0))
            oh = int(getattr(o, 'h', 0))
            # textured triangle if available
            tex = getattr(o, 'tex', None)
            if tex is not None:
                try:
                    screen.blit(tex, (ox, oy))
                    # outline
                    p1 = (ox, oy + oh)
                    p2 = (ox + ow, oy + oh)
                    p3 = (ox + ow // 2, oy)
                    pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=5)
                    continue
                except Exception:
                    pass
            # otherwise draw plain triangle (spike or normal)
            p1 = (ox, oy + oh)
            p2 = (ox + ow, oy + oh)
            p3 = (ox + ow // 2, oy)
            pygame.draw.polygon(screen, (120, 120, 120), [p1, p2, p3])
            pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=5)

        # HUD
        score = getattr(render_env, 'score', info.get('score', 0))
        fps_text = font.render(f"Score: {score}  Steps: {steps}", True, (0, 0, 0))
        screen.blit(fps_text, (8, 8))

        pygame.display.flip()

        steps += 1
        if args.max_steps and steps >= args.max_steps:
            print(f"Reached max steps={args.max_steps}, exiting")
            break

    try:
        env.close()
    except Exception:
        pass
    pygame.quit()


if __name__ == '__main__':
    main()
