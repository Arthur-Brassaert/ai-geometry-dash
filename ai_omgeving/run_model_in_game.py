"""Run a trained PPO model inside the visual game by wrapping `Game` as a Gym env.

This runner:
 - wraps `geometry_dash_game.Game` with a Gym-compatible wrapper that exposes
   the same binary occupancy observation used during training,
 - optionally loads VecNormalize stats saved with training,
 - loads the PPO model and uses it to decide actions each frame,
 - renders a simple visual (triangles/rectangles) so you can watch the agent play.

Usage: python run_model_in_game.py [--model path] [--deterministic] [--max-steps N]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from geometry_dash_game import Game

# Outline thickness (pixels) to match in-game look
TRIANGLE_OUTLINE_WIDTH = 6
BLOCK_OUTLINE_WIDTH = 4


class GameGymEnv(gym.Env):
    """Wrap the programmatic Game to provide the same observation used in training.

    Observations: binary occupancy vector of length OBS_HORIZON/OBS_RESOLUTION
    Actions: Discrete(2) (0=noop, 1=jump)
    """

    metadata = {"render_modes": ["human"], "render_fps": config.FPS}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.game = Game(seed=seed, no_audio=True)
        self.action_space = spaces.Discrete(2)
        obs_len = int(config.OBS_HORIZON / config.OBS_RESOLUTION)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)
        # cached rendering surfaces/font (created lazily)
        self._screen = None
        self._font = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.game = Game(seed=seed, no_audio=True)
        else:
            self.game.reset()
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # produce same binary occupancy vector as GeometryDashEnv._get_obs
        obs = np.zeros(int(config.OBS_HORIZON / config.OBS_RESOLUTION), dtype=np.float32)
        for o in self.game.obstacles:
            if o.x > self.game.player.x and (o.x - self.game.player.x) < config.OBS_HORIZON:
                idx = int((o.x - self.game.player.x) / config.OBS_RESOLUTION)
                if 0 <= idx < len(obs):
                    obs[idx] = 1.0
        return obs

    def step(self, action):
        # apply action to game
        if action == 1:
            self.game.ai_jump()
        else:
            self.game.ai_release()
        # advance one frame
        self.game.update(1.0 / config.FPS)
        obs = self._get_obs()
        reward = self.game.get_reward()
        done = self.game.is_game_over()
        truncated = False
        info = {"score": getattr(self.game, 'score', 0)}
        return obs, reward, done, truncated, info

    def render(self):
        # Use loaded assets when available; fallback to simple shapes
        # Initialize display once and reuse to avoid flicker
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
        screen = self._screen

        # background
        bg_drawn = False
        try:
            if hasattr(self.game, 'bg_images') and getattr(self.game, 'bg_images'):
                # use randomized background index if provided by Game.load_assets
                idx = getattr(self.game, 'bg_choice', None)
                try:
                    if idx is None or not (0 <= int(idx) < len(self.game.bg_images)):
                        idx = 0
                except Exception:
                    idx = 0
                try:
                    bg = self.game.bg_images[int(idx)]
                    screen.blit(bg, (0, 0))
                    bg_drawn = True
                except Exception:
                    bg_drawn = False
        except Exception:
            bg_drawn = False

        if not bg_drawn:
            screen.fill((135, 206, 235))

        # ground
        try:
            if hasattr(self.game, 'ground_surf') and self.game.ground_surf is not None:
                gs = self.game.ground_surf
                gw, gh = gs.get_size()
                # tile across the bottom
                y = config.HEIGHT - 80
                for x in range(0, config.WIDTH + gw, gw):
                    screen.blit(gs, (x, y))
            else:
                pygame.draw.rect(screen, (80, 80, 80), (0, config.HEIGHT - 80, config.WIDTH, 80))
        except Exception:
            pygame.draw.rect(screen, (80, 80, 80), (0, config.HEIGHT - 80, config.WIDTH, 80))

        # player
        p = self.game.player
        try:
            if hasattr(self.game, 'block_surf') and self.game.block_surf is not None:
                bs = pygame.transform.scale(self.game.block_surf, (int(p.w), int(p.h)))
                screen.blit(bs, (int(p.x), int(p.y)))
                # draw black outline around block
                try:
                    pygame.draw.rect(
                        screen,
                        (0, 0, 0),
                        (int(p.x), int(p.y), int(p.w), int(p.h)),
                        width=BLOCK_OUTLINE_WIDTH,
                    )
                except Exception:
                    pass
            else:
                pygame.draw.rect(screen, (200, 30, 30), (int(p.x), int(p.y), int(p.w), int(p.h)))
                try:
                    pygame.draw.rect(
                        screen,
                        (0, 0, 0),
                        (int(p.x), int(p.y), int(p.w), int(p.h)),
                        width=BLOCK_OUTLINE_WIDTH,
                    )
                except Exception:
                    pass
        except Exception:
            pygame.draw.rect(screen, (200, 30, 30), (int(p.x), int(p.y), int(p.w), int(p.h)))
            try:
                pygame.draw.rect(
                    screen,
                    (0, 0, 0),
                    (int(p.x), int(p.y), int(p.w), int(p.h)),
                    width=BLOCK_OUTLINE_WIDTH,
                )
            except Exception:
                pass

        # obstacles
        for o in self.game.obstacles:
            ox = int(o.x)
            oy = int(o.y)
            ow = int(o.w)
            oh = int(o.h)
            try:
                if hasattr(self.game, 'spike_surf') and self.game.spike_surf is not None:
                    ss = pygame.transform.scale(self.game.spike_surf, (ow, oh))
                    screen.blit(ss, (ox, oy))
                    # always draw a triangle outline over spikes so they have a clear black border
                    try:
                        p1 = (ox, oy + oh)
                        p2 = (ox + ow, oy + oh)
                        p3 = (ox + ow // 2, oy)
                        pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=TRIANGLE_OUTLINE_WIDTH)
                    except Exception:
                        pass
                else:
                    p1 = (ox, oy + oh)
                    p2 = (ox + ow, oy + oh)
                    p3 = (ox + ow // 2, oy)
                    pygame.draw.polygon(screen, (120, 120, 120), [p1, p2, p3])
                    pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=TRIANGLE_OUTLINE_WIDTH)
            except Exception:
                p1 = (ox, oy + oh)
                p2 = (ox + ow, oy + oh)
                p3 = (ox + ow // 2, oy)
                pygame.draw.polygon(screen, (120, 120, 120), [p1, p2, p3])
                pygame.draw.polygon(screen, (0, 0, 0), [p1, p2, p3], width=TRIANGLE_OUTLINE_WIDTH)

        if self._font is None:
            self._font = pygame.font.SysFont(None, 28)
        font = self._font
        txt = font.render(f"Score: {getattr(self.game,'score',0)}", True, (0,0,0))
        screen.blit(txt, (8,8))
        pygame.display.flip()

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass


def find_vecnormalize(best_model_dir: Path) -> Path | None:
    eval_p = best_model_dir / 'vec_normalize_eval.pkl'
    train_p = best_model_dir / 'vec_normalize.pkl'
    if eval_p.exists():
        return eval_p
    if train_p.exists():
        return train_p
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', default=str(Path(__file__).resolve().parent / 'best_model' / 'best_model.zip'))
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--no-audio', action='store_true', help='Disable music/sfx during playback')
    p.add_argument('--max-steps', type=int, default=0)
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print('Model not found:', model_path)
        return

    # Create wrapped env for VecNormalize compatibility
    def make_env():
        return GameGymEnv()

    env = DummyVecEnv([make_env])

    # Try to load VecNormalize stats if available
    vec_path = find_vecnormalize(model_path.parent)
    if vec_path is not None:
        try:
            env = VecNormalize.load(str(vec_path), env)
            env.training = False
            env.norm_reward = False
            print(f'Loaded VecNormalize from {vec_path}')
        except Exception as e:
            print('Failed to load VecNormalize:', e)

    # Load model
    model = PPO.load(str(model_path))
    try:
        model.set_env(env)
    except Exception:
        pass

    # Prepare clock for framerate control. Initialize pygame (but let env create display)
    pygame.init()
    clock = pygame.time.Clock()

    # Reset
    # Ensure programmatic Game loads assets (backgrounds/music) from repo before reset
    try:
        inner = env
        if hasattr(inner, 'venv'):
            inner = inner.venv
        if hasattr(inner, 'envs'):
            inner = inner.envs[0]
        # call load_assets on the underlying Game instance
        try:
            inner.game.load_assets(use_repo_assets=True, enable_music=(not args.no_audio))
        except Exception:
            pass
        # debug: report what was loaded so user can confirm textures/sounds
        try:
            g = inner.game
            bg_count = len(getattr(g, 'bg_images', []) or [])
            has_ground = getattr(g, 'ground_surf', None) is not None
            has_block = getattr(g, 'block_surf', None) is not None
            has_spike = getattr(g, 'spike_surf', None) is not None
            has_music = getattr(g, 'music', None) is not None
            playlist_len = 0
            current = None
            try:
                if g.music is not None:
                    playlist_len = len(getattr(g.music, 'playlist', []) or [])
                    current = getattr(g.music, 'current_track', None)
            except Exception:
                pass
            # no debug prints
        except Exception:
            pass
    except Exception:
        pass

    reset_ret = env.reset()
    obs = reset_ret[0] if isinstance(reset_ret, (tuple, list)) else reset_ret

    steps = 0
    running = True
    while running:
        dt = clock.tick(config.FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # model expects batch obs
        action, _ = model.predict(obs, deterministic=args.deterministic)
        result = env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, rewards, dones, infos = result
            done = bool(dones[0]) if isinstance(dones, (list,tuple,np.ndarray)) else bool(dones)

        # render via env's render
        try:
            # unwrap DummyVecEnv to call render on inner env
            inner = env
            if hasattr(inner, 'venv'):
                inner = inner.venv
            if hasattr(inner, 'envs'):
                inner = inner.envs[0]
            inner.render()
        except Exception:
            pass

        steps += 1
        if args.max_steps and steps >= args.max_steps:
            print('Reached max steps, exiting')
            break

    try:
        env.close()
    except Exception:
        pass
    pygame.quit()


if __name__ == '__main__':
    main()
