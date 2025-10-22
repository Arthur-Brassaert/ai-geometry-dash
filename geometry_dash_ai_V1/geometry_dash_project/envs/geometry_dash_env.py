"""
Gym-compatible environment wrapper for the Geometry Dash project.

Observations: uint8 RGB images, shape (H, W, C) channel-last.
Action space: Discrete(2) -> 0: no-op, 1: hold jump (space pressed)

This wrapper runs the game for one step per env.step() call and returns
frame-skipped, frame-stacked images suitable for Stable-Baselines3 CNN policies.

Requires pygame to be available and the top-level `geometry_dash_game` to
export a `main()` compatible API or a headless `Game` object. This wrapper
provides a minimal compatibility layer that uses the existing procedural
game loop rendering code by driving it frame-by-frame.
"""
from __future__ import annotations

import numpy as np
import gym
from gym import spaces
import pygame
import typing as t
import time
import os

# Try to import a headless Game-like API; fall back to driving the existing
# geometry_dash_game module by launching it in a headless mode if available.
try:
    # Prefer a Game class if someone created game_core.Game
    from game_core import Game as CoreGame
    _HAS_CORE = True
except Exception:
    CoreGame = None
    _HAS_CORE = False


class GeometryDashEnv(gym.Env):
    """Gym environment that returns RGB pixel observations (H, W, C uint8).

    Simple action mapping:
      0 -> do nothing
      1 -> hold jump (space)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, width: int = 1000, height: int = 600, frame_stack: int = 4, frame_skip: int = 1, seed: int | None = None):
        super().__init__()
        self.width = width
        self.height = height
        self.frame_stack = frame_stack
        self.frame_skip = max(1, int(frame_skip))

        # Action: 0 = noop, 1 = jump/hold
        self.action_space = spaces.Discrete(2)

        # Observations: stacked uint8 images channel-last (H, W, C*stack)
        self.observation_space = spaces.Box(0, 255, shape=(self.height, self.width, 3 * self.frame_stack), dtype=np.uint8)

        # internal state
        self._frames: t.List[np.ndarray] = []
        self._rng = np.random.RandomState(seed)

        # create or attach to a headless game instance
        if _HAS_CORE and CoreGame is not None:
            self.game = CoreGame(headless=True, width=self.width, height=self.height, seed=seed)
        else:
            # try to import geometry_dash_game and use its rendering by instantiating
            # a lightweight offscreen pygame display; geometry_dash_game has been
            # structured for interactive runs, but we can call its rendering helpers
            # by using a separate minimal runner.
            try:
                import geometry_dash_game as gd
                # We will embed a headless game by reusing the module's Player/Obstacle logic
                # but to keep this wrapper simple we run a small offscreen pygame Surface
                # and call the module-level functions as needed. For now store a reference.
                self._gd = gd
                # initialize pygame display in hidden mode
                os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
                pygame.display.init()
                self._screen = pygame.Surface((self.width, self.height))
                # create a very small dummy clock
                self._clock = pygame.time.Clock()
                # create minimal game state by calling the module to construct objects
                # We will reuse the module's Player and Obstacle classes to mimic behavior
                self._player = gd.Player()
                self._obstacles: t.List[gd.Obstacle] = []
                # copy some constants
                self._FPS = getattr(gd, 'FPS', 60)
                self._dt = 1.0 / float(self._FPS)
                # track spawn timer and speed similar to the original game
                self._spawn_timer = 0
                self._base_speed = getattr(gd, 'BASE_SPEED', 6.0)
                self._speed = self._base_speed
                self._elapsed = 0.0
                self._level = 0
                self._score = 0
                self._game_over = False
            except Exception as e:
                raise RuntimeError('No suitable game core found and failed to import geometry_dash_game: ' + str(e))

        # initialize frame buffer with blank frames
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._frames = [blank.copy() for _ in range(self.frame_stack)]

    def seed(self, seed: int | None = None):
        self._rng = np.random.RandomState(seed)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)

        # reset underlying game
        if _HAS_CORE and CoreGame is not None:
            self.game.reset()
        else:
            # reset minimal state
            self._player = self._gd.Player()
            self._obstacles = []
            self._spawn_timer = 0
            self._speed = self._base_speed
            self._elapsed = 0.0
            self._level = 0
            self._score = 0
            self._game_over = False

        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._frames = [blank.copy() for _ in range(self.frame_stack)]
        return self._get_observation()

    def step(self, action: int):
        # action: 1 means hold jump
        reward = 0.0
        done = False
        info = {}

        # apply action over frame_skip frames
        for _ in range(self.frame_skip):
            if _HAS_CORE and CoreGame is not None:
                # translate action into core game controls
                a = {'jump_held': bool(action == 1)}
                obs_rgb, r, d, inf = self.game.step(a)
                reward += r
                done = done or d
                info.update(inf or {})
                frame = obs_rgb
            else:
                # emulate: set jump_held and advance minimal physics
                self._player.jump_held = bool(action == 1)
                self._player.step(self._dt)
                # advance obstacles and spawn
                self._spawn_timer -= 1
                if self._spawn_timer <= 0:
                    # spawn a single obstacle near the right edge for training simplicity
                    x = self.width + 20
                    o = self._gd.Obstacle(x, w=self._player.w, h=self._player.h, kind='spike' if self._rng.rand() < getattr(self._gd, 'SPIKE_CHANCE', 0.25) else 'normal')
                    # prepare spike texture if available
                    if o.kind == 'spike' and hasattr(self._gd, 'spike_surf') and getattr(self._gd, 'spike_surf', None) is not None:
                        try:
                            tri = self._gd.prepare_spike_texture(getattr(self._gd, 'spike_surf', None), o.w, o.h)
                            if tri is not None:
                                o.tex = tri
                        except Exception:
                            pass
                    self._obstacles.append(o)
                    self._spawn_timer = max(8, int(self._FPS * 0.2))

                # move obstacles
                speed_s = self._speed * self._FPS
                for o in list(self._obstacles):
                    o.x -= speed_s * self._dt
                    if o.x + o.w < 0:
                        self._obstacles.remove(o)
                        reward += 1.0

                # collision check simplified
                hit_w = int(self._player.w * getattr(self._gd, 'HITBOX_SCALE', 0.5))
                hit_h = int(self._player.h * getattr(self._gd, 'HITBOX_SCALE', 0.5))
                hit_x = int(self._player.x + (self._player.w - hit_w) / 2)
                hit_y = int(self._player.y + (self._player.h - hit_h) / 2)
                collided = False
                for o in self._obstacles:
                    if hit_x < o.x + o.w and hit_x + hit_w > o.x and hit_y < o.y + o.h and hit_y + hit_h > o.y:
                        collided = True
                        break
                if collided:
                    reward -= 5.0
                    done = True

                # render current frame into RGB array using module renderer where possible
                try:
                    # draw onto the offscreen surface using module rendering code pieces
                    self._screen.fill((0, 0, 0))
                    # reuse many parts of geometry_dash_game drawing; fallback to basic draw
                    # draw ground
                    try:
                        pygame.draw.rect(self._screen, getattr(self._gd, 'GROUND_COLOR', (34,139,34)), (0, self.height-80, self.width, 80))
                    except Exception:
                        pass
                    # draw player
                    pygame.draw.rect(self._screen, getattr(self._gd, 'PLAYER_COLOR', (220,60,60)), (int(self._player.x), int(self._player.y), self._player.w, self._player.h))
                    # draw obstacles
                    for o in self._obstacles:
                        ox = int(o.x)
                        oy = int(o.y)
                        ow = o.w
                        oh = o.h
                        p1 = (ox, oy+oh)
                        p2 = (ox+ow, oy+oh)
                        p3 = (ox+ow//2, oy)
                        pygame.draw.polygon(self._screen, (120,120,120), [p1,p2,p3])
                    # copy to frame
                    arr = pygame.surfarray.array3d(self._screen)
                    # surfarray returns (W,H,C); transpose to H,W,C
                    frame = np.transpose(arr, (1,0,2)).astype(np.uint8)
                except Exception:
                    # fallback blank frame
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # push frame into stack
            self._frames.append(frame)
            if len(self._frames) > self.frame_stack:
                self._frames.pop(0)

        obs = self._get_observation()
        return obs, float(reward), bool(done), info

    def _get_observation(self):
        # concatenate stacked frames along channels
        return np.concatenate(self._frames, axis=2)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._frames[-1]
        elif mode == 'human':
            # show the latest frame in a pygame window
            try:
                if not pygame.display.get_init():
                    pygame.display.init()
                screen = pygame.display.set_mode((self.width, self.height))
                surf = pygame.surfarray.make_surface(np.transpose(self._frames[-1], (1,0,2)))
                screen.blit(surf, (0,0))
                pygame.display.flip()
            except Exception:
                pass

    def close(self):
        try:
            if hasattr(self, '_screen'):
                del self._screen
            pygame.display.quit()
        except Exception:
            pass


def make_env(**kwargs):
    return GeometryDashEnv(**kwargs)
